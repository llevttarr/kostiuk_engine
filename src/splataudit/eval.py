from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from splataudit.data import Scene


# ─── Public API ───────────────────────────────────────────────────────────────


def evaluate(
    cfg: dict,
    scene: "Scene",
    state: dict,
    split: dict[str, list[int]],
) -> list[dict]:
    """
    Render from test camera poses and compute quality metrics.

    Parameters
    ----------
    cfg : dict
        Uses ``cfg["eval"]``:
        metrics, depth_metrics (bool), floater_proxy (bool).
    scene : Scene
    state : dict
        CPU tensors from ``train.train()``:
        means, quats, scales, opacities, sh0, shN, sh_degree.
    split : dict
        Only ``split["test"]`` camera image_ids are evaluated.

    Returns
    -------
    list of per-image result dicts::

        {
          "image_id":          int,
          "psnr":              float,   # dB  ↑ better
          "ssim":              float,   # [0,1] ↑ better
          "lpips":             float,   # [0,∞) ↓ better (VGG)
          "depth_consistency": float,   # floater proxy ↓ better
          "depth_mae":         float,   # DTU only (NaN otherwise)
        }
    """
    from gsplat import rasterization
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ec = cfg.get("eval", {})

    # ── Move model to device ──────────────────────────────────────────────────
    params = {
        k: v.to(device)
        for k, v in state.items()
        if isinstance(v, torch.Tensor)
    }
    sh_degree = int(state["sh_degree"])
    sh_coeffs = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N, K, 3)

    # ── Metric objects ────────────────────────────────────────────────────────
    psnr_fn  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)

    results: list[dict] = []

    for iid in split["test"]:
        cam = scene.get_camera(iid)
        viewmat = torch.tensor(cam.w2c, dtype=torch.float32, device=device).unsqueeze(0)
        K_mat   = torch.tensor(cam.K,   dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            renders, alphas, _ = rasterization(
                means     = params["means"],
                quats     = F.normalize(params["quats"], dim=-1),
                scales    = torch.exp(params["scales"]),
                opacities = torch.sigmoid(params["opacities"]),
                colors    = sh_coeffs,
                viewmats  = viewmat,
                Ks        = K_mat,
                width     = cam.width,
                height    = cam.height,
                sh_degree = sh_degree,
                render_mode = "RGB+D",
                packed    = False,
            )

        render_rgb   = renders[0, :, :, :3].permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
        render_depth = renders[0, :, :,  3]

        gt_img = _load_gt(cam, device)

        row: dict = {
            "image_id": iid,
            "psnr":     float(psnr_fn(render_rgb, gt_img)),
            "ssim":     float(ssim_fn(render_rgb, gt_img)),
            "lpips":    float(lpips_fn(render_rgb * 2 - 1, gt_img * 2 - 1)),
        }

        # ── Floater proxy ─────────────────────────────────────────────────────
        if ec.get("floater_proxy", True):
            row["depth_consistency"] = _floater_proxy(
                params, sh_degree, sh_coeffs, cam, device
            )

        # ── Depth MAE (DTU only) ──────────────────────────────────────────────
        if ec.get("depth_metrics", True) and scene.dataset == "dtu":
            row["depth_mae"] = _depth_mae_dtu(render_depth, cam)
        else:
            row["depth_mae"] = float("nan")

        results.append(row)

        # Reset stateful metrics to avoid accumulation across images
        psnr_fn.reset()
        ssim_fn.reset()
        lpips_fn.reset()

    return results


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_gt(cam, device: torch.device) -> torch.Tensor:
    img = cam.load_image().astype(np.float32) / 255.0
    return torch.tensor(img, device=device).permute(2, 0, 1).unsqueeze(0)


def _floater_proxy(
    params: dict,
    sh_degree: int,
    sh_coeffs: torch.Tensor,
    cam,
    device: torch.device,
    eps: float = 0.05,
) -> float:
    """
    Depth-consistency proxy for floater detection.

    Renders depth from the original camera and from a small lateral
    perturbation (0.05 world units to the right in camera space).
    Returns the fraction of valid pixels where the depth changes by
    more than 20 % relative.

    Interpretation
    --------------
    * Clean geometry → smooth depth maps → low score.
    * Floaters (isolated Gaussians in empty space) → large, incoherent
      depth jumps → high score.
    """
    from gsplat import rasterization

    R = torch.tensor(cam.R, dtype=torch.float32, device=device)
    t = torch.tensor(cam.t, dtype=torch.float32, device=device)
    K_mat = torch.tensor(cam.K, dtype=torch.float32, device=device).unsqueeze(0)

    # Perturb: shift camera centre by eps in camera's right direction
    delta = torch.tensor([eps, 0.0, 0.0], device=device)
    t_pert = t + R @ delta        # world-to-cam translation after right-shift

    def _render_d(t_vec: torch.Tensor) -> torch.Tensor:
        vm = torch.eye(4, device=device)
        vm[:3, :3] = R
        vm[:3,  3] = t_vec
        with torch.no_grad():
            r, _, _ = rasterization(
                means=params["means"], quats=F.normalize(params["quats"], dim=-1),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=sh_coeffs, viewmats=vm.unsqueeze(0), Ks=K_mat,
                width=cam.width, height=cam.height, sh_degree=sh_degree,
                render_mode="D", packed=False,
            )
        return r[0, :, :, 0]  # (H, W)

    d0 = _render_d(t)
    d1 = _render_d(t_pert)

    valid = (d0 > 1e-6) & (d1 > 1e-6)
    if not valid.any():
        return 0.0

    rel_change = (d0 - d1).abs() / (d0.abs() + 1e-6)
    return float((rel_change[valid] > 0.20).float().mean())


def _depth_mae_dtu(render_depth: torch.Tensor, cam) -> float:
    """
    Depth MAE against DTU ground-truth depth (Chamfer proxy).

    DTU provides GT depth via dense point clouds (.ply).
    If the GT depth file is not present alongside the image,
    returns NaN — the runner will report this as missing.
    """
    gt_depth_path = cam.image_path.parent.parent / "depths" / f"{cam.image_path.stem}.npy"
    if not gt_depth_path.exists():
        return float("nan")

    gt = torch.tensor(np.load(str(gt_depth_path)), device=render_depth.device)
    valid = (gt > 0) & (render_depth > 0)
    if not valid.any():
        return float("nan")
    return float((render_depth[valid] - gt[valid]).abs().mean())
