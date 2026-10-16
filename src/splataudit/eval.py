from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from splataudit.splits import distance_bin

if TYPE_CHECKING:
    from splataudit.data import Scene


def evaluate(
    cfg: dict,
    scene: Scene,
    state: dict,
    record: dict,
) -> list[dict]:
    """
    Render every ``eval_all`` view of the split *record* and compute metrics
    (BENCHMARK_PLAN §7).

    ``eval_all`` ⊇ ``test``: rows outside the protocol's test set carry
    ``in_test = False`` and serve the P1 distance-group analysis.

    Metrics per image: PSNR, SSIM (11×11 Gaussian, as 3DGS), LPIPS with both
    VGG and AlexNet.  Renders are clamped to [0, 1] and evaluated at the
    training resolution.  DTU additionally gets RegNeRF's masked metrics:
    PSNR over foreground pixels only, SSIM / LPIPS with the background set
    to white in both images.

    Returns one partial row per image (run keys are added by the runner).
    """
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ec = cfg.get("eval", {})

    params = {k: v.to(device) for k, v in state.items() if isinstance(v, torch.Tensor)}
    sh_degree = int(state["sh_degree"])
    lpips = {
        net: LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True).to(device)
        for net in ec.get("lpips_nets", ["vgg", "alex"])
    }

    test_set = set(record["test"])
    rows: list[dict] = []
    for name in record["eval_all"]:
        cam = scene.by_name(name)
        render, _ = _render(params, sh_degree, cam, device, mode="RGB")
        pred = render[..., :3].permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)  # (1,3,H,W)
        gt = _to_tensor(cam.load_image(), device)

        dist = record["distances"][name]
        row: dict = {
            "test_image": name,
            "in_test": name in test_set,
            "ang_dist_centre": dist["centre_deg"],
            "ang_dist_optical": dist["optical_deg"],
            "dist_bin": distance_bin(dist["centre_deg"]),
            "psnr": float(peak_signal_noise_ratio(pred, gt, data_range=1.0)),
            "ssim": float(structural_similarity_index_measure(pred, gt, data_range=1.0)),
        }
        for net, fn in lpips.items():
            row[f"lpips_{net}"] = float(fn(pred, gt))

        if scene.dataset == "dtu":
            mask = cam.load_mask()
            if mask is None:
                raise FileNotFoundError(f"DTU eval mask missing: {cam.mask_path}")
            m = torch.tensor(mask, device=device, dtype=torch.float32)[None, None]
            fg = m.bool().expand_as(pred)
            row["psnr_masked"] = float(-10.0 * torch.log10(((pred - gt)[fg] ** 2).mean()))
            pred_w, gt_w = pred * m + (1 - m), gt * m + (1 - m)
            row["ssim_masked"] = float(
                structural_similarity_index_measure(pred_w, gt_w, data_range=1.0)
            )
            for net, fn in lpips.items():
                row[f"lpips_{net}_masked"] = float(fn(pred_w, gt_w))

        if ec.get("floater_proxy", False):
            row["floater_proxy"] = _floater_proxy(params, sh_degree, cam, device)
        rows.append(row)
    return rows


def _to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(img.astype(np.float32) / 255.0, device=device).permute(2, 0, 1).unsqueeze(0)


def _render(
    params: dict,
    sh_degree: int,
    cam,
    device: torch.device,
    mode: str,
    viewmat: torch.Tensor | None = None,
):
    from gsplat import rasterization

    if viewmat is None:
        viewmat = torch.tensor(cam.w2c, dtype=torch.float32, device=device)
    with torch.no_grad():
        out, alphas, _ = rasterization(
            means=params["means"],
            quats=F.normalize(params["quats"], dim=-1),
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=torch.cat([params["sh0"], params["shN"]], dim=1),
            viewmats=viewmat.unsqueeze(0),
            Ks=torch.tensor(cam.K, dtype=torch.float32, device=device).unsqueeze(0),
            width=cam.width,
            height=cam.height,
            sh_degree=sh_degree,
            render_mode=mode,
            packed=False,
        )
    return out[0], alphas[0]


def _floater_proxy(
    params: dict,
    sh_degree: int,
    cam,
    device: torch.device,
    rel_eps: float = 0.01,
    rel_thresh: float = 0.20,
) -> float:
    """
    Depth-consistency floater proxy (secondary metric; geometry is not an RQ).

    Shifts the camera centre to the right by ``rel_eps`` × the render's median
    depth, renders expected depth (``"ED"``) from both cameras and returns the
    share of valid pixels whose depth changes by more than ``rel_thresh``.
    Clean geometry changes smoothly under a small shift; floaters jump.
    """
    d0, a0 = _render(params, sh_degree, cam, device, mode="ED")
    d0 = d0[..., 0]
    valid0 = a0[..., 0] > 0.5
    if not valid0.any():
        return float("nan")
    step = rel_eps * float(d0[valid0].median())
    # camera centre C' = C + step·right  ⇒  t' = t − step·e_x (camera frame)
    w2c = torch.tensor(cam.w2c, dtype=torch.float32, device=device)
    w2c[0, 3] -= step
    d1, a1 = _render(params, sh_degree, cam, device, mode="ED", viewmat=w2c)
    valid = valid0 & (a1[..., 0] > 0.5)
    rel = (d0 - d1[..., 0]).abs() / d0.clamp_min(1e-6)
    return float((rel[valid] > rel_thresh).float().mean())
