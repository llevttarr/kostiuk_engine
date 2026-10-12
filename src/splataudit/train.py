from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
from tqdm import trange

if TYPE_CHECKING:
    from splataudit.data import Scene
    from splataudit.methods import Method

# SH DC coefficient: 1 / (2 * sqrt(π))
_C0 = 0.28209479177387814


# ─── Public API ───────────────────────────────────────────────────────────────


def train(
    cfg: dict,
    scene: "Scene",
    train_ids: list[int],
    xyz: np.ndarray,
    rgb: np.ndarray,
    method: "Method",
    depths: dict[int, np.ndarray] | None = None,
    confidence: dict[int, np.ndarray] | None = None,
) -> dict:
    """
    Run the 3DGS training loop via gsplat.

    Parameters
    ----------
    cfg : dict
        Uses ``cfg["train"]``:
        iterations, lambda_dssim, sh_degree, densify_every, densify_until,
        grad_threshold, opacity_reset_every, prune_opacity.
    scene : Scene
    train_ids : list[int]
    xyz : np.ndarray (N, 3) float32
        Initial Gaussian centres (output of ``init.get_init()``).
    rgb : np.ndarray (N, 3) float32
        Initial colours in [0, 1].
    method : Method
        Training strategy (Vanilla / FSGS / DNGaussian / DropGaussian / CrossViewGating).
    depths : dict | None
        ``image_id → (H, W) float32`` depth map.  Required when
        ``method.requires_depth`` is True.
    confidence : dict | None
        ``image_id → (H, W) float32`` confidence map in [0, 1].

    Returns
    -------
    state : dict
        CPU tensors:
        ``means``, ``quats``, ``scales``, ``opacities``, ``sh0``, ``shN``,
        ``sh_degree``, ``n_gaussians``.
    """
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc = cfg["train"]

    # ── Gaussian initialisation ───────────────────────────────────────────────
    params = _init_gaussians(xyz, rgb, tc["sh_degree"], device)
    optimizers = _make_optimizers(params)

    # ── Densification strategy ────────────────────────────────────────────────
    strategy = DefaultStrategy(
        prune_opa=tc["prune_opacity"],
        grow_grad2d=tc["grad_threshold"],
        grow_scale3d=0.01,
        grow_scale2d=0.05,
        prune_scale3d=0.1,
        prune_scale2d=10.0,
        refine_start_iter=tc["densify_every"],
        refine_stop_iter=tc["densify_until"],
        reset_every=tc["opacity_reset_every"],
        absgrad=False,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=float(scene.bounds[1]))

    # ── Move priors to device ─────────────────────────────────────────────────
    d_gpu = {iid: torch.tensor(d, device=device) for iid, d in (depths or {}).items()}
    c_gpu = {iid: torch.tensor(c, device=device) for iid, c in (confidence or {}).items()}

    train_cams = [scene.get_camera(iid) for iid in train_ids]
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    for step in trange(tc["iterations"], desc="Training", dynamic_ncols=True):

        # SH degree increases every 1 000 steps up to the configured maximum
        sh_degree = min(tc["sh_degree"], step // 1000)

        # Random training camera
        cam = random.choice(train_cams)
        viewmat = torch.tensor(cam.w2c, dtype=torch.float32, device=device).unsqueeze(0)
        K_mat   = torch.tensor(cam.K,   dtype=torch.float32, device=device).unsqueeze(0)
        gt_img  = _load_gt(cam, device)          # (1, 3, H, W) float32 [0,1]

        batch = {
            "camera":     cam,
            "gt_image":   gt_img,
            "depth":      d_gpu.get(cam.image_id),
            "confidence": c_gpu.get(cam.image_id),
        }

        # ── Hook: pre_render ──────────────────────────────────────────────────
        rp = method.pre_render(dict(params), step)

        # ── Rasterise ─────────────────────────────────────────────────────────
        renders, alphas, info = rasterization(
            means     = rp["means"],
            quats     = F.normalize(rp["quats"], dim=-1),
            scales    = torch.exp(rp["scales"]),
            opacities = torch.sigmoid(rp["opacities"]),
            colors    = torch.cat([rp["sh0"], rp["shN"]], dim=1),   # (N, K, 3)
            viewmats  = viewmat,
            Ks        = K_mat,
            width     = cam.width,
            height    = cam.height,
            sh_degree = sh_degree,
            render_mode = "RGB+D",
            packed    = False,
        )
        # renders: (1, H, W, 4);  last channel = depth
        render_rgb   = renders[0, :, :, :3].permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)
        render_depth = renders[0, :, :,  3]                                   # (H,W)

        render_out = {"rgb": render_rgb, "depth": render_depth, "alpha": alphas[0]}

        # ── Photometric loss ──────────────────────────────────────────────────
        l1   = F.l1_loss(render_rgb, gt_img)
        lssim = 1.0 - ssim_fn(render_rgb, gt_img)
        loss  = (1.0 - tc["lambda_dssim"]) * l1 + tc["lambda_dssim"] * lssim

        # ── Hook: extra_loss ──────────────────────────────────────────────────
        for lval in method.extra_loss(render_out, batch, step).values():
            loss = loss + lval

        # ── Strategy: pre-backward (sets up grad retention on means2d) ───────
        strategy.step_pre_backward(params, optimizers, strategy_state, info, step)

        loss.backward()

        # ── Hook: densify_mask (gate which Gaussians may be densified) ────────
        # Zero out means2d grads for ineligible Gaussians *before* the
        # strategy reads them for threshold-based clone/split decisions.
        if (
            info.get("means2d") is not None
            and getattr(info["means2d"], "grad", None) is not None
        ):
            dmask = method.densify_mask(info["means2d"].grad, dict(params), step)
            if dmask is not None:
                info["means2d"].grad[~dmask] = 0.0

        # ── Optimiser step ────────────────────────────────────────────────────
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad()

        # Exponential LR decay for means: 1.6e-4 → 1.6e-6 over all iterations
        lr_means = 1.6e-4 * (1.6e-6 / 1.6e-4) ** (step / tc["iterations"])
        _set_lr(optimizers["means"], lr_means)

        # ── Strategy: post-backward (densify + prune) ─────────────────────────
        strategy.step_post_backward(params, optimizers, strategy_state, info, step)

        # ── Hook: post_step ───────────────────────────────────────────────────
        method.post_step(dict(params), step)

    return {k: v.detach().cpu() for k, v in params.items()} | {
        "sh_degree":   tc["sh_degree"],
        "n_gaussians": params["means"].shape[0],
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _init_gaussians(
    xyz: np.ndarray,
    rgb: np.ndarray,
    sh_degree: int,
    device: torch.device,
) -> dict[str, torch.nn.Parameter]:
    """Initialise all 3DGS parameters from a point cloud (xyz, rgb)."""
    N    = len(xyz)
    n_sh = (sh_degree + 1) ** 2

    # Log-scale: mean distance to 3 nearest neighbours, replicated for x/y/z
    tree = KDTree(xyz)
    dists, _ = tree.query(xyz, k=4)                   # (N,4); col 0 = self
    nn_dist = dists[:, 1:].mean(axis=1).clip(min=1e-7)
    log_scales = np.log(nn_dist[:, None].repeat(3, axis=1)).astype(np.float32)

    # SH DC component from RGB: sigmoid(rgb) ≈ C0 * sh0 + 0.5 → sh0 = (rgb−0.5)/C0
    sh0 = ((rgb.clip(0, 1) - 0.5) / _C0).astype(np.float32)[:, None, :]   # (N,1,3)

    def P(arr):
        return torch.nn.Parameter(torch.tensor(arr, dtype=torch.float32, device=device))

    return {
        "means":     P(xyz.astype(np.float32)),
        "quats":     P(np.tile([1., 0., 0., 0.], (N, 1))),
        "scales":    P(log_scales),
        "opacities": P(np.full(N, -2.197, dtype=np.float32)),   # sigmoid⁻¹(0.1)
        "sh0":       P(sh0),
        "shN":       P(np.zeros((N, n_sh - 1, 3), dtype=np.float32)),
    }


def _make_optimizers(params: dict) -> dict[str, torch.optim.Adam]:
    lr = {
        "means":     1.6e-4,
        "quats":     1e-3,
        "scales":    5e-3,
        "opacities": 5e-2,
        "sh0":       1e-3,
        "shN":       1e-3 / 20,   # higher-order SH trained slower
    }
    return {k: torch.optim.Adam([v], lr=lr[k], eps=1e-15) for k, v in params.items()}


def _set_lr(optimizer: torch.optim.Adam, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _load_gt(cam: "Camera", device: torch.device) -> torch.Tensor:  # noqa: F821
    """Load and normalise a ground-truth image → (1, 3, H, W) float32 [0,1]."""
    img = cam.load_image().astype(np.float32) / 255.0
    return torch.tensor(img, device=device).permute(2, 0, 1).unsqueeze(0)
