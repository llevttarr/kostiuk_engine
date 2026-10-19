from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
from tqdm import trange

from splataudit.utils import CameraSampler

if TYPE_CHECKING:
    from splataudit.data import Scene
    from splataudit.methods import Method

# SH DC coefficient: 1 / (2 * sqrt(π))
_C0 = 0.28209479177387814


def train(
    cfg: dict,
    scene: Scene,
    train_ids: list[int],
    xyz: np.ndarray,
    rgb: np.ndarray,
    method: Method,
    depths: dict[int, np.ndarray] | None = None,
    confidence: dict[int, np.ndarray] | None = None,
) -> dict:
    """
    Run the 3DGS training loop via gsplat.

    Parameters
    ----------
    cfg : dict
        Uses ``cfg["seed"]`` and ``cfg["train"]``:
        iterations, lambda_dssim, sh_degree, densify_from, densify_every,
        densify_until, grad_threshold, opacity_reset_every, prune_opacity.
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
    from gsplat.strategy import DefaultStrategy
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc = cfg["train"]

    # 3DGS convention: scene scale = radius of the *training* cameras.
    # Scales the clone/split/prune thresholds and the means learning rate.
    scene_scale = scene.camera_extent(train_ids)

    params = _init_gaussians(xyz, rgb, tc["sh_degree"], device)
    optimizers = _make_optimizers(params, scene_scale)

    strategy = DefaultStrategy(
        prune_opa=tc["prune_opacity"],
        grow_grad2d=tc["grad_threshold"],
        grow_scale3d=0.01,
        grow_scale2d=0.05,
        prune_scale3d=0.1,
        refine_start_iter=tc["densify_from"],
        refine_every=tc["densify_every"],
        refine_stop_iter=tc["densify_until"],
        reset_every=tc["opacity_reset_every"],
        absgrad=False,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    d_gpu = {iid: torch.tensor(d, device=device) for iid, d in (depths or {}).items()}
    c_gpu = {iid: torch.tensor(c, device=device) for iid, c in (confidence or {}).items()}

    # seeded per-epoch permutation: seed s gives every method the same camera order
    sampler = CameraSampler([scene.get_camera(iid) for iid in train_ids], int(cfg["seed"]))
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    import os
    import sys
    import time

    worker = os.environ.get("SPLATAUDIT_WORKER")
    quiet_tqdm = bool(worker) or not sys.stderr.isatty()
    log_every = max(1, tc["iterations"] // 4) if tc["iterations"] >= 1000 else 0
    t_start = time.time()

    for step in trange(tc["iterations"], desc="Training", dynamic_ncols=True, disable=quiet_tqdm):
        # SH degree increases every 1 000 steps up to the configured maximum
        sh_degree = min(tc["sh_degree"], step // 1000)

        cam = sampler.next()
        gt_img = _load_gt(cam, device)  # (1, 3, H, W) float32 [0,1]
        render = _make_render(cam, sh_degree, device)
        act = _activate(params)

        render_out = render(method.pre_render(dict(act), step))
        info = render_out["info"]

        batch = {
            "camera": cam,
            "gt_image": gt_img,
            "act": act,
            "render": render,
            "depth": d_gpu.get(cam.image_id),
            "confidence": c_gpu.get(cam.image_id),
        }

        render_rgb = render_out["rgb"]
        l1 = F.l1_loss(render_rgb, gt_img)
        lssim = 1.0 - ssim_fn(render_rgb, gt_img)
        loss = (1.0 - tc["lambda_dssim"]) * l1 + tc["lambda_dssim"] * lssim

        for lval in method.extra_loss(render_out, batch, step).values():
            loss = loss + lval

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        loss.backward()

        # Zero out means2d grads for ineligible Gaussians *before* the
        # strategy reads them for threshold-based clone/split decisions.
        if info.get("means2d") is not None and getattr(info["means2d"], "grad", None) is not None:
            dmask = method.densify_mask(info["means2d"].grad, dict(params), step)
            if dmask is not None:
                # means2d.grad is (C, N, 2) with packed=False; dmask is (N,)
                info["means2d"].grad[:, ~dmask] = 0.0

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad()

        # Exponential LR decay for means: 1.6e-4 → 1.6e-6 (× scene scale) over all iterations
        lr_means = 1.6e-4 * (1.6e-6 / 1.6e-4) ** (step / tc["iterations"]) * scene_scale
        _set_lr(optimizers["means"], lr_means)

        strategy.step_post_backward(params, optimizers, strategy_state, step, info)

        method.post_step(dict(params), step)

        if quiet_tqdm and log_every and (step + 1) % log_every == 0:
            prefix = f"[gpu={worker}] " if worker is not None else ""
            elapsed = time.time() - t_start
            print(
                f"  {prefix}step {step + 1}/{tc['iterations']} "
                f"loss={float(loss.detach()):.4f} "
                f"gaussians={params['means'].shape[0]} ({elapsed:.0f}s)",
                flush=True,
            )

    return {k: v.detach().cpu() for k, v in params.items()} | {
        "sh_degree": tc["sh_degree"],
        "n_gaussians": params["means"].shape[0],
    }


def _activate(params):
    """Raw parameters -> what gsplat rasterises: unit quats, exp scales, sigmoid opacities, sh colours."""
    return {
        "means": params["means"],
        "quats": F.normalize(params["quats"], dim=-1),
        "scales": torch.exp(params["scales"]),
        "opacities": torch.sigmoid(params["opacities"]),
        "colors": torch.cat([params["sh0"], params["shN"]], dim=1),
    }


def _make_render(cam, sh_degree, device):
    """Renderer for one camera: activated gaussians -> rgb (1,3,H,W), depth (H,W), alpha, gsplat info."""
    from gsplat import rasterization

    viewmat = torch.tensor(cam.w2c, dtype=torch.float32, device=device).unsqueeze(0)
    K_mat = torch.tensor(cam.K, dtype=torch.float32, device=device).unsqueeze(0)

    def render(act, mode="RGB+D"):
        colors = act["colors"]
        renders, alphas, info = rasterization(
            means=act["means"],
            quats=act["quats"],
            scales=act["scales"],
            opacities=act["opacities"],
            colors=colors,
            viewmats=viewmat,
            Ks=K_mat,
            width=cam.width,
            height=cam.height,
            sh_degree=None if colors.dim() == 2 else sh_degree,
            render_mode=mode,
            packed=False,
        )
        out = {
            "rgb": renders[0, :, :, :3].permute(2, 0, 1).unsqueeze(0),
            "alpha": alphas[0],
            "info": info,
        }
        if "D" in mode:
            out["depth"] = renders[0, :, :, 3]
        return out

    return render


def _init_gaussians(
    xyz: np.ndarray,
    rgb: np.ndarray,
    sh_degree: int,
    device: torch.device,
) -> dict[str, torch.nn.Parameter]:
    """Initialise all 3DGS parameters from a point cloud (xyz, rgb)."""
    N = len(xyz)
    n_sh = (sh_degree + 1) ** 2

    # Log-scale: mean distance to 3 nearest neighbours, replicated for x/y/z
    tree = KDTree(xyz)
    dists, _ = tree.query(xyz, k=4)  # (N,4); col 0 = self
    nn_dist = dists[:, 1:].mean(axis=1).clip(min=1e-7)
    log_scales = np.log(nn_dist[:, None].repeat(3, axis=1)).astype(np.float32)

    # SH DC component from RGB: sigmoid(rgb) ≈ C0 * sh0 + 0.5 → sh0 = (rgb−0.5)/C0
    sh0 = ((rgb.clip(0, 1) - 0.5) / _C0).astype(np.float32)[:, None, :]  # (N,1,3)

    def P(arr):
        return torch.nn.Parameter(torch.tensor(arr, dtype=torch.float32, device=device))

    return {
        "means": P(xyz.astype(np.float32)),
        "quats": P(np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))),
        "scales": P(log_scales),
        "opacities": P(np.full(N, -2.197, dtype=np.float32)),  # sigmoid⁻¹(0.1)
        "sh0": P(sh0),
        "shN": P(np.zeros((N, n_sh - 1, 3), dtype=np.float32)),
    }


def _make_optimizers(params: dict, scene_scale: float) -> dict[str, torch.optim.Adam]:
    # 3DGS defaults; position lr is relative to the scene scale
    lr = {
        "means": 1.6e-4 * scene_scale,
        "quats": 1e-3,
        "scales": 5e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,  # higher-order SH trained slower
    }
    return {k: torch.optim.Adam([v], lr=lr[k], eps=1e-15) for k, v in params.items()}


def _set_lr(optimizer: torch.optim.Adam, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _load_gt(cam: Camera, device: torch.device) -> torch.Tensor:  # noqa: F821
    """Load and normalise a ground-truth image → (1, 3, H, W) float32 [0,1]."""
    img = cam.load_image().astype(np.float32) / 255.0
    return torch.tensor(img, device=device).permute(2, 0, 1).unsqueeze(0)
