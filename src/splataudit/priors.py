from __future__ import annotations

from pathlib import Path

import numpy as np

from splataudit.data import Scene


# ─── Constants ────────────────────────────────────────────────────────────────

_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
_CACHE_SUBDIR   = ".depth_cache"   # relative to scene's image parent dir


# ─── Public API ───────────────────────────────────────────────────────────────


def depth_prior(
    scene: Scene,
    view_ids: list[int],
) -> dict[int, np.ndarray]:
    """
    Run Depth Anything V2 on the specified training views.

    Only *view_ids* (train images) are processed — test images are never
    touched, so no held-out information leaks through the depth prior.

    Results are cached as ``.npy`` files in ``<image_dir>/../.depth_cache/``
    so that repeated runs skip inference.  Delete the cache directory to
    force recomputation (e.g. after switching model variants).

    Parameters
    ----------
    scene : Scene
    view_ids : list[int]
        ``image_id`` values of training cameras.

    Returns
    -------
    dict  ``image_id → depth map (H, W) float32``

    Notes
    -----
    * Output is **affine-invariant** (relative depth, not metric).
      FSGS and DNGaussian fit a per-image scale + shift during training.
      The confidence function normalises depth independently for
      cross-view consistency.
    * Model: ``depth-anything/Depth-Anything-V2-Large-hf``.
      Switch to ``Small`` or ``Base`` for faster iteration.

    Dependencies
    ------------
    ``pip install splataudit[priors]``  → ``transformers >= 4.44``
    """
    results: dict[int, np.ndarray] = {}
    missing: list[int] = []

    for iid in view_ids:
        cam  = scene.get_camera(iid)
        path = _cache_path(cam)
        if path.exists():
            results[iid] = np.load(str(path))
        else:
            missing.append(iid)

    if not missing:
        return results

    # ── Lazy-load Depth Anything V2 ───────────────────────────────────────────
    try:
        from transformers import pipeline as hf_pipeline
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError(
            "Depth Anything V2 requires: pip install splataudit[priors]\n"
            "(installs transformers >= 4.44)"
        ) from None

    print(f"[priors] Loading {_DEPTH_MODEL_ID} for {len(missing)} images …")
    pipe = hf_pipeline(
        task="depth-estimation",
        model=_DEPTH_MODEL_ID,
        device=0,    # first GPU; falls back to CPU if unavailable
    )

    for iid in missing:
        cam     = scene.get_camera(iid)
        img_pil = PILImage.fromarray(cam.load_image())
        result  = pipe(img_pil)

        # ``result["depth"]`` is a PIL Image in float mode
        depth = np.array(result["depth"], dtype=np.float32)

        path = _cache_path(cam)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), depth)

        results[iid] = depth

    return results


def confidence(
    scene: Scene,
    train_ids: list[int],
    depths: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """
    Per-pixel geometric reliability from cross-view depth agreement.

    Algorithm
    ---------
    For each training view *i*, every other training view *j* contributes
    a discrepancy estimate:

    1. **Normalise** both depth maps to ``[0, 1]`` with percentile clipping
       (robust to sky / background outliers).
    2. **Project** pixels from view *i* into view *j* using GT camera poses.
       The depth is used only to compute the z-coordinate for projection;
       we use a pseudo-metric scale derived from the scene bounds so that
       the projection is geometrically consistent across views.
    3. **Sample** view *j*'s normalised depth at the projected locations.
    4. **Affine-fit** corrects any residual scale/shift between the two
       normalised depth maps in the overlap region (per-pair robust fitting).
    5. **Residual discrepancy** after affine alignment → confidence via
       ``exp(−mean_disc · σ)`` with ``σ = 5``.

    Returns
    -------
    dict ``image_id → confidence map (H, W) float32 in [0, 1]``

    Notes
    -----
    High confidence → views agree on geometry → reliable depth estimate.
    Low confidence → potential floaters, occlusion boundaries, or
    textureless / sky regions → CrossViewGating suppresses densification.
    """
    near, far = scene.bounds
    depth_scale = far - near          # pseudo-metric scale reference

    results: dict[int, np.ndarray] = {}
    train_cam_map = {iid: scene.get_camera(iid) for iid in train_ids}

    for iid in train_ids:
        cam_i = train_cam_map[iid]
        d_i   = _normalise_depth(depths[iid])
        H, W  = d_i.shape

        disc_sum = np.zeros((H, W), dtype=np.float64)
        n_pairs  = 0

        for jid in train_ids:
            if jid == iid:
                continue
            cam_j = train_cam_map[jid]
            d_j   = _normalise_depth(depths[jid])

            disc = _pairwise_discrepancy(cam_i, d_i, cam_j, d_j, depth_scale)
            disc_sum += disc
            n_pairs  += 1

        mean_disc = disc_sum / max(n_pairs, 1)
        results[iid] = np.exp(-mean_disc * 5.0).astype(np.float32)

    return results


# ─── Private helpers ──────────────────────────────────────────────────────────


def _cache_path(cam) -> Path:
    return cam.image_path.parent.parent / _CACHE_SUBDIR / f"{cam.image_path.stem}.npy"


def _normalise_depth(d: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Percentile-robust normalisation to [0, 1]."""
    p_lo = np.percentile(d, lo)
    p_hi = np.percentile(d, hi)
    if p_hi - p_lo < 1e-6:
        return np.zeros_like(d, dtype=np.float32)
    return ((d - p_lo) / (p_hi - p_lo)).clip(0.0, 1.0).astype(np.float32)


def _pairwise_discrepancy(
    cam_i,
    d_i_norm: np.ndarray,
    cam_j,
    d_j_norm: np.ndarray,
    depth_scale: float,
) -> np.ndarray:
    """
    Warp d_i_norm into cam_j's frame and measure discrepancy with d_j_norm.

    Uses normalised depth as a pseudo-metric (scaled by depth_scale) for
    the 3D projection.  An affine fit (scale + shift) corrects any residual
    normalisation mismatch in the overlap region.

    Returns
    -------
    discrepancy : (H_i, W_i) float64 in [0, ∞)
    """
    import torch
    import torch.nn.functional as F

    H, W = d_i_norm.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64))

    # Pseudo-metric depth (depth_scale * normalised)
    z_i = d_i_norm.astype(np.float64) * depth_scale + 1e-6

    # Unproject pixels from cam_i to world
    x_cam = (u - cam_i.cx) / cam_i.fx * z_i    # (H, W)
    y_cam = (v - cam_i.cy) / cam_i.fy * z_i
    pts_cam_i = np.stack([x_cam, y_cam, z_i], axis=-1).reshape(-1, 3)  # (HW, 3)

    # World coordinates: p_world = R_i^T @ (p_cam - t_i)
    pts_world = (pts_cam_i - cam_i.t) @ cam_i.R   # (HW, 3)

    # Project into cam_j: p_cam_j = R_j @ p_world + t_j
    pts_cam_j = pts_world @ cam_j.R.T + cam_j.t   # (HW, 3)

    z_j = pts_cam_j[:, 2].reshape(H, W)
    u_j = (pts_cam_j[:, 0] / pts_cam_j[:, 2].clip(1e-6) * cam_j.fx + cam_j.cx).reshape(H, W)
    v_j = (pts_cam_j[:, 1] / pts_cam_j[:, 2].clip(1e-6) * cam_j.fy + cam_j.cy).reshape(H, W)

    Hj, Wj = d_j_norm.shape
    in_frustum = (u_j >= 0) & (u_j < Wj) & (v_j >= 0) & (v_j < Hj) & (z_j > 0)

    # Sample d_j at projected locations via bilinear interpolation
    d_j_t = torch.tensor(d_j_norm).unsqueeze(0).unsqueeze(0)   # (1,1,Hj,Wj)
    gx    = torch.tensor(2.0 * u_j / (Wj - 1) - 1.0, dtype=torch.float32)
    gy    = torch.tensor(2.0 * v_j / (Hj - 1) - 1.0, dtype=torch.float32)
    grid  = torch.stack([gx, gy], dim=-1).unsqueeze(0)         # (1,H,W,2)
    d_j_sampled = F.grid_sample(
        d_j_t, grid, align_corners=True, mode="bilinear", padding_mode="border"
    )[0, 0].numpy()                                              # (H, W)

    # Per-pair affine alignment in the overlap region (corrects scale/shift)
    d_proj_norm = (z_j / depth_scale).clip(0.0, 1.0)           # normalise projected depth
    a, b = _affine_fit(d_j_sampled[in_frustum], d_proj_norm[in_frustum])
    d_j_aligned = a * d_j_sampled + b

    # Discrepancy: L1 after affine correction
    disc = np.zeros((H, W), dtype=np.float64)
    disc[in_frustum]  = np.abs(d_proj_norm[in_frustum] - d_j_aligned[in_frustum])
    disc[~in_frustum] = 1.0     # invisible regions → maximum discrepancy

    return disc


def _affine_fit(src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    """
    Robust least-squares fit of (a, b) minimising ||a·src + b − dst||².
    Returns (1.0, 0.0) if fewer than 10 valid samples.
    """
    if len(src) < 10:
        return 1.0, 0.0
    A  = np.stack([src, np.ones_like(src)], axis=1)
    ab, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)
    return float(ab[0]), float(ab[1])
