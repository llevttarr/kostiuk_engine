from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from splataudit.data import Scene


# ─── Public entry point ───────────────────────────────────────────────────────


def colmap_init(
    scene: Scene,
    train_ids: list[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sparse 3D point cloud from COLMAP triangulation on train images only.

    Pipeline
    --------
    1. Extract SIFT features from each train image.
    2. Exhaustive matching (fast for small n_views ≤ 9).
    3. Incremental mapping → sparse reconstruction.
    4. Sim3 (Umeyama) alignment of the resulting point cloud to the
       dataset's GT coordinate system using the known train camera centres.

    Why alignment?
    --------------
    COLMAP produces an arbitrary coordinate system.  The GT camera poses
    used during training live in the dataset's coordinate system
    (e.g. the LLFF poses_bounds.npy frame).  We align the two systems
    using the 3–9 train camera centres that are known in both frames.

    Why NOT use all images?
    -----------------------
    Using test images for COLMAP would leak held-out view information into
    the initial point cloud geometry.  Only *train_ids* are passed here.

    Parameters
    ----------
    scene : Scene
    train_ids : list[int]
    seed : int   (unused; kept for uniform interface)

    Returns
    -------
    xyz : (N, 3) float32 in the GT coordinate system
    rgb : (N, 3) float32 in [0, 1]

    Raises
    ------
    RuntimeError
        If COLMAP fails to register any images (too few features or
        insufficient overlap with n_views=3).

    Dependencies
    ------------
    ``pycolmap >= 3.10``
    """
    import pycolmap

    train_cams = [scene.get_camera(iid) for iid in train_ids]
    image_dir  = train_cams[0].image_path.parent
    image_names = [c.image_path.name for c in train_cams]

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp       = Path(tmp_str)
        db_path   = tmp / "db.db"
        out_path  = tmp / "sparse"
        out_path.mkdir()

        # ── 1. Feature extraction ─────────────────────────────────────────────
        img_list_file = tmp / "image_list.txt"
        img_list_file.write_text("\n".join(image_names))

        pycolmap.extract_features(
            database_path=db_path,
            image_path=image_dir,
            image_list=img_list_file,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model="PINHOLE",
            camera_params=",".join(
                f"{v:.6f}"
                for v in [train_cams[0].fx, train_cams[0].fy,
                           train_cams[0].cx, train_cams[0].cy]
            ),
        )

        # ── 2. Exhaustive matching ────────────────────────────────────────────
        pycolmap.match_exhaustive(database_path=db_path)

        # ── 3. Incremental mapping ────────────────────────────────────────────
        maps = pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=image_dir,
            output_path=out_path,
        )

        if not maps:
            raise RuntimeError(
                f"COLMAP mapping failed for scene '{scene.name}' with "
                f"train_ids={train_ids}.  "
                "This can happen with only 3 views and low texture overlap.  "
                "Try increasing n_views to 9, or switch to 'dust3r' init."
            )

        # Take the largest reconstruction (most registered images)
        reconstruction = max(maps.values(), key=lambda r: r.num_points3D())

        # ── 4. Extract points ─────────────────────────────────────────────────
        xyz, rgb = _extract_points(reconstruction)
        if len(xyz) == 0:
            raise RuntimeError(
                f"COLMAP produced no 3D points for scene '{scene.name}'."
            )

        # ── 5. Sim3 alignment to GT coordinate system ─────────────────────────
        xyz_aligned = _align_to_gt(reconstruction, xyz, train_cams)

    return xyz_aligned.astype(np.float32), rgb.astype(np.float32)


# ─── Private helpers ──────────────────────────────────────────────────────────


def _extract_points(reconstruction) -> tuple[np.ndarray, np.ndarray]:
    """Parse points3D from a pycolmap Reconstruction."""
    pts = reconstruction.points3D
    if not pts:
        return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.float32)
    xyz = np.array([p.xyz   for p in pts.values()], dtype=np.float64)
    rgb = np.array([p.color / 255.0 for p in pts.values()], dtype=np.float32)
    return xyz, rgb


def _align_to_gt(reconstruction, xyz: np.ndarray, train_cams: list) -> np.ndarray:
    """
    Umeyama sim3 alignment of COLMAP points to the GT coordinate system.

    Correspondence: train image names appear in both the COLMAP reconstruction
    (``reconstruction.images``) and ``train_cams``.

    Returns
    -------
    xyz_aligned : (N, 3) float64 in GT coordinate system
    """
    # Camera centre in COLMAP frame: C = -R^T @ t
    colmap_centers: dict[str, np.ndarray] = {}
    for img in reconstruction.images.values():
        R = img.cam_from_world.rotation.matrix()
        t = img.cam_from_world.translation
        colmap_centers[img.name] = -(R.T @ t)

    # Camera centre in GT frame (from our Scene)
    gt_centers = {cam.image_path.name: cam.center for cam in train_cams}

    # Paired arrays (only images registered in both systems)
    pairs = [
        (colmap_centers[name], gt_centers[name])
        for name in colmap_centers
        if name in gt_centers
    ]

    if len(pairs) < 3:
        # Not enough correspondences for reliable sim3; return unaligned.
        # This should rarely happen (all train images should be registered).
        import warnings
        warnings.warn(
            f"Only {len(pairs)} image pairs for Umeyama alignment "
            "(need ≥ 3).  Returning unaligned COLMAP points.",
            RuntimeWarning,
        )
        return xyz

    src = np.array([p[0] for p in pairs])   # (M, 3) COLMAP centres
    dst = np.array([p[1] for p in pairs])   # (M, 3) GT centres

    s, R_um, t_um = _umeyama(src, dst)
    return (s * (R_um @ xyz.T) + t_um[:, None]).T


def _umeyama(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama (1991) similarity (sim3) alignment.

    Finds scale *s*, rotation *R*, translation *t* minimising::

        Σᵢ ‖dst_i − (s · R @ src_i + t)‖²

    Parameters
    ----------
    src, dst : (N, 3) float64

    Returns
    -------
    s  : float
    R  : (3, 3) rotation matrix
    t  : (3,)   translation vector
    """
    N = len(src)
    mu_s = src.mean(0)
    mu_d = dst.mean(0)

    src_c = src - mu_s
    dst_c = dst - mu_d

    var_s = (src_c ** 2).sum() / N
    H = (src_c.T @ dst_c) / N
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d  = np.sign(np.linalg.det(Vt.T @ U.T))
    D  = np.diag([1.0, 1.0, d])

    R  = Vt.T @ D @ U.T
    s  = float((S * D.diagonal()).sum() / var_s) if var_s > 1e-12 else 1.0
    t  = mu_d - s * R @ mu_s

    return s, R, t
