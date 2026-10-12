from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


# Type alias for the initial point cloud
InitPoints = tuple[np.ndarray, np.ndarray]  # (xyz: (N,3) float32, rgb: (N,3) float32)


def get_init(
    name: str,
    scene: "Scene",  # noqa: F821  (forward ref)
    train_ids: list[int],
    seed: int,
) -> InitPoints:
    """
    Build the initial 3D point cloud for a 3DGS experiment.

    Only *train_ids* cameras are used — test cameras are never touched,
    ensuring no information about held-out views leaks into the geometry.

    Parameters
    ----------
    name : str
        ``"random"``  — uniform random points inside the scene bounding box.
        ``"colmap"``  — sparse reconstruction via pycolmap (train views only).
        ``"dust3r"``  — dense reconstruction via DUSt3R (train views only).
    scene : Scene
    train_ids : list[int]
        ``image_id`` values of training cameras.
    seed : int
        RNG seed (used by ``random``; passed through for reproducibility).

    Returns
    -------
    xyz : np.ndarray, shape (N, 3), dtype float32
        3D point positions in world coordinates.
    rgb : np.ndarray, shape (N, 3), dtype float32
        Per-point colours in [0, 1].
    """
    if name == "random":
        from splataudit.init.random import random_init
        return random_init(scene, train_ids, seed)
    elif name == "colmap":
        from splataudit.init.colmap import colmap_init
        return colmap_init(scene, train_ids, seed)
    elif name == "dust3r":
        from splataudit.init.dust3r import dust3r_init
        return dust3r_init(scene, train_ids, seed)
    else:
        raise ValueError(
            f"Unknown init {name!r}. Choose 'random', 'colmap', or 'dust3r'."
        )
