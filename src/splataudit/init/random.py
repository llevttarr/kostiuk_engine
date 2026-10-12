from __future__ import annotations

import numpy as np

from splataudit.data import Scene


def random_init(
    scene: Scene,
    train_ids: list[int],
    seed: int,
    n_points: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniform random point cloud inside the scene bounding box.

    Points are sampled from a cube scaled by the scene's far bound.
    This is the weakest initialisation baseline — no geometric structure.

    Parameters
    ----------
    n_points : int
        Number of random 3D points to generate.

    Returns
    -------
    xyz : (n_points, 3) float32
    rgb : (n_points, 3) float32,  uniform in [0, 1]
    """
    rng = np.random.default_rng(seed)
    _, far = scene.bounds

    # Uniform sample in a cube centred at scene_center, side = 2 * far
    xyz = scene.scene_center + rng.uniform(-far, far, (n_points, 3))
    rgb = rng.uniform(0.0, 1.0, (n_points, 3))

    return xyz.astype(np.float32), rgb.astype(np.float32)
