from __future__ import annotations

import numpy as np

from splataudit.data import Scene


def dust3r_init(
    scene: Scene,
    train_ids: list[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dense point cloud from DUSt3R run **only** on training views.

    Pipeline
    --------
    1. Load the DUSt3R / MASt3R model from ``external/``.
    2. Run pairwise forward passes over all train-view image pairs.
    3. Global point-map optimisation (``dust3r.cloud_opt``).
    4. Extract the fused 3D point cloud and per-point colours.
    5. **Sim3 alignment**: align DUSt3R world coordinates to the dataset's
       GT coordinate system using Umeyama alignment of the train camera
       centres (known in both coordinate systems).

    Why alignment?
    --------------
    DUSt3R produces its own arbitrary metric scale and orientation.
    We align it to the dataset GT so that the depth-prior scale and the
    GT camera poses are consistent during training.

    Notes
    -----
    * Only *train_ids* images are passed to DUSt3R.
    * The resulting point cloud is typically 10–100× denser than COLMAP,
      which is the key advantage of this initialisation.

    Dependencies
    ------------
    Place the DUSt3R repo under ``external/dust3r/`` (git submodule).
    """
    raise NotImplementedError(
        "DUSt3R init not yet implemented. "
        "Requires: external/dust3r submodule + sim3 alignment."
    )
