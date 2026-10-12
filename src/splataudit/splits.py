from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from splataudit.data import Scene


# Type alias for a split: maps role → list of image_ids
Split = dict[str, list[int]]


# ─── Public API ───────────────────────────────────────────────────────────────


def make_splits(
    scene: Scene,
    n_views: int,
    mode: str,
    seed: int,
    *,
    sector_frac: float = 0.30,
    llff_holdout: int = 8,
) -> Split:
    """
    Build a train/test split for *scene*.

    Parameters
    ----------
    scene : Scene
    n_views : int
        Number of training views to select (typically 3 or 9).
    mode : str
        ``"interp"``  — test views are interleaved along the trajectory
                        (~5° angular gap to the nearest train view).
                        Reproduces the standard LLFF evaluation protocol.

        ``"sector"``  — test views form a contiguous, unobserved angular
                        sector (~30 % of cameras).  Forces spatial
                        extrapolation rather than near-trajectory
                        interpolation.
    seed : int
        RNG seed for reproducible train-view selection.
    sector_frac : float
        Fraction of cameras reserved as the test sector (``mode="sector"``).
    llff_holdout : int
        Every k-th image becomes a test view (``mode="interp"``).

    Returns
    -------
    dict
        ``{"train": [image_id, ...], "test": [image_id, ...]}``
        Train and test image_ids are disjoint and sorted.
    """
    rng = np.random.default_rng(seed)

    if mode == "interp":
        return _interp_split(scene, n_views, llff_holdout, rng)
    elif mode == "sector":
        return _sector_split(scene, n_views, sector_frac, rng)
    else:
        raise ValueError(f"Unknown split mode {mode!r}. Choose 'interp' or 'sector'.")


def save_split(path: Path, split: Split, meta: dict) -> None:
    """
    Persist a split and its metadata to JSON.

    The JSON file is the single source of truth for reproducibility:
    it records the exact image_ids used for train/test alongside the
    config parameters that produced them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"train": split["train"], "test": split["test"], **meta}
    path.write_text(json.dumps(payload, indent=2))


def load_split(path: Path) -> tuple[Split, dict]:
    """
    Load a persisted split.

    Returns
    -------
    split : dict  ``{"train": [...], "test": [...]}``
    meta  : dict  remaining metadata (scene, mode, seed, …)
    """
    payload = json.loads(Path(path).read_text())
    split: Split = {
        "train": payload.pop("train"),
        "test":  payload.pop("test"),
    }
    return split, payload


# ─── Private helpers ──────────────────────────────────────────────────────────


def _interp_split(
    scene: Scene,
    n_views: int,
    holdout: int,
    rng: np.random.Generator,
) -> Split:
    """
    Interleaved split: every ``holdout``-th image goes to test.

    Train views are chosen with even spacing along the remaining candidates
    so they cover the full trajectory regardless of scene length.
    """
    all_ids = [cam.image_id for cam in scene.cameras]

    test_ids = [iid for i, iid in enumerate(all_ids) if i % holdout == 0]
    test_set = set(test_ids)
    train_pool = [iid for iid in all_ids if iid not in test_set]

    if len(train_pool) < n_views:
        raise ValueError(
            f"[{scene.name}] Only {len(train_pool)} candidate train images after "
            f"holdout={holdout}, but n_views={n_views} requested."
        )

    train_ids = _evenly_spaced(train_pool, n_views)
    return {"train": sorted(train_ids), "test": sorted(test_ids)}


def _sector_split(
    scene: Scene,
    n_views: int,
    sector_frac: float,
    rng: np.random.Generator,
) -> Split:
    """
    Sector split: a contiguous arc of cameras is held out as test.

    Ordering
    --------
    * **Mip-NeRF 360 / DTU** — sort by azimuth angle (360° or spherical).
    * **LLFF** (forward-facing) — sort by camera-center X coordinate,
      representing position along the capture trajectory.

    The test sector starts at the midpoint of the ordered sequence so that
    train candidates are spread on both sides, giving reasonable coverage.
    """
    cameras = scene.cameras

    if scene.dataset in ("mip360", "dtu"):
        ordered = sorted(cameras, key=lambda c: c.azimuth)
    else:
        # LLFF: forward-facing trajectory — order by X position
        ordered = sorted(cameras, key=lambda c: float(c.center[0]))

    n_total = len(ordered)
    n_test = max(1, round(n_total * sector_frac))

    # Place sector starting at midpoint (train candidates wrap around both ends)
    sector_start = n_total // 2
    sector_indices = {(sector_start + i) % n_total for i in range(n_test)}

    test_ids  = [ordered[i].image_id for i in range(n_total) if i in sector_indices]
    train_pool = [ordered[i].image_id for i in range(n_total) if i not in sector_indices]

    if len(train_pool) < n_views:
        raise ValueError(
            f"[{scene.name}] Only {len(train_pool)} non-sector images, "
            f"but n_views={n_views} requested.  Reduce sector_frac."
        )

    train_ids = _evenly_spaced(train_pool, n_views)
    return {"train": sorted(train_ids), "test": sorted(test_ids)}


def _evenly_spaced(pool: list[int], n: int) -> list[int]:
    """Select *n* evenly-spaced elements from *pool* (deterministic)."""
    indices = np.round(np.linspace(0, len(pool) - 1, n)).astype(int)
    return [pool[i] for i in indices]
