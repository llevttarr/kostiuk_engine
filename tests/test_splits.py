"""Split invariants (BENCHMARK_PLAN §3 checks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from splataudit.data import DATASETS, Camera, Scene, load_scene
from splataudit.splits import (
    DTU_EXCLUDE,
    DTU_TEST,
    DTU_TRAIN,
    SECTOR_POSITIONS,
    _evenly_spaced,
    capture_order,
    distance_bin,
    load_split,
    make_split,
    split_path,
)

DATA = Path(__file__).resolve().parents[1] / "data"
SPLITS = Path(__file__).resolve().parents[1] / "splits"


def _have(dataset: str, scene: str) -> bool:
    if dataset == "dtu":
        return (DATA / "dtu" / "Calibration" / "cal18" / "pos_049.txt").exists()
    return (DATA / dataset / scene / "sparse" / "0").exists()


def _scenes(dataset: str):
    return [
        pytest.param(
            dataset,
            s,
            marks=pytest.mark.skipif(not _have(dataset, s), reason="data not downloaded"),
        )
        for s in DATASETS[dataset]["scenes"]
    ]


ALL_SCENES = _scenes("llff") + _scenes("mip360") + _scenes("dtu")


def test_dtu_regnerf_ids():
    assert len(DTU_TEST) == 25
    assert not set(DTU_TRAIN) & set(DTU_TEST)
    assert not set(DTU_EXCLUDE) & set(DTU_TEST)
    assert sorted(DTU_TRAIN + DTU_EXCLUDE + DTU_TEST) == list(range(49))
    assert DTU_TEST[:6] == [1, 2, 9, 10, 11, 12]


def test_evenly_spaced_matches_fsgs_rounding():
    # FSGS: [round(i) for i in np.linspace(0, len-1, n)], round half to even
    assert _evenly_spaced(list(range(36)), 3) == [0, 18, 35]  # 17.5 → 18
    assert _evenly_spaced(list(range(48)), 3) == [0, 24, 47]  # 23.5 → 24
    assert _evenly_spaced(list(range(17)), 3) == [0, 8, 16]


def test_distance_bins():
    assert distance_bin(0.0) == "0-5"
    assert distance_bin(4.99) == "0-5"
    assert distance_bin(5.0) == "5-10"
    assert distance_bin(25.0) == ">20"


def _ring_scene(n: int = 80, dataset: str = "mip360") -> Scene:
    """Synthetic 360° capture: n cameras on a circle looking at the origin."""
    cams = []
    for i in range(n):
        a = 2 * np.pi * i / n
        c = np.array([np.cos(a), 0.0, np.sin(a)]) * 3.0
        z = -c / np.linalg.norm(c)  # forward → origin
        x = np.cross([0.0, -1.0, 0.0], z)  # right; camera y = down
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        cams.append(
            Camera(
                image_id=i,
                name=f"img{i:03d}.png",
                image_path=Path("/nonexistent"),
                R=R,
                t=-R @ c,
                fx=100,
                fy=100,
                cx=50,
                cy=50,
                width=100,
                height=100,
            )
        )
    return Scene("ring", dataset, Path("/nonexistent"), cams, (0.5, 6.0), np.zeros((1, 3)))


def test_ring_order_is_circular_and_contiguous():
    sc = _ring_scene()
    order, kind = capture_order(sc, list(range(len(sc))))
    assert kind == "azimuth-circular"
    steps = np.diff(order + order[:1]) % len(sc)
    assert set(steps) == {1} or set(steps) == {len(sc) - 1}  # one direction around the ring


@pytest.mark.parametrize("k", range(SECTOR_POSITIONS["mip360"]))
def test_p2_on_ring(k):
    sc = _ring_scene()
    p0, p2 = make_split(sc, "p0", 12), make_split(sc, "p2", 12, k)
    assert len(p2["train"]) == len(p0["train"]) == 12
    assert len(p2["test"]) == len(p0["test"]) == 10  # matched count, ⌈80/8⌉
    test = sorted(sc.ids(p2["test"]))
    # contiguous block on the ring
    gaps = np.diff([*test, test[0] + len(sc)])
    assert sorted(gaps)[:-1] == [1] * (len(test) - 1)
    # gap of K views on each side: no training view within K+1 ring steps of the sector
    assert p2["gap"] == 10
    ring_dist = [
        min((t - s) % len(sc), (s - t) % len(sc)) for t in sc.ids(p2["train"]) for s in test
    ]
    assert min(ring_dist) > 10
    # sector views are farther from training than P0's test views
    far = np.mean([p2["distances"][v]["centre_deg"] for v in p2["test"]])
    near = np.mean([p0["distances"][v]["centre_deg"] for v in p0["test"]])
    assert far > near


def test_no_p2_on_llff():
    with pytest.raises(ValueError, match="no P2"):
        make_split(_ring_scene(dataset="llff"), "p2", 3, 0)


@pytest.mark.parametrize("dataset,name", _scenes("mip360") + _scenes("dtu"))
def test_p2_is_harder_than_p0(scenes, dataset, name):
    """The reason P2 exists: its test views are farther from training than P0's."""
    sc = scenes(dataset, name)
    n = DATASETS[dataset]["n_views"]

    def mean_dist(rec):
        return np.mean([rec["distances"][v]["centre_deg"] for v in rec["test"]])

    p0 = mean_dist(make_split(sc, "p0", n))
    for k in range(SECTOR_POSITIONS[dataset]):
        assert mean_dist(make_split(sc, "p2", n, k)) > 1.5 * p0


def _fsgs_reference(scene: Scene, n: int):
    """Independent transcription of FSGS dataset_readers.readColmapSceneInfo."""
    names = sorted(c.name for c in scene.cameras)
    train = [c for i, c in enumerate(names) if i % 8 != 0]
    test = [c for i, c in enumerate(names) if i % 8 == 0]
    idx_sub = [round(i) for i in np.linspace(0, len(train) - 1, n)]
    train = [c for i, c in enumerate(train) if i in idx_sub]
    return train, test


@pytest.fixture(scope="module")
def scenes():
    cache = {}

    def get(dataset, name):
        if (dataset, name) not in cache:
            cache[dataset, name] = load_scene(DATA, dataset, name)
        return cache[dataset, name]

    return get


@pytest.mark.parametrize("dataset,name", _scenes("llff") + _scenes("mip360"))
def test_p0_matches_fsgs(scenes, dataset, name):
    sc = scenes(dataset, name)
    n = DATASETS[dataset]["n_views"]
    rec = make_split(sc, "p0", n)
    train, test = _fsgs_reference(sc, n)
    assert rec["train"] == train
    assert rec["test"] == test


@pytest.mark.skipif(not _have("llff", "fern"), reason="data not downloaded")
def test_fern_known_split(scenes):
    rec = make_split(scenes("llff", "fern"), "p0", 3)
    sc = scenes("llff", "fern")
    assert sc.ids(rec["train"]) == [1, 10, 19]
    assert sc.ids(rec["test"]) == [0, 8, 16]


@pytest.mark.parametrize("dataset,name", ALL_SCENES)
def test_split_invariants(scenes, dataset, name):
    sc = scenes(dataset, name)
    n = DATASETS[dataset]["n_views"]
    p0 = make_split(sc, "p0", n)
    for k in range(SECTOR_POSITIONS[dataset]):
        p2 = make_split(sc, "p2", n, k)
        for rec in (p0, p2):
            assert not set(rec["train"]) & set(rec["eval_all"])
            assert set(rec["test"]) <= set(rec["eval_all"])
            assert set(rec["distances"]) == set(rec["eval_all"])
            assert len(rec["train"]) == n
        assert len(p2["test"]) == len(p0["test"])  # matched count
        if dataset == "dtu":
            assert not set(sc.ids(p2["eval_all"])) & set(DTU_EXCLUDE)


@pytest.mark.parametrize("dataset,name", _scenes("dtu"))
def test_p2_sector_is_outside_training_range(scenes, dataset, name):
    """On open captures (DTU) the sector lies entirely beyond the training views."""
    sc = scenes(dataset, name)
    n = DATASETS[dataset]["n_views"]
    for k in range(SECTOR_POSITIONS[dataset]):
        rec = make_split(sc, "p2", n, k)
        pool = [i for i in range(len(sc)) if dataset != "dtu" or i not in DTU_EXCLUDE]
        order, _ = capture_order(sc, pool)
        pos = {iid: j for j, iid in enumerate(order)}
        tr = [pos[i] for i in sc.ids(rec["train"])]
        te = [pos[i] for i in sc.ids(rec["test"])]
        assert max(te) < min(tr) if k == 0 else min(te) > max(tr)


@pytest.mark.parametrize("dataset,name", ALL_SCENES)
def test_committed_split_files_are_current(scenes, dataset, name):
    """The files in splits/ are exactly what the generator produces now."""
    n = DATASETS[dataset]["n_views"]
    path = split_path(SPLITS, dataset, name, "p0", n)
    if not path.exists():
        pytest.skip("split files not generated")
    sc = scenes(dataset, name)
    jobs = [("p0", None)] + [("p2", k) for k in range(SECTOR_POSITIONS[dataset])]
    for protocol, k in jobs:
        assert load_split(split_path(SPLITS, dataset, name, protocol, n, k)) == make_split(
            sc, protocol, n, k
        )
