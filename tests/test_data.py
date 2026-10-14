"""Loader, init and bookkeeping checks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from splataudit.data import DATASETS, load_scene
from splataudit.init import init_quality, random_init, read_ply, write_ply
from splataudit.utils import CameraSampler, append_rows, finished_runs

DATA = Path(__file__).resolve().parents[1] / "data"


def _colmap_scenes():
    out = []
    for ds in ("llff", "mip360"):
        for s in DATASETS[ds]["scenes"]:
            have = (DATA / ds / s / "poses_bounds.npy").exists()
            out.append(pytest.param(ds, s, marks=pytest.mark.skipif(not have, reason="no data")))
    return out


@pytest.mark.parametrize("dataset,name", _colmap_scenes())
def test_poses_match_poses_bounds(dataset, name):
    """sparse/0 poses == poses_bounds.npy (LLFF [down, right, back] columns)."""
    sc = load_scene(DATA, dataset, name)
    pb = np.load(DATA / dataset / name / "poses_bounds.npy")[:, :15].reshape(-1, 3, 5)
    np.testing.assert_allclose(pb[:, :, 3], sc.centers, atol=1e-6)
    for p, cam in zip(pb, sc.cameras):
        c2w_cv = np.stack([p[:, 1], p[:, 0], -p[:, 2]], axis=1)
        np.testing.assert_allclose(c2w_cv, cam.R.T, atol=1e-6)
    cam = sc.cameras[0]
    assert cam.load_image().shape == (cam.height, cam.width, 3)


@pytest.mark.skipif(not (DATA / "dtu/Calibration/cal18/pos_049.txt").exists(), reason="no calib")
def test_dtu_cameras_look_at_object():
    sc = load_scene(DATA, "dtu", "scan8")
    assert len(sc) == 49 and (sc.cameras[0].width, sc.cameras[0].height) == (400, 300)
    near, far = sc.bounds
    cen = sc.centroid
    for cam in sc.cameras:
        to_obj = cen - cam.center
        dist = np.linalg.norm(to_obj)
        assert near < dist < far
        assert np.degrees(np.arccos(to_obj @ cam.forward / dist)) < 1.0


def test_ply_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(500, 3)).astype(np.float32)
    rgb = rng.uniform(size=(500, 3)).astype(np.float32)
    write_ply(tmp_path / "p.ply", xyz, rgb)
    x2, c2 = read_ply(tmp_path / "p.ply")
    np.testing.assert_array_equal(x2, xyz)
    np.testing.assert_allclose(c2, rgb, atol=1 / 255)


@pytest.mark.skipif(not (DATA / "llff/fern/sparse/0").exists(), reason="no data")
def test_random_init_inside_training_frusta():
    sc = load_scene(DATA, "llff", "fern")
    train = [1, 10, 19]
    xyz, _ = random_init(sc, train, seed=0, n_points=5000)
    near, far = sc.bounds
    in_some = np.zeros(len(xyz), bool)
    for iid in train:
        cam = sc.get_camera(iid)
        pc = xyz @ cam.R.T + cam.t
        u = cam.fx * pc[:, 0] / pc[:, 2] + cam.cx
        v = cam.fy * pc[:, 1] / pc[:, 2] + cam.cy
        in_some |= ((pc[:, 2] >= near * 0.999) & (pc[:, 2] <= far * 1.001)
                    & (u >= -1e-3) & (u <= cam.width + 1e-3) & (v >= -1e-3) & (v <= cam.height + 1e-3))
    assert in_some.all()
    dense, _ = random_init(sc, train, seed=0, n_points=100_000)    # the default count
    assert init_quality(sc, train, dense)["coverage"] > 0.95
    np.testing.assert_array_equal(xyz, random_init(sc, train, seed=0, n_points=5000)[0])
    assert not np.array_equal(xyz, random_init(sc, train, seed=1, n_points=5000)[0])


def test_camera_sampler_is_seeded_and_covers_every_view():
    a = [CameraSampler(list("abc"), seed=3).next() for _ in range(1)]
    s1, s2, s3 = CameraSampler(list("abc"), 3), CameraSampler(list("abc"), 3), CameraSampler(list("abc"), 4)
    seq1 = [s1.next() for _ in range(30)]
    assert seq1 == [s2.next() for _ in range(30)]
    assert seq1 != [s3.next() for _ in range(30)]
    for epoch in range(10):
        assert sorted(seq1[3 * epoch:3 * epoch + 3]) == ["a", "b", "c"]
    assert a[0] in "abc"


def test_results_resume(tmp_path):
    csv = tmp_path / "results.csv"
    base = {"dataset": "llff", "scene": "fern", "n_views": 3, "protocol": "p0", "k": None,
            "init": "random", "method": "vanilla", "seed": 0}
    append_rows(csv, [base | {"test_image": "a", "psnr": 20.0, "status": "ok"}])
    append_rows(csv, [base | {"method": "fsgs", "status": "not_implemented", "error": "x"}])
    done = finished_runs(csv)
    assert len(done) == 1
    assert ("llff", "fern", "3", "p0", "", "random", "vanilla", "0") in done
