import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from splataudit.init import align_sim3, compute_init, init_path, read_ply, write_ply

TRUE_SCALE = 2.5
TRUE_SHIFT = np.array([1.0, -2.0, 0.5])


def poses(n=4, seed=0):
    """n camera (R, t) pairs, world to camera."""
    rng = np.random.default_rng(seed)
    return list(zip(Rotation.random(n, rng=seed).as_matrix(), rng.normal(size=(n, 3)), strict=True))


def shifted(src, s=TRUE_SCALE, rot=None, T=TRUE_SHIFT):
    """The same cameras seen in a world scaled by s, rotated by rot and shifted by T."""
    rot = Rotation.random(1, rng=3).as_matrix()[0] if rot is None else rot
    dst = [(r @ rot.T, s * t - (r @ rot.T) @ T) for r, t in src]
    return dst, rot


def test_align_sim3_recovers_a_known_similarity():
    src = poses()
    dst, rot = shifted(src)
    s, R, T = align_sim3(src, dst)
    assert s == pytest.approx(TRUE_SCALE)
    assert np.allclose(R, rot, atol=1e-8)
    assert np.allclose(T, TRUE_SHIFT, atol=1e-6)


def test_align_sim3_moves_points_into_the_target_frame():
    src = poses()
    dst, rot = shifted(src)
    s, R, T = align_sim3(src, dst)
    p_src = np.array([[0.3, -1.2, 4.0], [0.0, 0.0, 0.0]])
    expected = TRUE_SCALE * (p_src @ rot.T) + TRUE_SHIFT
    assert np.allclose(s * (p_src @ R.T) + T, expected, atol=1e-6)


def test_align_sim3_is_identity_for_identical_poses():
    src = poses()
    s, R, T = align_sim3(src, src)
    assert s == pytest.approx(1.0)
    assert np.allclose(R, np.eye(3), atol=1e-8)
    assert np.allclose(T, 0.0, atol=1e-8)


def test_align_sim3_works_with_three_nearly_collinear_cameras():
    # llff trains on 3 cameras in almost a straight line, where fitting on centres alone fails
    rots = Rotation.random(3, rng=1).as_matrix()
    centres = np.array([[0.0, 0, 0], [1.0, 1e-7, 0], [2.0, -1e-7, 0]])
    src = [(r, -r @ c) for r, c in zip(rots, centres, strict=True)]
    dst, rot = shifted(src)
    s, R, _ = align_sim3(src, dst)
    assert s == pytest.approx(TRUE_SCALE)
    assert np.allclose(R, rot, atol=1e-6)


def test_align_sim3_refuses_a_single_camera_centre():
    src = [(np.eye(3), np.zeros(3))]
    with pytest.raises(AssertionError):
        align_sim3(src, src)


def test_ply_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(100, 3)).astype(np.float32)
    rgb = rng.uniform(size=(100, 3)).astype(np.float32)
    write_ply(tmp_path / "p.ply", xyz, rgb)
    back_xyz, back_rgb = read_ply(tmp_path / "p.ply")
    assert np.allclose(back_xyz, xyz)
    assert np.allclose(back_rgb, rgb, atol=1 / 255)


def test_init_path_tags_only_the_seeded_inits():
    assert init_path("cache", "llff", "fern", "p0_3v", "random", 2).name == "random_s2.ply"
    assert init_path("cache", "llff", "fern", "p0_3v", "sparse_sfm", 2).name == "sparse_sfm.ply"


def test_foundation_init_reports_its_missing_dependency(monkeypatch):
    import sys

    pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "vggt.models.vggt", None)
    with pytest.raises(NotImplementedError, match="vggt"):
        compute_init("foundation", None, [0], None)


def test_unknown_init_is_rejected():
    with pytest.raises(ValueError):
        compute_init("sfm_but_better", None, [0], None)
