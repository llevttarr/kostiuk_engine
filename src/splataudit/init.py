import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

INITS = ("random", "sparse_sfm", "dense_mvs", "foundation")
SEEDED_INITS = {"random"}

FSGS_SFM = {  # tools/colmap_llff.py
    "max_image_size": 4032,
    "max_num_features": 32768,
    "estimate_affine_shape": True,
    "domain_size_pooling": True,
    "guided_matching": True,
    "max_num_matches": 32768,
    "ba_local_max_num_iterations": 40,
    "ba_local_max_refinements": 3,
    "ba_global_max_num_iterations": 100,
    "num_threads": -1,
}
SMOKE_SFM = {
    **FSGS_SFM,
    "max_image_size": 1008,
    "max_num_features": 8192,
    "estimate_affine_shape": False,
    "domain_size_pooling": False,
    "guided_matching": False,
    "num_threads": 2,
}


def init_path(cache, dataset, scene, split_stem, init, seed):
    tag = init + (f"_s{seed}" if init in SEEDED_INITS else "")
    return Path(cache) / "inits" / dataset / scene / split_stem / f"{tag}.ply"


def build_init(name, scene, train_ids, split_stem, seed, cache, overwrite=False):
    """Compute (or reuse) an init, returns its stats dict."""
    ply = init_path(cache, scene.dataset, scene.name, split_stem, name, seed)
    meta = ply.with_suffix(".json")
    if meta.exists() and ply.exists() and not overwrite:
        cached = json.loads(meta.read_text())
        if cached["status"] == "ok":
            return cached

    stats = {
        "dataset": scene.dataset,
        "scene": scene.name,
        "split": split_stem,
        "init": name,
        "seed": seed if name in SEEDED_INITS else None,
        "train": scene.names(train_ids),
    }
    if name in ("sparse_sfm", "dense_mvs"):
        stats["sfm_settings"] = sfm_settings(scene.dataset)
    t0 = time.time()
    try:
        xyz, rgb = compute_init(name, scene, train_ids, seed)
        write_ply(ply, xyz, rgb)
        stats.update(status="ok", **init_quality(scene, train_ids, xyz))
    except NotImplementedError as e:
        stats.update(status="not_implemented", error=str(e))
    except Exception as e:
        stats.update(status="failed", error=f"{type(e).__name__}: {e}")
    stats["seconds"] = round(time.time() - t0, 2)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(stats, indent=1) + "\n")
    return stats


def get_init(name, scene, train_ids, split_stem, seed, cache):
    """Cached init for training, built first if missing."""
    stats = build_init(name, scene, train_ids, split_stem, seed, cache)
    if stats["status"] != "ok":
        raise RuntimeError(f"init {name} unavailable: {stats['status']} {stats.get('error', '')}")
    assert stats["train"] == scene.names(train_ids), f"cached {name} built from other train views"
    return read_ply(init_path(cache, scene.dataset, scene.name, split_stem, name, seed))


def compute_init(name, scene, train_ids, seed):
    """(xyz (N, 3), rgb (N, 3) in [0, 1]) float32, from the train views only."""
    if name == "random":
        return random_init(scene, train_ids, 0 if seed is None else seed)
    if name == "sparse_sfm":
        return sparse_sfm_init(scene, train_ids)
    if name == "dense_mvs":
        return dense_mvs_init(scene, train_ids)
    if name == "foundation":
        return foundation_init(scene, train_ids)
    raise ValueError(f"unknown init {name}")


def init_quality(scene, train_ids, xyz, cell=8):
    """Point count + coverage: share of cell x cell pixel blocks hit by a projected point."""
    cov = []
    for iid in train_ids:
        cam = scene.get_camera(iid)
        pc = xyz @ cam.R.T + cam.t
        pc = pc[pc[:, 2] > 1e-6]
        u = cam.fx * pc[:, 0] / pc[:, 2] + cam.cx
        v = cam.fy * pc[:, 1] / pc[:, 2] + cam.cy
        inside = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        gw, gh = -(-cam.width // cell), -(-cam.height // cell)
        cells = np.unique((v[inside] // cell).astype(int) * gw + (u[inside] // cell).astype(int))
        cov.append(len(cells) / (gw * gh))
    return {
        "n_points": len(xyz),
        "coverage": round(float(np.mean(cov)), 4),
        "coverage_per_view": [round(float(c), 4) for c in cov],
    }


def random_init(scene, train_ids, seed, n_points=100_000):
    rng = np.random.default_rng(seed)
    near, far = scene.bounds
    which = rng.integers(0, len(train_ids), n_points)
    xyz = np.empty((n_points, 3))
    for j, iid in enumerate(train_ids):
        cam = scene.get_camera(iid)
        m = which == j
        u = rng.uniform(0, cam.width, m.sum())
        v = rng.uniform(0, cam.height, m.sum())
        z = rng.uniform(near, far, m.sum())
        pc = np.stack([(u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z], axis=1)
        xyz[m] = (pc - cam.t) @ cam.R
    rgb = rng.uniform(0, 1, (n_points, 3))
    return xyz.astype(np.float32), rgb.astype(np.float32)


def sfm_settings(dataset):
    if os.environ.get("SPLATAUDIT_SFM") == "smoke":
        return dict(SMOKE_SFM)
    return {**FSGS_SFM, "max_num_features": 16384} if dataset == "mip360" else dict(FSGS_SFM)


def sparse_sfm_init(scene, train_ids):
    """Colmap points triangulated from the train views, dataset poses held fixed."""
    with tempfile.TemporaryDirectory(prefix="splataudit_sfm_") as tmp:
        rec = triangulate(scene, train_ids, Path(tmp), sfm_settings(scene.dataset))
        pts = list(rec.points3D.values())
        xyz = np.array([p.xyz for p in pts], dtype=np.float32).reshape(-1, 3)
        rgb = np.array([p.color for p in pts], dtype=np.float32).reshape(-1, 3) / 255
    if len(xyz) == 0:
        raise RuntimeError("triangulation produced no points")
    return xyz, rgb


def dense_mvs_init(scene, train_ids):
    """Fsgs's dense init: patch-match stereo + fusion on top of the triangulation."""
    import pycolmap

    if not pycolmap.has_cuda:
        raise NotImplementedError("dense_mvs needs a cuda build of colmap (patch_match_stereo)")
    with tempfile.TemporaryDirectory(prefix="splataudit_mvs_") as tmp:
        tmp = Path(tmp)
        rec = triangulate(scene, train_ids, tmp, sfm_settings(scene.dataset))
        rec.write(str(tmp / "triangulated"))
        dense = tmp / "dense"
        pycolmap.undistort_images(str(dense), str(tmp / "triangulated"), str(tmp / "images"))
        pycolmap.patch_match_stereo(str(dense))
        pycolmap.stereo_fusion(str(dense / "fused.ply"), str(dense))
        return read_fused_ply(dense / "fused.ply")


def foundation_init(scene, train_ids):
    """Dust3r / mast3r / vggt pointmaps on the train views."""
    raise NotImplementedError("foundation init not implemented yet (needs torch + model weights)")


def triangulate(scene, train_ids, tmp, s):
    """Sift + exhaustive matching + point_triangulator with the dataset poses (fsgs recipe)."""
    import pycolmap

    cams = [scene.get_camera(i) for i in train_ids]
    img_dir = tmp / "images"
    img_dir.mkdir()
    for c in cams:
        os.symlink(c.sfm_image_path.resolve(), img_dir / c.name)
    db = tmp / "database.db"

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = cams[0].sfm_camera["model"]
    reader.camera_params = ",".join(f"{v:.10g}" for v in cams[0].sfm_camera["params"])
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = s["max_image_size"]
    ext.num_threads = s["num_threads"]
    ext.sift.max_num_features = s["max_num_features"]
    ext.sift.estimate_affine_shape = s["estimate_affine_shape"]
    ext.sift.domain_size_pooling = s["domain_size_pooling"]
    pycolmap.extract_features(
        db,
        img_dir,
        image_names=[c.name for c in cams],
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=ext,
    )

    match = pycolmap.FeatureMatchingOptions()
    match.guided_matching = s["guided_matching"]
    match.max_num_matches = s["max_num_matches"]
    match.num_threads = s["num_threads"]
    pycolmap.match_exhaustive(db, matching_options=match)

    database = pycolmap.Database.open(db)
    rec = pycolmap.Reconstruction()
    for cam in database.read_all_cameras():
        rec.add_camera_with_trivial_rig(cam)
    for im in database.read_all_images():
        c = scene.by_name(im.name)
        rec.add_image_with_trivial_frame(
            pycolmap.Image(name=im.name, camera_id=im.camera_id, image_id=im.image_id),
            pycolmap.Rigid3d(pycolmap.Rotation3d(c.R), c.t),
        )
    database.close()

    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_local_max_num_iterations = s["ba_local_max_num_iterations"]
    opts.ba_local_max_refinements = s["ba_local_max_refinements"]
    opts.ba_global_max_num_iterations = s["ba_global_max_num_iterations"]
    (tmp / "triangulated").mkdir()
    return pycolmap.triangulate_points(rec, db, img_dir, tmp / "triangulated", options=opts)


PLY_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
)


def write_ply(path, xyz, rgb):
    """Binary ply, xyz float32 + rgb uint8 (rgb given in [0, 1])."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.empty(len(xyz), dtype=PLY_DTYPE)
    data["x"], data["y"], data["z"] = np.asarray(xyz, np.float32).T
    data["red"], data["green"], data["blue"] = np.clip(np.round(np.asarray(rgb) * 255), 0, 255).T
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path.write_bytes(header.encode("ascii") + data.tobytes())


def read_ply(path):
    """Ply from write_ply -> (xyz float32, rgb float32 in [0, 1])."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int(
        next(
            line.split()[-1]
            for line in raw[:end].decode().splitlines()
            if line.startswith("element vertex")
        )
    )
    d = np.frombuffer(raw[end:], dtype=PLY_DTYPE, count=n)
    xyz = np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float32)
    rgb = np.stack([d["red"], d["green"], d["blue"]], axis=1).astype(np.float32) / 255
    return xyz, rgb


def read_fused_ply(path):
    """Colmap's fused.ply (x y z nx ny nz r g b) -> (xyz, rgb in [0, 1])."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int(
        next(
            line.split()[-1]
            for line in raw[:end].decode().splitlines()
            if line.startswith("element vertex")
        )
    )
    dt = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
        ]
    )
    d = np.frombuffer(raw[end:], dtype=dt, count=n)
    return (
        np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float32),
        np.stack([d["r"], d["g"], d["b"]], 1).astype(np.float32) / 255,
    )
