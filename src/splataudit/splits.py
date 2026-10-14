import json
import math
from pathlib import Path

import numpy as np

PROTOCOLS = ("p0", "p2")
SECTOR_POSITIONS = {"llff": 0, "dtu": 2, "mip360": 4}
# without the gap the mip360 sector is no farther from training than p0 test views
SECTOR_GAP = {"dtu": 0, "mip360": 1}
LLFF_HOLD = 8

DTU_TRAIN = [25, 22, 28, 40, 44, 48, 0, 8, 13]
DTU_EXCLUDE = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
DTU_TEST = [i for i in range(49) if i not in DTU_TRAIN + DTU_EXCLUDE]

DISTANCE_BINS = (0.0, 5.0, 10.0, 20.0, math.inf)
SPLIT_VERSION = 2


def make_split(scene, protocol, n_views, k=None):
    """Split record for one scene: train / test / eval_all names + distances."""
    extra = {}
    if protocol == "p0":
        train, test, eval_all = p0_ids(scene, n_views)
        ordering = "filename"
    elif protocol == "p2":
        if SECTOR_POSITIONS[scene.dataset] == 0:
            raise ValueError(f"no P2 on {scene.dataset}")
        train, test, eval_all, ordering, gap = p2_ids(scene, n_views, k)
        p0_test = set(p0_ids(scene, n_views)[1])
        extra = {"gap": gap, "shared_with_p0": scene.names(sorted(set(test) & p0_test))}
    else:
        raise ValueError(f"unknown protocol {protocol}")

    assert not set(train) & set(eval_all) and set(test) <= set(eval_all), "train views leak into eval"
    return {
        "version": SPLIT_VERSION,
        "dataset": scene.dataset,
        "scene": scene.name,
        "protocol": protocol,
        "n_views": n_views,
        "k": k,
        "ordering": ordering,
        "train": scene.names(train),
        "test": scene.names(test),
        "eval_all": scene.names(eval_all),
        "distances": view_distances(scene, train, eval_all),
        **extra,
    }


def split_path(splits_root, dataset, scene, protocol, n_views, k=None):
    tag = f"{protocol}_{n_views}v" + (f"_k{k}" if k is not None else "")
    return Path(splits_root) / dataset / scene / f"{tag}.json"


def save_split(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")


def load_split(path):
    return json.loads(Path(path).read_text())


def resolve(scene, record):
    """Split record names -> this scene's image ids."""
    return {key: scene.ids(record[key]) for key in ("train", "test", "eval_all")}


def view_distances(scene, train_ids, view_ids):
    """Angle (deg) from each view to its nearest train view, keyed by name."""
    centroid = scene.centroid

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    dirs = unit(np.stack([scene.get_camera(i).center for i in train_ids]) - centroid)
    fwd = np.stack([scene.get_camera(i).forward for i in train_ids])
    out = {}
    for vid in view_ids:
        cam = scene.get_camera(vid)
        centre = np.degrees(np.arccos(np.clip(dirs @ unit(cam.center - centroid), -1, 1)))
        optical = np.degrees(np.arccos(np.clip(fwd @ cam.forward, -1, 1)))
        j = int(centre.argmin())
        out[cam.name] = {
            "centre_deg": round(float(centre[j]), 4),
            "optical_deg": round(float(optical.min()), 4),
            "nearest_train": scene.get_camera(train_ids[j]).name,
        }
    return out


def distance_bin(deg):
    """P1 group label of an angle in degrees."""
    for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        if lo <= deg < hi:
            return f"{lo:g}-{hi:g}" if math.isfinite(hi) else f">{lo:g}"


def capture_order(scene, ids):
    """Ids ordered along the capture, plus the kind of ordering."""
    cams = [scene.get_camera(i) for i in ids]
    C = np.stack([c.center for c in cams])
    if scene.dataset == "llff":
        _, _, vt = np.linalg.svd(C - C.mean(0), full_matrices=False)
        proj = (C - C.mean(0)) @ vt[0]
        if np.corrcoef(proj, np.arange(len(ids)))[0, 1] < 0:
            proj = -proj
        return [ids[j] for j in np.argsort(proj, kind="stable")], "pc1"

    up = np.mean([c.up for c in scene.cameras], axis=0)
    up /= np.linalg.norm(up)
    v = C - scene.centroid
    v -= (v @ up)[:, None] * up
    e1 = v[0] / np.linalg.norm(v[0])
    az = np.arctan2(v @ np.cross(up, e1), v @ e1) % (2 * np.pi)
    order = list(np.argsort(az, kind="stable"))
    if scene.dataset == "mip360":
        return [ids[j] for j in order], "azimuth-circular"

    # dtu is an open arc, cut it at the largest azimuth gap
    a = az[order]
    start = (int(np.diff(np.concatenate([a, a[:1] + 2 * np.pi])).argmax()) + 1) % len(order)
    return [ids[j] for j in order[start:] + order[:start]], "azimuth-linear"


def p0_ids(scene, n):
    """Published split -> (train, test, eval_all) ids."""
    if scene.dataset == "dtu":
        return sorted(DTU_TRAIN[:n]), list(DTU_TEST), list(DTU_TEST)
    ids = [c.image_id for c in scene.cameras]
    test = [i for i in ids if i % LLFF_HOLD == 0]
    train = _evenly_spaced([i for i in ids if i % LLFF_HOLD != 0], n)
    return train, test, [i for i in ids if i not in train]


def p2_ids(scene, n, k):
    """Sector split -> (train, test, eval_all, ordering, gap)."""
    pool = [c.image_id for c in scene.cameras]
    if scene.dataset == "dtu":
        pool = [i for i in pool if i not in DTU_EXCLUDE]
    K = len(p0_ids(scene, n)[1])
    B = SECTOR_GAP[scene.dataset] * K
    ordered, ordering = capture_order(scene, pool)
    N = len(ordered)

    if ordering == "azimuth-circular":
        start = round(k * N / SECTOR_POSITIONS[scene.dataset])
        test = [ordered[(start + i) % N] for i in range(K)]
        rest = [ordered[(start + K + B + i) % N] for i in range(N - K - 2 * B)]
    else:
        test = ordered[:K] if k == 0 else ordered[N - K:]
        rest = ordered[K + B:] if k == 0 else ordered[:N - K - B]
    train = _evenly_spaced(rest, n)
    return sorted(train), sorted(test), [i for i in pool if i not in train], ordering, B


def _evenly_spaced(pool, n):
    # fsgs pick, python round is half to even
    return [pool[round(float(x))] for x in np.linspace(0, len(pool) - 1, n)]
