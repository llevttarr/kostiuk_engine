from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

DATASETS = {
    "llff": {"image_factor": 8, "n_views": 3,
             "scenes": ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]},
    "mip360": {"image_factor": 8, "n_views": 12,
               "scenes": ["bicycle", "garden", "room", "counter"]},
    "dtu": {"image_factor": 4, "n_views": 3, "light": 3,
            "scenes": ["scan8", "scan21", "scan30", "scan31", "scan34", "scan38", "scan40", "scan41",
                       "scan45", "scan55", "scan63", "scan82", "scan103", "scan110", "scan114"]},
}

DTU_BOUNDS = (0.5, 3.5)  # regnerf near / far
DTU_IDR_SCANS = {"scan40", "scan55", "scan63", "scan110", "scan114"}


class Camera:

    def __init__(self, image_id, name, image_path, R, t, fx, fy, cx, cy, width, height,
                 factor=1, mask_path=None, sfm_image_path=None, sfm_camera=None):
        self.image_id = image_id
        self.name = name
        self.image_path = image_path
        self.R, self.t = R, t
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width, self.height = width, height
        self.factor = factor
        self.mask_path = mask_path
        self.sfm_image_path = sfm_image_path
        self.sfm_camera = sfm_camera or {}

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    @property
    def w2c(self):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.R, self.t
        return m

    @property
    def c2w(self):
        return np.linalg.inv(self.w2c)

    @property
    def center(self):
        return -self.R.T @ self.t

    @property
    def forward(self):
        return self.R[2]

    @property
    def up(self):
        return -self.R[1]

    def load_image(self):
        """(H, W, 3) uint8 rgb at the training resolution, cached."""
        return read_image(str(self.image_path), self.width, self.height)

    def load_mask(self):
        """(H, W) bool foreground mask, or None if there is no mask file."""
        if self.mask_path is None or not self.mask_path.exists():
            return None
        return read_image(str(self.mask_path), self.width, self.height)[..., 0] > 127


@lru_cache(maxsize=512)
def read_image(path, width, height):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img.setflags(write=False)
    return img


class Scene:

    def __init__(self, name, dataset, root, cameras, bounds, ref_points=None):
        self.name = name
        self.dataset = dataset
        self.root = root
        self.cameras = cameras
        self.bounds = bounds
        self.ref_points = ref_points
        self._by_id = {c.image_id: c for c in cameras}
        self._by_name = {c.name: c for c in cameras}

    def __len__(self):
        return len(self.cameras)

    def get_camera(self, image_id):
        return self._by_id[image_id]

    def by_name(self, name):
        return self._by_name[name]

    def ids(self, names):
        return [self._by_name[n].image_id for n in names]

    def names(self, ids):
        return [self._by_id[i].name for i in ids]

    @property
    def centers(self):
        return np.stack([c.center for c in self.cameras])

    @property
    def centroid(self):
        """Median of the reference points, or the optical axes intersection (dtu)."""
        if self.ref_points is not None and len(self.ref_points):
            return np.median(self.ref_points, axis=0)
        return axes_intersection(self.centers, np.stack([c.forward for c in self.cameras]))

    def camera_extent(self, image_ids):
        """3dgs getNerfppNorm radius of the given (training) cameras."""
        c = np.stack([self.get_camera(i).center for i in image_ids])
        return float(1.1 * np.linalg.norm(c - c.mean(0), axis=1).max())


def axes_intersection(origins, dirs):
    """Point closest (least squares) to a set of lines."""
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    P = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    return np.linalg.solve(P.sum(0), (P @ origins[:, :, None]).sum(0))[:, 0]


def load_scene(data_root, dataset, scene, image_factor=None):
    """Load one scene of llff / mip360 / dtu."""
    factor = image_factor or DATASETS[dataset]["image_factor"]
    data_root = Path(data_root)
    if dataset == "dtu":
        return load_dtu(data_root / "dtu", scene, factor, DATASETS["dtu"]["light"])
    return load_colmap(data_root / dataset / scene, dataset, factor)


def load_colmap(root, dataset, factor):
    """Llff / mip360: poses + intrinsics from sparse/0, bounds from poses_bounds.npy."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(root / "sparse" / "0"))
    images = sorted(rec.images.values(), key=lambda im: im.name)

    # images_8/ names differ from images/ on fern, so pair by sorted order like fsgs
    img_dir = root / (f"images_{factor}" if factor > 1 else "images")
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    h, w = cv2.imread(str(files[0])).shape[:2]

    cameras = []
    for i, (im, path) in enumerate(zip(images, files)):
        cam = rec.cameras[im.camera_id]
        p = list(cam.params)
        if cam.model.name in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx, fy, cx, cy = p[:4]
        else:
            fx, fy, cx, cy = p[0], p[0], p[1], p[2]
        sx, sy = w / cam.width, h / cam.height
        pose = im.cam_from_world()
        cameras.append(Camera(
            i, im.name, path, np.asarray(pose.rotation.matrix()), np.asarray(pose.translation),
            fx * sx, fy * sy, cx * sx, cy * sy, w, h,
            sfm_image_path=root / "images" / im.name,
            sfm_camera={"model": cam.model.name, "width": cam.width, "height": cam.height,
                        "params": [float(v) for v in cam.params]},
        ))

    pb = np.load(root / "poses_bounds.npy")
    bounds = (float(pb[:, -2].min()), float(pb[:, -1].max()))
    ref = np.array([p.xyz for p in rec.points3D.values()])
    return Scene(root.name, dataset, root, cameras, bounds, ref)


def load_dtu(root, scan, factor, light):
    """Dtu like the regnerf loader: decompose cal18 matrices, recenter, rescale."""
    c2ws, Ks, paths = [], [], []
    for pos in range(1, 50):
        P = np.loadtxt(root / "Calibration" / "cal18" / f"pos_{pos:03d}.txt")
        K, R, C = cv2.decomposeProjectionMatrix(P)[:3]
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R.T, (C[:3] / C[3])[:, 0]
        c2ws.append(c2w)
        Ks.append(K / K[2, 2])
        paths.append(root / "Rectified" / scan / f"rect_{pos:03d}_{light}_r5000.png")

    c2ws = dtu_normalise(np.stack(c2ws))
    w, h = 1600 // factor, 1200 // factor
    mask_dir = root / "idrmasks" / scan / ("mask" if scan in DTU_IDR_SCANS else "")

    cameras = []
    for i, (c2w, K, path) in enumerate(zip(c2ws, Ks, paths)):
        w2c = np.linalg.inv(c2w)
        cameras.append(Camera(
            i, path.name, path, w2c[:3, :3], w2c[:3, 3],
            K[0, 0] / factor, K[1, 1] / factor, K[0, 2] / factor, K[1, 2] / factor, w, h,
            factor=factor, mask_path=mask_dir / f"{i:03d}.png", sfm_image_path=path,
            sfm_camera={"model": "PINHOLE", "width": 1600, "height": 1200,
                        "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]},
        ))
    return Scene(scan, "dtu", root / "Rectified" / scan, cameras, DTU_BOUNDS)


def dtu_normalise(c2ws):
    """Regnerf recenter_poses + rescale_poses on (N, 4, 4) camera-to-world."""
    def unit(v):
        return v / np.linalg.norm(v)

    z = unit(c2ws[:, :3, 2].mean(0))
    x = unit(np.cross(c2ws[:, :3, 1].mean(0), z))
    avg = np.eye(4)
    avg[:3] = np.stack([x, unit(np.cross(z, x)), z, c2ws[:, :3, 3].mean(0)], axis=1)
    out = np.linalg.inv(avg) @ c2ws
    out[:, :3, 3] /= np.abs(out[:, :3, 3]).max()
    u, _, vt = np.linalg.svd(out[:, :3, :3])
    out[:, :3, :3] = u @ vt
    return out
