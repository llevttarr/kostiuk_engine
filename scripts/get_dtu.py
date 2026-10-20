# pull single DTU scans out of the remote archive
# the dtu server supports http range requests, so zipfile can read the central directory and fetch only the entries we want.
#
#  python scripts/get_dtu.py list
#  python scripts/get_dtu.py get 8 21 30 34 --light 3 - benchmark lighting only (~0.1 GB/scan)
#  python scripts/get_dtu.py get 8 21 30 34 - all lightings (~0.75 GB/scan)
#  python scripts/get_dtu.py get --light 3 - all 15 standard scans, lighting 3
#  python scripts/get_dtu.py calib - cal18 projection matrices (49 small txt files)
#  python scripts/get_dtu.py masks - RegNeRF idrmasks for evaluation (google drive)
#  python scripts/get_dtu.py get ... --points - also the STL ground-truth clouds

import io
import shutil
import sys
import zipfile
from pathlib import Path

import requests

BASE = "https://roboimagedata2.compute.dtu.dk/data/MVS"
OUT = Path("data/dtu")
STANDARD = [8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]  # few-shot set
# RegNeRF/DNGaussian evaluation masks ("backed up from RegNeRF", DNGaussian README)
MASKS_GDRIVE = "https://drive.google.com/drive/folders/1OEmJcbP0XUVfG647mdYWxEy9yOw7L_Si"


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.session = requests.Session()
        h = self.session.head(url, allow_redirects=True, timeout=30).headers
        self.size = int(h["Content-Length"])
        assert h.get("Accept-Ranges") == "bytes", "server has no range support"

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        rng = f"bytes={self.pos}-{self.pos + n - 1}"
        r = self.session.get(self.url, headers={"Range": rng}, timeout=120)
        r.raise_for_status()
        self.pos += len(r.content)
        return r.content

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_zip(name):
    f = io.BufferedReader(HttpFile(f"{BASE}/{name}"), buffer_size=1 << 20)
    return zipfile.ZipFile(f)


def list_archives():
    for name in ("Rectified.zip", "Points.zip"):
        z = open_zip(name)
        names = z.namelist()
        folders = sorted({n.split("/")[1] for n in names if n.count("/") > 1})
        print(f"\n{name}: {len(names)} entries, {len(folders)} folders")
        for n in names[:6]:
            print("  ", n)
        print("  folders:", folders[:8], "...")


def get(scans, light=None, points=False):
    # images: Rectified/scan8/rect_<pos>_<light>_r5000.png, light in 0..6 and "max"
    # clouds: Points/stl/stl008_total.ply
    want = {f"scan{s}" for s in scans}

    def is_img(n):
        if not n.endswith(".png") or not any(f"/{w}/" in n for w in want):
            return False
        return light is None or Path(n).stem.split("_")[2] == light

    def is_cloud(n):
        return any(f"stl{s:03d}_" in Path(n).name for s in scans)

    OUT.mkdir(parents=True, exist_ok=True)
    archives = [("Rectified.zip", is_img)] + ([("Points.zip", is_cloud)] if points else [])
    for name, keep in archives:
        z = open_zip(name)
        sel = [n for n in z.namelist() if keep(n)]
        gb = sum(z.getinfo(n).file_size for n in sel) / 2**30
        print(f"{name}: {len(sel)} files, {gb:.2f} GB", flush=True)
        if not sel:
            print("  nothing matched, run `list` and fix the filter", flush=True)
            continue
        for i, n in enumerate(sel, 1):
            z.extract(n, OUT)
            if i % 25 == 0 or i == len(sel):
                print(f"  {i}/{len(sel)}", flush=True)


def get_calib():
    # cal18 = calibration of the 1600x1200 rectified images (what RegNeRF loads).
    # it only ships inside SampleSet.zip, so range-read just those entries
    z = open_zip("SampleSet.zip")
    sel = [n for n in z.namelist() if "/Calibration/cal18/pos_" in n and n.endswith(".txt")]
    out = OUT / "Calibration" / "cal18"
    out.mkdir(parents=True, exist_ok=True)
    for n in sel:
        (out / Path(n).name).write_bytes(z.read(n))
    print(f"{len(sel)} projection matrices -> {out}")


def get_masks():
    """Regnerf's eval masks, unpacked to data/dtu/idrmasks/<scan>/."""
    import gdown

    tmp = OUT / "masks_download"
    if not list(tmp.glob("*.zip")):
        gdown.download_folder(MASKS_GDRIVE, output=str(tmp), quiet=False)
    n = 0
    for archive in sorted(tmp.glob("*.zip")):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if "/idrmasks/" not in name or name.endswith("/"):
                    continue
                dst = OUT / "idrmasks" / name.split("/idrmasks/", 1)[1]
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(name))
                n += 1
    shutil.rmtree(tmp, ignore_errors=True)
    scans = sorted(p.name for p in (OUT / "idrmasks").glob("scan*"))
    print(f"{n} masks for {len(scans)} scans -> {OUT / 'idrmasks'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    light = None
    if "--light" in args:
        i = args.index("--light")
        light, args = args[i + 1], args[:i] + args[i + 2 :]
    points = "--points" in args
    args = [a for a in args if a != "--points"]
    if not args or args[0] == "list":
        list_archives()
    elif args[0] == "get":
        get([int(a) for a in args[1:]] or STANDARD, light, points)
    elif args[0] == "calib":
        get_calib()
    elif args[0] == "masks":
        get_masks()
    else:
        print("usage: get_dtu.py list | get [<scan>...] [--light 3] [--points] | calib | masks")
