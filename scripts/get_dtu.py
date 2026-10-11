# pull single DTU scans out of the remote archive
# the dtu server supports http range requests, so zipfile can read the central directory and fetch only the entries we want.
#
#  python scripts/get_dtu.py list
#  python scripts/get_dtu.py get 8 21 30 34 - 1568 files(3 GB)
#  python scripts/get_dtu.py get 8 --light max - one lighting(0.5 GB)

import io
import sys
import zipfile
import requests
from pathlib import Path

BASE = "https://roboimagedata2.compute.dtu.dk/data/MVS"
OUT = Path("data/dtu")
STANDARD = [8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114] # few-shot set


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        h = requests.head(url, allow_redirects=True, timeout=30).headers
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
        r = requests.get(self.url, headers={"Range": rng}, timeout=120)
        r.raise_for_status()
        self.pos += len(r.content)
        return r.content
    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
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


def get(scans, light=None):
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
    for name, keep in (("Rectified.zip", is_img), ("Points.zip", is_cloud)):
        z = open_zip(name)
        sel = [n for n in z.namelist() if keep(n)]
        gb = sum(z.getinfo(n).file_size for n in sel) / 2 ** 30
        print(f"{name}: {len(sel)} files, {gb:.2f} GB")
        if not sel:
            print("  nothing matched, run `list` and fix the filter")
            continue
        for i, n in enumerate(sel, 1):
            z.extract(n, OUT)
            if i % 25 == 0 or i == len(sel):
                print(f"  {i}/{len(sel)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    light = None
    if "--light" in args:
        i = args.index("--light")
        light, args = args[i + 1], args[:i] + args[i + 2:]
    if not args or args[0] == "list":
        list_archives()
    elif args[0] == "get":
        get([int(a) for a in args[1:]] or STANDARD[:4], light)
    else:
        print("usage: get_dtu.py list | get <scan>... [--light max]")
