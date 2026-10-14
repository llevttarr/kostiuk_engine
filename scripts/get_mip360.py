# pull single mip-nerf 360 scenes
#
#  python scripts/get_mip360.py list
#  python scripts/get_mip360.py get bicycle garden room counter - 7.6 GB
#
import io
import sys
import zipfile
from pathlib import Path
from get_dtu import HttpFile
BASE = "http://storage.googleapis.com/gresearch/refraw360"
OUT = Path("data/mip360")

ZIPS = {"360_v2.zip": ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"],
        "360_extra_scenes.zip": ["flowers", "treehill"]}
SIZES = {"bicycle": 2.33, "bonsai": 1.34, "counter": 1.16, "garden": 2.80, "kitchen": 1.49,
         "room": 1.32, "stump": 1.44, "flowers": 2.43, "treehill": 1.79}
STANDARD = ["bicycle", "garden", "room", "counter"]
def open_zip(name):
    f = io.BufferedReader(HttpFile(f"{BASE}/{name}"), buffer_size=1 << 20)
    return zipfile.ZipFile(f)
def list_archives():
    for name, scenes in ZIPS.items():
        z = open_zip(name)
        names = z.namelist()
        print(f"\n{name}: {len(names)} entries")
        for s in scenes:
            print(f"  {s:10s} {SIZES[s]:5.2f} GB")
        for n in names[:5]:
            print("   ", n)
def get(scenes):
    want = set(scenes)
    unknown = want - set(SIZES)
    if unknown:
        print("unknown scenes:", sorted(unknown), "- run `list`")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    for name, have in ZIPS.items():
        pick = want & set(have)
        if not pick:
            continue
        z = open_zip(name)
        sel = [n for n in z.namelist() if n.split("/")[0] in pick]
        gb = sum(z.getinfo(n).file_size for n in sel) / 2 ** 30
        print(f"{name}: {sorted(pick)}, {len(sel)} files, {gb:.2f} GB")
        for i, n in enumerate(sel, 1):
            z.extract(n, OUT)
            if i % 50 == 0 or i == len(sel):
                print(f"  {i}/{len(sel)}")
if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "list":
        list_archives()
    elif args[0] == "get":
        get(args[1:] or STANDARD)
    else:
        print("usage: get_mip360.py list | get <scene>...")
