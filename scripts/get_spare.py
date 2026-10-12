# spare (AIM 2024 sparse neural rendering challenge), from huggingface
#  python scripts/get_spare.py list
#  python scripts/get_spare.py get - eval files only, 0.8 GB
#  python scripts/get_spare.py get --train - adds the 150 GB training sets, don't
# useful detail from their docs: lighting 3 is the condition used for evaluation
import sys
import zipfile
from pathlib import Path

import requests

BASE = ("https://huggingface.co/datasets/aim2024-sparse-rendering/AIM2024-SparseNeuralRendering/resolve/main")
OUT = Path("data/spare")

EVAL = [
    "Track1_split_development.json", # 3 views, validation
    "Track1_split_test.json", # 3 views, test
    "Track2_split_development.json", # 9 views, validation
    "Track2_split_test.json", # 9 views, test
    "synth_calibration.zip", # projection matrices, regnerf-style loader
    "dtu_calibration.zip",
    "synth_development_3views.zip",
    "synth_development_9views.zip",
    "synth_test_3views.zip",
    "synth_test_9views.zip",
    "dtu_development_3views.zip",
    "dtu_development_9views.zip",
    "dtu_test_3views.zip",
    "dtu_test_9views.zip",
]
TRAIN = [
    "synth_training_p1.zip", # 33.3 GB
    "synth_training_p2.zip", # 30.5 GB
    "dtu_training_p1.zip", # 44.4 GB
    "dtu_training_p2.zip", # 41.7 GB
]
def size_of(name):
    h = requests.head(f"{BASE}/{name}", allow_redirects=True, timeout=30).headers
    return int(h.get("Content-Length", 0))
def list_files():
    for group, names in (("eval", EVAL), ("train", TRAIN)):
        total = 0
        print(f"\n{group}:")
        for n in names:
            b = size_of(n)
            total += b
            print(f"  {n:34s} {b / 2 ** 20:9.1f} MB")
        print(f"  {'total':34s} {total / 2 ** 30:9.2f} GB")
def fetch(name):
    dst = OUT / name
    if dst.exists():
        print(f"  {name} already here")
        return dst
    r = requests.get(f"{BASE}/{name}", stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    done = 0
    with open(dst, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {name} {100 * done // total}%", end="")
    print(f"\r  {name} done ({done / 2 ** 20:.1f} MB)")
    return dst

def get(with_train=False):
    OUT.mkdir(parents=True, exist_ok=True)
    names = EVAL + (TRAIN if with_train else [])
    if with_train:
        print("warning ts is 150gb")
    for n in names:
        p = fetch(n)
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                z.extractall(OUT)
            p.unlink()

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "list":
        list_files()
    elif args[0] == "get":
        get("--train" in args)
    else:
        print("usage: get_spare.py list | get [--train]")
