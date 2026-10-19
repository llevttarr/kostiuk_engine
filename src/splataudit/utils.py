import csv
import fcntl
import hashlib
import json
import os
import random
import subprocess
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np

RUN_KEYS = ["dataset", "scene", "n_views", "protocol", "k", "init", "method", "seed"]

FIELDS = [
    *RUN_KEYS,
    "test_image",
    "in_test",
    "ang_dist_centre",
    "ang_dist_optical",
    "dist_bin",
    "psnr",
    "ssim",
    "lpips_vgg",
    "lpips_alex",
    "masked",
    "psnr_masked",
    "ssim_masked",
    "lpips_vgg_masked",
    "lpips_alex_masked",
    "floater_proxy",
    "n_gaussians",
    "train_time_s",
    "git_commit",
    "config_hash",
    "status",
    "error",
]


def seed_everything(seed):
    """Seed random, numpy and torch (cpu + cuda)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CameraSampler:
    """3dgs camera order: a seeded permutation of the train views every epoch."""

    def __init__(self, items, seed):
        self.items = list(items)
        self.rng = np.random.default_rng([seed, 0x5EED])
        self.queue = []

    def next(self):
        if not self.queue:
            self.queue = list(self.rng.permutation(len(self.items)))
        return self.items[self.queue.pop()]


def run_key(cfg):
    return tuple("" if cfg.get(k) is None else str(cfg.get(k)) for k in RUN_KEYS)


def run_tag(cfg):
    """Run directory name, includes every run key."""
    k = "" if cfg.get("k") is None else f"_k{cfg['k']}"
    return (
        f"{cfg['dataset']}/{cfg['scene']}/{cfg['protocol']}_{cfg['n_views']}v{k}/"
        f"init={cfg['init']}__method={cfg['method']}__seed={cfg['seed']}"
    )


def append_rows(path, rows):
    """Append result rows to the csv with an exclusive file lock so multi-GPU workers never race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            new = f.tell() == 0
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def finished_runs(path):
    """Run keys that already have ok rows, for resuming a sweep."""
    if not Path(path).exists():
        return set()
    with open(path, newline="") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return {
                tuple(r[k] for k in RUN_KEYS) for r in csv.DictReader(f) if r.get("status") == "ok"
            }
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def claim_run(run_dir):
    """Non-blocking OS lock on a run directory so parallel GPU workers skip cells in progress."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(run_dir / ".lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claimed = True
    except BlockingIOError:
        claimed = False
    try:
        yield claimed
    finally:
        if claimed:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def notify(message, target=None):
    """Send a best-effort live status message to ntfy.sh or a Discord/Slack webhook."""
    target = target or os.environ.get("SPLATAUDIT_NOTIFY")
    if not target:
        return
    url = target if target.startswith(("http://", "https://")) else f"https://ntfy.sh/{target}"
    try:
        if "discord.com/api/webhooks" in url or "hooks.slack.com" in url:
            data = json.dumps({"content": message, "text": message}).encode()
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
        else:
            req = urllib.request.Request(
                url, data=message.encode(), headers={"Title": "splataudit"}
            )
        urllib.request.urlopen(req, timeout=5).close()
    except Exception:
        pass


def config_hash(cfg):
    """Short hash of the config without run keys and paths."""
    skip = set(RUN_KEYS) | {"output_dir", "data_root", "splits_root", "cache_root", "scenes"}
    blob = json.dumps({k: v for k, v in cfg.items() if k not in skip}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def git_commit():
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
