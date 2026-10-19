#!/usr/bin/env bash
# every dataset the benchmark needs, run from the repo root (about 12 GB)
set -eu

bash scripts/get_llff.sh

python scripts/get_dtu.py calib
python scripts/get_dtu.py get --light 3
python scripts/get_dtu.py masks

# full images, not --small: sparse_sfm runs colmap on images/
python scripts/get_mip360.py get

python -m splataudit make-splits

du -sh data/*
