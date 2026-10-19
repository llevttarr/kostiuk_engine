# KOSTIUK engine

## PRECOMMIT AND STYLING !

```bash
# precommit config (after git clone one time)
pip install pre-commit && pre-commit install
# after this just commit, ruff will work on its own


# manual 
ruff format src/        # format
ruff check src/         # lint
ruff check src/ --fix   # lint + autocorrect
```

**please dont avoid precommit warnings**

## usage

```bash
# data (writes to data/, run from the repo root), about 12 GB
bash scripts/get_all.sh

# or one dataset at a time
bash scripts/get_llff.sh
python scripts/get_dtu.py calib
python scripts/get_dtu.py get --light 3
python scripts/get_dtu.py masks
python scripts/get_mip360.py get

# splits: generated once and committed (splits/)
PYTHONPATH=src python -m splataudit make-splits
# inits: cached in cache/ (git-ignored), one per (scene, split), shared by every method
PYTHONPATH=src python -m splataudit make-inits --init random sparse_sfm foundation --protocol p0 p2

# the grid: 372 runs, resumable
PYTHONPATH=src python -m splataudit sweep --dry-run
PYTHONPATH=src python -m splataudit sweep
# parallel across 2 GPUs with live push notifications via ntfy.sh:
PYTHONPATH=src python -m splataudit sweep --gpus 0 1 --notify kostiuk_splataudit
PYTHONPATH=src python -m splataudit report --results results/results.csv

# tests (164, all cpu)
python -m pytest
```

## Code layout

| Module | What |
|---|---|
| `data.py` | `Camera`, `Scene`, loaders for LLFF, Mip-360 and DTU |
| `splits.py` | P0 / P2 splits, angular distances (P1) |
| `init.py` | random, sparse SfM, dense MVS and VGGT inits, Sim(3) alignment, PLY I/O, init cache |
| `utils.py` | seeding, camera sampler, results CSV |
| `train.py`, `eval.py`, `priors.py` | training loop, metrics, depth priors |
| `methods.py` | vanilla, FSGS, DNGaussian, DropGaussian (CrossViewGating still raises) |
| `runner.py`, `cli.py` | one run / sweep / report, command line |

The GPU machine has its own sequence in `GPU_SETUP.md`. The method axis, its constants and
every deviation from the released code are in `BENCHMARK_PLAN.md` §6.

## Memory note

`sparse_sfm` with FSGS's settings (4032 px covariant SIFT, guided matching) needs a lot of
RAM on CPU and ran the 14 GB dev laptop out of memory, so build it on the GPU machine.
`SPLATAUDIT_SFM=smoke` runs a light version for pipeline checks only. Training does not
refuse smoke inits, so always point smoke runs at a separate `--cache` directory.
