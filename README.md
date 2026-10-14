# KOSTIUK engine
## usage

```bash
# data (writes to data/, run from the repo root)
bash scripts/get_llff.sh
python scripts/get_dtu.py calib                 # cal18 cameras
python scripts/get_dtu.py get --light 3         # 15 standard scans, lighting 3
python scripts/get_dtu.py masks                 # RegNeRF eval masks
python scripts/get_mip360.py get                # bicycle garden room counter

# splits: generated once and committed (splits/)
PYTHONPATH=src python -m splataudit make-splits
# inits: cached in cache/ (git-ignored)
PYTHONPATH=src python -m splataudit make-inits --init random sparse_sfm

# tests
python -m pytest
```

## Code layout

| Module | What |
|---|---|
| `data.py` | `Camera`, `Scene`, loaders for LLFF, Mip-360 and DTU |
| `splits.py` | P0 / P2 splits, angular distances (P1) |
| `init.py` | random, sparse SfM, dense MVS and foundation inits, PLY I/O, init cache |
| `utils.py` | seeding, camera sampler, results CSV |
| `train.py`, `eval.py`, `methods.py`, `priors.py` | training loop, metrics, methods, depth priors |
| `runner.py`, `cli.py` | one run / sweep / report, command line |

## Memory note

`sparse_sfm` with FSGS's settings (4032 px covariant SIFT, guided matching) needs a lot of
RAM on CPU and ran the 14 GB dev laptop out of memory, so build it on the GPU machine.
`SPLATAUDIT_SFM=smoke` runs a light version for pipeline checks only. Training does not
refuse smoke inits, so always point smoke runs at a separate `--cache` directory.
