import itertools
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from splataudit.data import DATASETS
from splataudit.utils import (
    RUN_KEYS,
    append_rows,
    claim_run,
    config_hash,
    finished_runs,
    git_commit,
    notify,
    run_key,
    run_tag,
)

RUN_MINUTES = {"llff": 4.0, "dtu": 3.0, "mip360": 8.0}
METHOD_COST = {"vanilla": 1.0, "dropgaussian": 1.05, "fsgs": 1.1, "dngaussian": 2.6}


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def with_defaults(cfg):
    """Fill n_views / scenes from the dataset, drop k for p0."""
    ds = DATASETS[cfg["dataset"]]
    cfg = {
        **cfg,
        "n_views": cfg.get("n_views") or ds["n_views"],
        "scenes": cfg.get("scenes") or ds["scenes"],
    }
    if cfg.get("protocol") == "p0":
        cfg["k"] = None
    return cfg


def run_cell(cfg):
    """Train + evaluate one cell, every outcome (also failures) becomes rows in results.csv."""
    from splataudit.data import load_scene
    from splataudit.splits import load_split, resolve, split_path

    cfg = with_defaults(cfg)
    out_dir = Path(cfg["output_dir"])
    run_dir = out_dir / "runs" / run_tag(cfg)
    base = {k: cfg.get(k) for k in RUN_KEYS} | {
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
    }

    t0 = time.time()
    try:
        from splataudit.utils import seed_everything

        seed_everything(int(cfg["seed"]))

        scene = load_scene(cfg["data_root"], cfg["dataset"], cfg["scene"])
        sp = split_path(
            cfg["splits_root"],
            cfg["dataset"],
            cfg["scene"],
            cfg["protocol"],
            cfg["n_views"],
            cfg.get("k"),
        )
        record = load_split(sp)
        ids = resolve(scene, record)
        method = build_method(cfg)

        from splataudit.init import get_init

        xyz, rgb = get_init(
            cfg["init"], scene, ids["train"], sp.stem, int(cfg["seed"]), cfg["cache_root"]
        )

        depths = confidence_maps = None
        if method.requires_depth:
            from splataudit.priors import depth_prior

            depths = depth_prior(scene, ids["train"], cfg["cache_root"])
        if method.requires_confidence:
            from splataudit.priors import confidence

            confidence_maps = confidence(scene, ids["train"], depths)

        from splataudit.eval import evaluate
        from splataudit.train import train

        state = train(cfg, scene, ids["train"], xyz, rgb, method, depths, confidence_maps)
        train_time = round(time.time() - t0, 1)
        rows = evaluate(cfg, scene, state, record)
        for r in rows:
            r.update(
                base, n_gaussians=int(state["n_gaussians"]), train_time_s=train_time, status="ok"
            )
    except NotImplementedError as e:
        rows = [base | {"status": "not_implemented", "error": str(e)}]
    except Exception as e:
        rows = [base | {"status": "failed", "error": f"{type(e).__name__}: {e}"}]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "traceback.txt").write_text(traceback.format_exc())

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
    append_rows(out_dir / "results.csv", rows)
    return rows


def plan_sweep(cfg):
    """Every cell of the sweep grid, in the order they would run."""
    from splataudit.splits import SECTOR_POSITIONS

    sw = cfg["sweep"]
    cells = []
    for dataset in sw.get("datasets") or [cfg["dataset"]]:
        base = with_defaults({**cfg, "dataset": dataset})
        protocols = sw["protocols"]
        protocols = protocols.get(dataset, []) if isinstance(protocols, dict) else protocols
        protocol_k = []
        for protocol in protocols:
            if protocol == "p0":
                protocol_k.append(("p0", None))
            else:
                n = min(sw.get("p2_positions") or 99, SECTOR_POSITIONS[dataset])
                protocol_k += [("p2", k) for k in range(n)]
        for scene, (protocol, k), init, method, seed in itertools.product(
            base["scenes"], protocol_k, sw["inits"], sw["methods"], sw["seeds"]
        ):
            cells.append(
                base
                | {
                    "scene": scene,
                    "protocol": protocol,
                    "k": k,
                    "init": init,
                    "method": method,
                    "seed": seed,
                }
            )
    return cells


def estimate_hours(cells, results_csv=None):
    """Rough wall clock for a list of cells, from measured train_time_s when results exist."""
    measured = {}
    if results_csv and Path(results_csv).exists():
        df = pd.read_csv(results_csv)
        ok = df[df["status"] == "ok"].drop_duplicates(RUN_KEYS)
        if not ok.empty:
            measured = (ok.groupby(["dataset", "method"])["train_time_s"].mean() / 60).to_dict()
    minutes = 0.0
    for c in cells:
        fallback = RUN_MINUTES[c["dataset"]] * METHOD_COST.get(c["method"], 1.0)
        minutes += measured.get((c["dataset"], c["method"]), fallback)
    return minutes / 60


def sweep(cfg, max_runs=None, shard=None):
    """Run the sweep grid, skipping cells that already have rows or are claimed by another worker."""
    out_dir = Path(cfg["output_dir"])
    results_csv = out_dir / "results.csv"
    all_cells = plan_sweep(cfg)
    if shard is not None:
        idx, total = shard
        all_cells = all_cells[idx::total]

    done = finished_runs(results_csv)
    cells = [c for c in all_cells if run_key(c) not in done]

    worker = os.environ.get("SPLATAUDIT_WORKER")
    prefix = f"[gpu={worker}] " if worker is not None else ""

    n_rows = 0
    ran = 0
    for i, cell in enumerate(cells, 1):
        if max_runs and ran >= max_runs:
            break
        if run_key(cell) in finished_runs(results_csv):
            continue
        cell_cfg = with_defaults(cell)
        run_dir = out_dir / "runs" / run_tag(cell_cfg)
        with claim_run(run_dir) as claimed:
            if not claimed or run_key(cell) in finished_runs(results_csv):
                continue
            print(f"{prefix}[{i}/{len(cells)}] [start] {run_tag(cell_cfg)}", flush=True)
            rows = run_cell(cell)
            ran += 1
            n_rows += len(rows)
            status = rows[0]["status"]
            detail = f" {rows[0].get('error')}" if status != "ok" else _format_ok_summary(rows)
            msg = f"{prefix}[{i}/{len(cells)}] [{status}] {run_tag(cell_cfg)}{detail}"
            print(msg, flush=True)
            notify(msg)
    return n_rows


def _format_ok_summary(rows):
    """Short metric summary for a finished cell (mean test PSNR, Gaussian count, seconds)."""
    test_psnr = [float(r["psnr"]) for r in rows if str(r.get("in_test")) == "True" and "psnr" in r]
    psnr_str = f" psnr={np.mean(test_psnr):.2f}dB" if test_psnr else ""
    t_str = f" {rows[0].get('train_time_s', '?')}s"
    g_str = f" n={rows[0].get('n_gaussians', '?')}"
    return f"{psnr_str}{g_str}{t_str}"


def report(df, metrics=("psnr", "ssim", "lpips_vgg")):
    """Descriptive summary per cell, scene is the unit (not the §8 analysis)."""
    cell = ["dataset", "n_views", "protocol", "init", "method"]
    status = (
        df.drop_duplicates(RUN_KEYS).groupby(cell)["status"].value_counts().unstack(fill_value=0)
    )

    ok = df[(df["status"] == "ok") & (df["in_test"].astype(str) == "True")]
    if ok.empty:
        return status
    metrics = [m for m in metrics if m in ok.columns]
    per_run = ok.groupby(RUN_KEYS, dropna=False)[metrics].mean().reset_index()
    per_scene = per_run.groupby([*cell, "scene"], dropna=False)[metrics].agg(["mean", "std"])

    out = {}
    for m in metrics:
        g = per_scene[(m, "mean")].groupby(level=cell)
        out[(m, "mean")] = g.mean()
        out[(m, "std_scenes")] = g.std()
        out[(m, "std_seeds")] = per_scene[(m, "std")].groupby(level=cell).mean()
    summary = pd.DataFrame(out)
    summary[("n", "scenes")] = per_scene.groupby(level=cell).size()
    status.columns = pd.MultiIndex.from_product([["status"], status.columns])
    return summary.join(status, how="outer").round(3)


def build_method(cfg):
    """Method instance for cfg["method"], hyperparameters taken from cfg["methods"][name]."""
    from splataudit import methods

    name = cfg["method"]
    if name not in methods.METHODS:
        raise ValueError(f"unknown method {name}")
    params = {
        **(cfg.get("methods") or {}).get(name, {}),
        "dataset": cfg["dataset"],
        "seed": cfg["seed"],
        "iterations": cfg["train"]["iterations"],
    }
    return methods.build(name, params)
