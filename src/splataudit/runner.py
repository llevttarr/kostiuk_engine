import itertools
import json
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

from splataudit.data import DATASETS
from splataudit.utils import (
    RUN_KEYS,
    append_rows,
    config_hash,
    finished_runs,
    git_commit,
    run_key,
    run_tag,
)


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
            from splataudit.priors import confidence, depth_prior

            depths = depth_prior(scene, ids["train"], cfg["cache_root"])
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


def sweep(cfg):
    """Run the cfg["sweep"] grid, skipping runs that already have ok rows."""
    from splataudit.splits import SECTOR_POSITIONS

    cfg = with_defaults(cfg)
    sw = cfg["sweep"]
    done = finished_runs(Path(cfg["output_dir"]) / "results.csv")

    protocol_k = []
    for protocol in sw["protocols"]:
        if protocol == "p0":
            protocol_k.append(("p0", None))
        else:
            protocol_k += [("p2", k) for k in range(SECTOR_POSITIONS[cfg["dataset"]])]

    n_rows = 0
    for scene, (protocol, k), init, method, seed in itertools.product(
        cfg["scenes"], protocol_k, sw["inits"], sw["methods"], sw["seeds"]
    ):
        cell = cfg | {
            "scene": scene,
            "protocol": protocol,
            "k": k,
            "init": init,
            "method": method,
            "seed": seed,
        }
        if run_key(cell) in done:
            continue
        rows = run_cell(cell)
        n_rows += len(rows)
        status = rows[0]["status"]
        print(
            f"[{status}] {run_tag(cell)}" + (f" {rows[0].get('error')}" if status != "ok" else "")
        )
    return n_rows


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
    return summary.join(status, how="outer").round(3)


def build_method(cfg):
    from splataudit import methods as M

    name = cfg["method"]
    if name == "vanilla":
        return M.Vanilla()
    if name == "fsgs":
        return M.FSGS()
    if name == "dngaussian":
        return M.DNGaussian()
    if name == "dropgaussian":
        return M.DropGaussian(drop_rate=cfg.get("drop_rate", 0.1))
    if name == "cross_view_gating":
        return M.CrossViewGating(confidence_threshold=cfg.get("confidence_threshold", 0.5))
    raise ValueError(f"unknown method {name}")
