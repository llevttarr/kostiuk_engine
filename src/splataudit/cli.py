import argparse

import numpy as np

from splataudit.data import DATASETS

INITS = ["random", "sparse_sfm", "dense_mvs", "foundation"]
METHODS = ["vanilla", "fsgs", "dngaussian", "dropgaussian", "cross_view_gating"]


def main():
    parser = argparse.ArgumentParser(
        prog="splataudit", description="few-shot 3dgs audit: init x method x protocol"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-splits", help="write p0 / p2 split files")
    add_data_args(p)
    p.add_argument("--out", default="splits")

    p = sub.add_parser("make-inits", help="build and cache initial point clouds")
    add_data_args(p)
    p.add_argument("--splits", default="splits")
    p.add_argument("--cache", default="cache")
    p.add_argument("--init", nargs="+", default=["random", "sparse_sfm"], choices=INITS)
    p.add_argument("--protocol", nargs="+", default=["p0", "p2"], choices=["p0", "p2"])
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2], help="for the seeded inits (random)"
    )
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("run", help="run one experiment cell")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset", choices=list(DATASETS))
    p.add_argument("--scene")
    p.add_argument("--init", choices=INITS)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--protocol", choices=["p0", "p2"])
    p.add_argument("--k", type=int, help="p2 sector position")
    p.add_argument("--seed", type=int)
    p.add_argument("--output-dir")

    p = sub.add_parser("sweep", help="run the init x method x protocol grid")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output-dir")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS))
    p.add_argument("--scenes", nargs="+", help="only with a single dataset")
    p.add_argument("--inits", nargs="+", choices=INITS)
    p.add_argument("--methods", nargs="+", choices=METHODS)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--protocols", nargs="+", choices=["p0", "p2"])
    p.add_argument("--max-runs", type=int, help="stop after this many new runs")
    p.add_argument("--dry-run", action="store_true", help="print the grid and the estimate only")

    p = sub.add_parser("report", help="per-scene summary of a results csv")
    p.add_argument("--results", required=True)
    p.add_argument("--latex", action="store_true")

    args = parser.parse_args()
    {
        "make-splits": make_splits,
        "make-inits": make_inits,
        "run": run,
        "sweep": sweep,
        "report": report,
    }[args.command](args)


def add_data_args(p):
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--scenes", nargs="+", help="default: the dataset's benchmark scenes")
    p.add_argument("--n-views", type=int, help="default: the dataset's primary count")


def scene_jobs(args):
    from splataudit.splits import SECTOR_POSITIONS

    for ds in args.dataset:
        jobs = [("p0", None)] + [("p2", k) for k in range(SECTOR_POSITIONS[ds])]
        for name in args.scenes or DATASETS[ds]["scenes"]:
            yield ds, name, args.n_views or DATASETS[ds]["n_views"], jobs


def make_splits(args):
    from splataudit.data import load_scene
    from splataudit.splits import make_split, save_split, split_path

    print(
        f"{'dataset':7s} {'scene':9s} {'N':>4s}  split      {'#test':>5s} {'test°mean':>9s} "
        f"{'test°max':>8s}  {'#≤5°':>5s} of eval_all"
    )
    for ds, name, n, jobs in scene_jobs(args):
        scene = load_scene(args.data_root, ds, name)
        for protocol, k in jobs:
            rec = make_split(scene, protocol, n, k)
            path = split_path(args.out, ds, name, protocol, n, k)
            save_split(path, rec)
            t = np.array([rec["distances"][v]["centre_deg"] for v in rec["test"]])
            a = np.array([rec["distances"][v]["centre_deg"] for v in rec["eval_all"]])
            print(
                f"{ds:7s} {name:9s} {len(scene):4d}  {path.stem:10s} {len(t):5d} {t.mean():9.2f} "
                f"{t.max():8.2f}  {int((a < 5).sum()):5d}/{len(a)}"
            )


def make_inits(args):
    from splataudit.data import load_scene
    from splataudit.init import SEEDED_INITS, build_init
    from splataudit.splits import load_split, resolve, split_path

    for ds, name, n, jobs in scene_jobs(args):
        scene = load_scene(args.data_root, ds, name)
        for protocol, k in jobs:
            if protocol not in args.protocol:
                continue
            path = split_path(args.splits, ds, name, protocol, n, k)
            train_ids = resolve(scene, load_split(path))["train"]
            for init in args.init:
                for seed in args.seeds if init in SEEDED_INITS else [None]:
                    s = build_init(
                        init, scene, train_ids, path.stem, seed, args.cache, args.overwrite
                    )
                    tag = f" seed={seed}" if seed is not None else ""
                    print(
                        f"{ds}/{name}/{path.stem} {init}{tag}: {s['status']}, "
                        f"{s.get('n_points', 0)} points, coverage {s.get('coverage', float('nan')):.3f}"
                    )


def run(args):
    from splataudit.runner import load_config, run_cell

    cfg = load_config(args.config)
    for key in ("dataset", "scene", "init", "method", "protocol", "k", "seed"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    rows = run_cell(cfg)
    print(f"done, {sum(r['status'] == 'ok' for r in rows)}/{len(rows)} views evaluated ok")


def sweep(args):
    from pathlib import Path

    from splataudit.runner import estimate_hours, load_config, plan_sweep
    from splataudit.runner import sweep as run_sweep
    from splataudit.utils import finished_runs, run_key, run_tag

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.scenes:
        cfg["scenes"] = args.scenes
    for key in ("datasets", "inits", "methods", "seeds", "protocols"):
        if getattr(args, key):
            cfg["sweep"][key] = getattr(args, key)

    results = Path(cfg["output_dir"]) / "results.csv"
    cells = plan_sweep(cfg)
    done = finished_runs(results)
    todo = [c for c in cells if run_key(c) not in done]
    print(
        f"{len(cells)} cells, {len(todo)} still to run, "
        f"rough estimate {estimate_hours(todo, results):.1f} h"
    )
    if args.dry_run:
        for c in todo[:20]:
            print("  ", run_tag(c))
        if len(todo) > 20:
            print(f"   ... and {len(todo) - 20} more")
        return
    print(f"\nsweep done, {run_sweep(cfg, args.max_runs)} new rows in {results}")


def report(args):
    import pandas as pd

    from splataudit.runner import report as summarize

    summary = summarize(pd.read_csv(args.results))
    print(summary.to_latex(float_format="%.3f") if args.latex else summary.to_string())


if __name__ == "__main__":
    main()
