from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="splataudit",
        description="Few-shot 3DGS audit: initialisation × method × protocol.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── splataudit run ────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run a single experiment cell.")
    p_run.add_argument("--config",     required=True, help="Path to config YAML.")
    p_run.add_argument("--scene",      default=None,  help="Override scene name.")
    p_run.add_argument("--init",       default=None,  choices=["random", "colmap", "dust3r"])
    p_run.add_argument("--method",     default=None,
                       choices=["vanilla", "fsgs", "dngaussian", "dropgaussian", "cross_view_gating"])
    p_run.add_argument("--protocol",   default=None,  choices=["interp", "sector"])
    p_run.add_argument("--seed",       default=None,  type=int)
    p_run.add_argument("--output-dir", default="results")

    # ── splataudit sweep ──────────────────────────────────────────────────────
    p_sweep = sub.add_parser(
        "sweep",
        help="Run the full init × method × protocol grid.",
    )
    p_sweep.add_argument("--config", required=True, help="Path to config YAML.")
    p_sweep.add_argument(
        "--vary",
        nargs="+",
        default=["init", "method", "protocol"],
        choices=["init", "method", "protocol", "seed"],
        metavar="AXIS",
        help="Axes to sweep (default: init method protocol).",
    )
    p_sweep.add_argument("--output-dir", default="results")

    # ── splataudit report ─────────────────────────────────────────────────────
    p_report = sub.add_parser("report", help="Aggregate and print results.")
    p_report.add_argument("--results",  required=True, help="Path to results CSV.")
    p_report.add_argument(
        "--group-by",
        nargs="+",
        default=["init", "method", "protocol"],
        metavar="COL",
    )
    p_report.add_argument("--latex", action="store_true", help="Print as LaTeX table.")

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "sweep":
        _cmd_sweep(args)
    elif args.command == "report":
        _cmd_report(args)


# ─── Command implementations ──────────────────────────────────────────────────


def _cmd_run(args) -> None:
    import yaml
    from splataudit.runner import run_cell

    cfg = yaml.safe_load(Path(args.config).read_text())
    # CLI overrides
    for key in ("scene", "init", "method", "protocol", "seed"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    cfg["output_dir"] = args.output_dir

    rows = run_cell(cfg)
    if rows:
        metrics = {k: v for k, v in rows[0].items() if isinstance(v, float)}
        print("Result (first test view):", metrics)
    print(f"Done. {len(rows)} test views evaluated.")


def _cmd_sweep(args) -> None:
    import yaml
    from splataudit.runner import sweep

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["output_dir"] = args.output_dir

    df = sweep(cfg, vary=args.vary)
    print(f"\nSweep complete. {len(df)} rows saved to {args.output_dir}/results.csv")


def _cmd_report(args) -> None:
    import pandas as pd
    from splataudit.runner import report

    df = pd.read_csv(args.results)
    summary = report(df, group_by=args.group_by)

    if args.latex:
        print(summary.to_latex(float_format="%.3f"))
    else:
        print(summary.to_string())
