from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pandas as pd


# ─── Experiment grid ──────────────────────────────────────────────────────────

#: All valid values for each cube axis.
AXES: dict[str, list[str]] = {
    "init":     ["random", "colmap", "dust3r"],
    "method":   ["vanilla", "fsgs", "dngaussian", "dropgaussian", "cross_view_gating"],
    "protocol": ["interp", "sector"],
    "seed":     ["0", "1", "2"],           # kept as str for uniform grid handling
}


# ─── Public API ───────────────────────────────────────────────────────────────


def run_cell(cfg: dict) -> list[dict]:
    """
    Run a single experiment cell: one (scene, init, method, protocol, seed).

    Parameters
    ----------
    cfg : dict
        Must contain at minimum:
        ``scene``, ``dataset``, ``init``, ``method``, ``protocol``, ``seed``,
        ``n_views``, ``output_dir``.

    Returns
    -------
    list of per-image result dicts (see ``eval.evaluate``).
    """
    import yaml
    from splataudit.data import load_scene
    from splataudit.splits import make_splits, save_split
    from splataudit.init import get_init
    from splataudit.train import train
    from splataudit.eval import evaluate

    scene_root = Path(cfg["data_root"]) / cfg["dataset"] / cfg["scene"]
    scene = load_scene(scene_root)

    split = make_splits(
        scene,
        n_views=cfg["n_views"],
        mode=cfg["protocol"],
        seed=int(cfg["seed"]),
        sector_frac=cfg.get("split", {}).get("sector_frac", 0.30),
        llff_holdout=cfg.get("split", {}).get("llff_holdout", 8),
    )

    out_dir = Path(cfg["output_dir"]) / cfg["scene"] / _cell_tag(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_split(out_dir / "split.json", split, meta={
        "scene": cfg["scene"], "dataset": cfg["dataset"],
        "mode": cfg["protocol"], "n_views": cfg["n_views"], "seed": cfg["seed"],
    })

    # Initialise point cloud (train views only)
    xyz, rgb = get_init(cfg["init"], scene, split["train"], int(cfg["seed"]))

    # Depth / confidence priors (only if the method needs them)
    method = _build_method(cfg)
    depths, confidence_maps = None, None
    if method.requires_depth:
        from splataudit.priors import depth_prior, confidence
        depths = depth_prior(scene, split["train"])
        confidence_maps = confidence(scene, split["train"], depths)

    # Train
    state = train(cfg, scene, split["train"], xyz, rgb, method, depths, confidence_maps)

    # Evaluate
    results = evaluate(cfg, scene, state, split)
    _save_results(results, out_dir / "results.json")

    return results


def sweep(cfg: dict, vary: list[str] | None = None) -> pd.DataFrame:
    """
    Run all combinations of the specified axes and collect results.

    Parameters
    ----------
    vary : list[str] | None
        Axes to sweep.  Defaults to ``["init", "method", "protocol"]``.
        Seeds are always added automatically (3 seeds per cell).

    Example
    -------
    ::

        df = sweep(cfg, vary=["init", "method", "protocol"])
        # runs 3 × 5 × 2 = 30 cells × scenes × 3 seeds

    Results are saved to ``cfg["output_dir"]/results.csv``.
    """
    if vary is None:
        vary = ["init", "method", "protocol"]

    sweep_axes = vary if "seed" in vary else [*vary, "seed"]
    grid = list(itertools.product(*[AXES[k] for k in sweep_axes]))

    scenes = cfg.get("scenes") or _all_scenes(cfg)
    rows: list[dict] = []

    for scene_name in scenes:
        for combo in grid:
            cell_cfg = {
                **cfg,
                "scene": scene_name,
                **dict(zip(sweep_axes, combo)),
                "seed": int(combo[sweep_axes.index("seed")]) if "seed" in sweep_axes else 0,
            }
            try:
                cell_rows = run_cell(cell_cfg)
                for row in cell_rows:
                    row.update({k: cell_cfg[k] for k in sweep_axes})
                    row["scene"] = scene_name
                rows.extend(cell_rows)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Cell {cell_cfg} failed: {exc}")

    df = pd.DataFrame(rows)
    out = Path(cfg.get("output_dir", "results"))
    out.mkdir(exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    return df


def report(
    df: pd.DataFrame,
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate results into a human-readable summary table.

    Uses **paired** statistics: groups by (init, method, protocol) and
    reports mean ± std across (scene × seed) pairs, preserving per-scene
    pairing so that scene difficulty is a nuisance variable that cancels out.

    Parameters
    ----------
    group_by : list[str] | None
        Grouping columns.  Defaults to ``["init", "method", "protocol"]``.

    Returns
    -------
    DataFrame with MultiIndex columns (metric, stat).
    """
    if group_by is None:
        group_by = ["init", "method", "protocol"]

    metric_cols = [c for c in ["psnr", "ssim", "lpips", "depth_mae", "depth_consistency"]
                   if c in df.columns]
    agg = {m: ["mean", "std"] for m in metric_cols}

    return df.groupby(group_by).agg(agg).round(3)


# ─── Private helpers ──────────────────────────────────────────────────────────


def _cell_tag(cfg: dict) -> str:
    """Compact string identifier for a single experiment cell."""
    return f"init={cfg['init']}__method={cfg['method']}__proto={cfg['protocol']}__seed={cfg['seed']}"


def _all_scenes(cfg: dict) -> list[str]:
    """Return all scene names for the configured dataset."""
    dataset_scenes = {
        "llff":   ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"],
        "dtu":    ["scan8", "scan21", "scan30", "scan31", "scan34", "scan38",
                   "scan40", "scan41", "scan45", "scan55", "scan63", "scan82",
                   "scan103", "scan110", "scan114"],
        "mip360": ["bicycle", "bonsai", "counter", "garden", "kitchen",
                   "room", "stump", "flowers", "treehill"],
    }
    dataset = cfg.get("dataset", "llff")
    return dataset_scenes.get(dataset, [])


def _build_method(cfg: dict):
    """Instantiate a Method object from config."""
    from splataudit import methods as M
    name = cfg["method"]
    if name == "vanilla":         return M.Vanilla()
    if name == "fsgs":            return M.FSGS()
    if name == "dngaussian":      return M.DNGaussian()
    if name == "dropgaussian":    return M.DropGaussian(drop_rate=cfg.get("drop_rate", 0.1))
    if name == "cross_view_gating":
        return M.CrossViewGating(confidence_threshold=cfg.get("confidence_threshold", 0.5))
    raise ValueError(f"Unknown method {name!r}")


def _save_results(results: list[dict], path: Path) -> None:
    import json
    path.write_text(json.dumps(results, indent=2))
