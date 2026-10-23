#!/usr/bin/env python3
"""Render Ground Truth vs Rendered showcase images for the midterm presentation.

Supports one-command execution on a local GPU (e.g. GTX 1660 Super 6 GB VRAM):
- Automatically fetches only the single required DTU scan (~60 MB) via HTTP Range
  requests if `data/dtu/` is not present yet.
- Builds initializations (`sparse_sfm`, `random`, `foundation` / VGGT) with 6 GB VRAM
  safety (automatic CPU fallback if GPU VRAM is tight).
- Trains `vanilla` 3DGS (10,000 steps) and caches checkpoints in `cache/showcase/`
  so re-rendering additional camera views is instant (<1s).
- Saves both individual PNGs (`gt.png`, `rendered_*.png`) and ready-to-insert
  side-by-side comparison panels (`pair_*.png`) into `results/showcase/<preset>/`.

Usage examples:
  PYTHONPATH=src python scripts/render_showcase.py --list
  PYTHONPATH=src python scripts/render_showcase.py --preset dtu_scan30
  PYTHONPATH=src python scripts/render_showcase.py --preset dtu_scan8
  PYTHONPATH=src python scripts/render_showcase.py --preset dtu_scan103
  PYTHONPATH=src python scripts/render_showcase.py --preset llff_room
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


PRESETS: dict[str, dict] = {
    "dtu_scan30": {
        "dataset": "dtu",
        "scene": "scan30",
        "protocol": "p0",
        "seed": 0,
        "inits": ["sparse_sfm", "foundation"],
        "rescue_view": "rect_027_3_r5000.png",
        "extrap_fail_view": "rect_044_3_r5000.png",
        "description": (
            "DTU scan30 (білі коробки/інструменти, 400x300): "
            "Case 1 (rect_027, θ=11.6°): Sparse SfM+Vanilla падає до 7.49 dB (4.41 dB masked), "
            "а VGGT+Vanilla дає 19.76 dB (22.89 dB masked, SSIM 0.940) [+12.27 dB / +18.47 dB masked]. "
            "Case 2 (rect_044, θ=28.6° >20°): екстраполяційний Fail Case — навіть VGGT падає до 11.27 dB."
        ),
    },
    "dtu_scan8": {
        "dataset": "dtu",
        "scene": "scan8",
        "protocol": "p0",
        "seed": 0,
        "inits": ["sparse_sfm", "foundation"],
        "rescue_view": "rect_030_3_r5000.png",
        "extrap_fail_view": "rect_042_3_r5000.png",
        "description": (
            "DTU scan8 (гарбузи/черепи, 400x300): "
            "Case 1 (rect_030, θ=11.5°): Sparse SfM+Vanilla дає 5.15 dB (3.94 dB masked), "
            "а VGGT+Vanilla піднімає до 19.42 dB (22.55 dB masked) [+14.27 dB / +18.61 dB masked]. "
            "Case 2 (rect_042, θ=23.2° >20°): екстраполяційний Fail Case — навіть VGGT падає до 5.96 dB."
        ),
    },
    "dtu_scan103": {
        "dataset": "dtu",
        "scene": "scan103",
        "protocol": "p0",
        "seed": 0,
        "inits": ["sparse_sfm", "random", "foundation"],
        "rescue_view": "rect_025_3_r5000.png",
        "extrap_fail_view": "rect_003_3_r5000.png",
        "description": (
            "DTU scan103 (рожева скарбничка-свинка, 400x300): "
            "Case 1 (rect_025, θ=11.6°): Sparse SfM+Vanilla колапсує до 1 гаусіани (4.76 dB), "
            "Random+Vanilla дає хмару флоатерів (8.84 dB), а VGGT+Vanilla дає 15.99 dB (24.26 dB masked, SSIM 0.899) [+19.38 dB masked!]. "
            "Case 2 (rect_003, θ=34.2° >20°): екстраполяційний Fail Case (10.43 dB)."
        ),
    },
    "dtu_scan110": {
        "dataset": "dtu",
        "scene": "scan110",
        "protocol": "p0",
        "seed": 0,
        "inits": ["random", "foundation"],
        "rescue_view": "rect_025_3_r5000.png",
        "extrap_fail_view": "rect_044_3_r5000.png",
        "description": (
            "DTU scan110 (золотий кролик, глянцева поверхня без текстур, 400x300): "
            "Sparse SfM повністю падає з 0 триангульованими точками! "
            "Random+Vanilla дає 16.04 dB (SSIM 0.347), а VGGT+Vanilla видає 24.38 dB (SSIM 0.767 / 0.911 masked)."
        ),
    },
    "llff_room": {
        "dataset": "llff",
        "scene": "room",
        "protocol": "p0",
        "seed": 0,
        "inits": ["sparse_sfm", "foundation"],
        "rescue_view": "DJI_20200226_143948_113.JPG",
        "extrap_fail_view": "DJI_20200226_143912_008.JPG",
        "description": (
            "LLFF room (реальна кімната, 504x378): "
            "Case 1 (DJI_..._113.JPG, θ=4.2°): Sparse SfM+Vanilla дає 13.34 dB (SSIM 0.496), "
            "а VGGT+Vanilla дає 21.39 dB (SSIM 0.832) [+8.06 dB]. "
            "Case 2 (DJI_..._008.JPG, θ=9.1°): падіння якості на краю траєкторії (16.81 dB VGGT / 12.62 dB SfM)."
        ),
    },
}


def ensure_data(dataset: str, scene: str, data_root: Path) -> None:
    """Download only the single required DTU scan if missing; check LLFF path."""
    if dataset == "dtu":
        import get_dtu

        get_dtu.OUT = data_root / "dtu"
        calib_dir = data_root / "dtu" / "Calibration" / "cal18"
        if not (calib_dir / "pos_001.txt").exists():
            print("[data] Завантаження калібрувальних матриць DTU (~50 KB)...", flush=True)
            get_dtu.get_calib()
        scan_dir = data_root / "dtu" / "Rectified" / scene
        if not scan_dir.exists() or len(list(scan_dir.glob("*.png"))) < 49:
            scan_num = int(scene.replace("scan", ""))
            print(
                f"[data] Завантаження лише одного скану DTU {scene} (light 3, ~60 MB)...",
                flush=True,
            )
            get_dtu.get([scan_num], light="3", points=False)
    elif dataset == "llff":
        scene_dir = data_root / "llff" / scene
        if not scene_dir.exists():
            raise FileNotFoundError(
                f"Сцену {scene_dir} не знайдено. Запустіть спершу: bash scripts/get_llff.sh"
            )


def make_side_by_side(
    gt_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    title_left: str,
    title_right: str,
    subtitle: str,
) -> np.ndarray:
    """Create a clean dark-themed side-by-side comparison image (RGB uint8)."""
    import cv2

    h, w, _ = gt_rgb.shape
    pad = 16
    header_h = 68
    canvas_w = w * 2 + pad * 3
    canvas_h = h + header_h + pad * 2

    canvas = np.full((canvas_h, canvas_w, 3), (15, 23, 42), dtype=np.uint8)  # #0f172a
    y0 = header_h + pad
    x_left = pad
    x_right = pad * 2 + w

    canvas[y0 : y0 + h, x_left : x_left + w] = gt_rgb
    canvas[y0 : y0 + h, x_right : x_right + w] = pred_rgb

    # Subtle 1px borders around images
    cv2.rectangle(canvas, (x_left - 1, y0 - 1), (x_left + w, y0 + h), (71, 85, 105), 1)
    cv2.rectangle(canvas, (x_right - 1, y0 - 1), (x_right + w, y0 + h), (71, 85, 105), 1)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(canvas, subtitle, (pad, 26), font, 0.52, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.putText(canvas, title_left, (x_left, 56), font, 0.58, (248, 250, 252), 1, cv2.LINE_AA)
    cv2.putText(canvas, title_right, (x_right, 56), font, 0.58, (56, 189, 248), 1, cv2.LINE_AA)
    return canvas


def make_triple_panel(
    gt_rgb: np.ndarray,
    fail_rgb: np.ndarray,
    win_rgb: np.ndarray,
    title_gt: str,
    title_fail: str,
    title_win: str,
    subtitle: str,
) -> np.ndarray:
    """Create a 3-column panel: Ground Truth | Fail (SfM/Random) | Win (VGGT)."""
    import cv2

    h, w, _ = gt_rgb.shape
    pad = 16
    header_h = 68
    canvas_w = w * 3 + pad * 4
    canvas_h = h + header_h + pad * 2

    canvas = np.full((canvas_h, canvas_w, 3), (15, 23, 42), dtype=np.uint8)
    y0 = header_h + pad
    xs = [pad, pad * 2 + w, pad * 3 + w * 2]
    imgs = [gt_rgb, fail_rgb, win_rgb]
    titles = [title_gt, title_fail, title_win]
    colors = [(248, 250, 252), (248, 113, 113), (52, 211, 153)]

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(canvas, subtitle, (pad, 26), font, 0.52, (148, 163, 184), 1, cv2.LINE_AA)
    for x, im, title, col in zip(xs, imgs, titles, colors, strict=True):
        canvas[y0 : y0 + h, x : x + w] = im
        cv2.rectangle(canvas, (x - 1, y0 - 1), (x + w, y0 + h), (71, 85, 105), 1)
        cv2.putText(canvas, title, (x, 56), font, 0.54, col, 1, cv2.LINE_AA)
    return canvas


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def render_view(state: dict, cam, device) -> tuple[np.ndarray, float, float]:
    """Render a single camera view from trained state and compute PSNR & SSIM."""
    import torch
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )

    from splataudit.eval import _render, _to_tensor

    params = {k: v.to(device) for k, v in state.items() if isinstance(v, torch.Tensor)}
    sh_degree = int(state["sh_degree"])
    render, _ = _render(params, sh_degree, cam, device, mode="RGB")
    pred = render[..., :3].permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
    gt = _to_tensor(cam.load_image(), device)

    psnr = float(peak_signal_noise_ratio(pred, gt, data_range=1.0))
    ssim = float(structural_similarity_index_measure(pred, gt, data_range=1.0))
    pred_uint8 = (
        (pred[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0)
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    )
    return pred_uint8, psnr, ssim


def run_showcase(
    preset_name: str, iterations: int | None = None, force_train: bool = False
) -> None:
    import torch

    from splataudit.data import load_scene
    from splataudit.init import get_init
    from splataudit.runner import build_method, load_config, with_defaults
    from splataudit.splits import load_split, resolve, split_path
    from splataudit.train import train
    from splataudit.utils import seed_everything

    preset = PRESETS[preset_name]
    cfg = load_config(ROOT / "config.yaml")
    cfg.update(
        dataset=preset["dataset"],
        scene=preset["scene"],
        protocol=preset["protocol"],
        method="vanilla",
        seed=preset["seed"],
    )
    cfg = with_defaults(cfg)
    if iterations is not None:
        cfg["train"]["iterations"] = iterations

    data_root = ROOT / cfg["data_root"]
    ensure_data(cfg["dataset"], cfg["scene"], data_root)

    scene = load_scene(data_root, cfg["dataset"], cfg["scene"])
    sp = split_path(
        ROOT / cfg["splits_root"],
        cfg["dataset"],
        cfg["scene"],
        cfg["protocol"],
        cfg["n_views"],
        cfg.get("k"),
    )
    record = load_split(sp)
    ids = resolve(scene, record)

    out_dir = ROOT / "results" / "showcase" / preset_name
    ckpt_dir = ROOT / "cache" / "showcase_ckpts" / preset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Запуск пресету '{preset_name}' на пристрої: {device} ===")
    print(f"Опис: {preset['description']}\n")

    states: dict[str, dict] = {}
    for init_name in preset["inits"]:
        ckpt_path = ckpt_dir / f"{init_name}__vanilla__iters={cfg['train']['iterations']}.pt"
        if ckpt_path.exists() and not force_train:
            print(f"[{init_name}] Завантаження збереженого чекпойнта: {ckpt_path}")
            states[init_name] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            continue

        t0 = time.time()
        print(f"[{init_name}] Крок 1/2: Ініціалізація хмари точок...")
        seed_everything(int(cfg["seed"]))
        xyz, rgb = get_init(
            init_name, scene, ids["train"], sp.stem, int(cfg["seed"]), ROOT / cfg["cache_root"]
        )
        print(f"[{init_name}] Отримано {len(xyz):,} точок за {time.time() - t0:.1f} с.")

        t1 = time.time()
        print(
            f"[{init_name}] Крок 2/2: Тренування vanilla 3DGS ({cfg['train']['iterations']} ітерацій)..."
        )
        method = build_method(cfg)
        state = train(cfg, scene, ids["train"], xyz, rgb, method)
        torch.save(state, ckpt_path)
        states[init_name] = state
        print(
            f"[{init_name}] Готово за {time.time() - t1:.1f} с "
            f"({state['n_gaussians']:,} гаусіан) -> {ckpt_path}\n"
        )

    # 1. Render Case 1: VGGT rescues Vanilla (and contrast fail case of Sparse SfM / Random on the same view)
    rescue_name = preset["rescue_view"]
    cam_rescue = scene.by_name(rescue_name)
    gt_rescue = cam_rescue.load_image()
    theta_rescue = record["distances"][rescue_name]["centre_deg"]

    save_rgb(out_dir / f"case1_{rescue_name}_GT.png", gt_rescue)

    rendered_rescue: dict[str, tuple[np.ndarray, float, float]] = {}
    for init_name, state in states.items():
        pred_img, psnr, ssim = render_view(state, cam_rescue, device)
        rendered_rescue[init_name] = (pred_img, psnr, ssim)
        save_rgb(out_dir / f"case1_{rescue_name}_{init_name}_vanilla.png", pred_img)

        pair = make_side_by_side(
            gt_rescue,
            pred_img,
            "Ground Truth",
            f"{init_name} + vanilla ({psnr:.2f} dB, SSIM {ssim:.3f})",
            f"{cfg['dataset']}/{cfg['scene']} | view={rescue_name} | theta_min={theta_rescue:.1f} deg",
        )
        save_rgb(out_dir / f"pair_case1_{init_name}_vs_GT.png", pair)

    baseline_init = "sparse_sfm" if "sparse_sfm" in rendered_rescue else preset["inits"][0]
    if "foundation" in rendered_rescue and baseline_init in rendered_rescue:
        fail_img, fail_psnr, _ = rendered_rescue[baseline_init]
        win_img, win_psnr, _ = rendered_rescue["foundation"]
        triple = make_triple_panel(
            gt_rescue,
            fail_img,
            win_img,
            "Ground Truth",
            f"FAIL: {baseline_init}+vanilla ({fail_psnr:.2f} dB)",
            f"WIN: VGGT+vanilla ({win_psnr:.2f} dB, +{win_psnr - fail_psnr:.1f} dB)",
            f"RQ1 Showcase | {cfg['dataset']}/{cfg['scene']} ({rescue_name}, theta_min={theta_rescue:.1f} deg)",
        )
        save_rgb(out_dir / "slide_RQ1_gt_vs_fail_vs_vggt.png", triple)

    # 2. Render Case 2: High-angle extrapolation Fail Case (theta > 20 deg)
    fail_name = preset["extrap_fail_view"]
    cam_fail = scene.by_name(fail_name)
    gt_fail = cam_fail.load_image()
    theta_fail = record["distances"][fail_name]["centre_deg"]

    save_rgb(out_dir / f"case2_extrap_{fail_name}_GT.png", gt_fail)
    for init_name, state in states.items():
        pred_img, psnr, ssim = render_view(state, cam_fail, device)
        save_rgb(out_dir / f"case2_extrap_{fail_name}_{init_name}_vanilla.png", pred_img)
        pair = make_side_by_side(
            gt_fail,
            pred_img,
            "Ground Truth",
            f"FAIL (>20 deg): {init_name}+vanilla ({psnr:.2f} dB, SSIM {ssim:.3f})",
            f"RQ2 Extrapolation Fail | {cfg['dataset']}/{cfg['scene']} | view={fail_name} | theta_min={theta_fail:.1f} deg",
        )
        save_rgb(out_dir / f"pair_case2_extrap_{init_name}_vs_GT.png", pair)

    print(f"=== Усі зображення успішно збережено у {out_dir} ===")
    for p in sorted(out_dir.glob("*.png")):
        print("  ->", p.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генерація Ground Truth vs Rendered зображень для презентації"
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        default="dtu_scan30",
        help="Вибір сцени-представника з results_final.csv (за замовчуванням: dtu_scan30)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Вивести список доступних пресетів та їхні метрики з results_final.csv",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Кількість ітерацій тренування (за замовчуванням 10000 з config.yaml; можна 3000 для швидкого прев'ю)",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Перетренувати моделі, навіть якщо чекпойнт вже є в cache/showcase_ckpts/",
    )
    args = parser.parse_args()

    if args.list:
        print("Доступні пресети (обрані з results/results_final.csv):\n")
        for name, p in PRESETS.items():
            print(f"  * {name:12s} ({p['dataset']}/{p['scene']}, inits={p['inits']})")
            print(f"    {p['description']}\n")
        return

    run_showcase(args.preset, iterations=args.iterations, force_train=args.force_train)


if __name__ == "__main__":
    main()
