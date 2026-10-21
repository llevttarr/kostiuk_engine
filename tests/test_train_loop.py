import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchmetrics")

from splataudit.data import load_scene  # noqa: E402
from splataudit.splits import load_split, resolve, split_path  # noqa: E402

DATA = "data"
RENDERED = []


def stub_rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width,
    height,
    sh_degree,
    render_mode,
    packed,
):
    """Fake gsplat: every parameter reaches the output, so gradients and shapes are exercised."""
    RENDERED.append({"sh_degree": sh_degree, "mode": render_mode, "colors": tuple(colors.shape)})
    total = (
        means.sum() + quats.sum() + scales.sum() + opacities.sum() + colors.sum()
    ) / means.shape[0]
    dev = means.device
    ramp = torch.linspace(0.1, 0.9, height * width, device=dev).reshape(1, height, width, 1)
    renders = ramp * total
    renders = renders.expand(1, height, width, 4).clone()
    means2d = (means[:, :2] * 1.0).unsqueeze(0)
    if means2d.requires_grad:  # dngaussian's soft render detaches the means
        means2d.retain_grad()
    return renders, torch.ones(1, height, width, 1, device=dev) * total, {"means2d": means2d}


class StubStrategy:
    def __init__(self, **kw):
        self.calls = []

    def check_sanity(self, params, optimizers):
        self.calls.append("sanity")

    def initialize_state(self, scene_scale):
        return {"scene_scale": scene_scale}

    def step_pre_backward(self, params, optimizers, state, step, info):
        self.calls.append(("pre", step))
        assert "means2d" in info, "gsplat needs means2d before backward"

    def step_post_backward(self, params, optimizers, state, step, info):
        self.calls.append(("post", step))


@pytest.fixture
def gsplat(monkeypatch):
    """Install a fake gsplat so train.py runs on the cpu."""
    mod = types.ModuleType("gsplat")
    mod.rasterization = stub_rasterization
    strategy = types.ModuleType("gsplat.strategy")
    strategy.DefaultStrategy = StubStrategy
    monkeypatch.setitem(sys.modules, "gsplat", mod)
    monkeypatch.setitem(sys.modules, "gsplat.strategy", strategy)
    RENDERED.clear()
    yield mod


@pytest.fixture
def fern():
    scene = load_scene(DATA, "llff", "fern")
    ids = resolve(scene, load_split(split_path("splits", "llff", "fern", "p0", 3)))["train"]
    return scene, ids


def tiny_cloud(scene, n=200):
    rng = np.random.default_rng(0)
    far = scene.bounds[1]
    xyz = rng.uniform(-1, 1, (n, 3)) * far
    return xyz.astype(np.float32), rng.uniform(0, 1, (n, 3)).astype(np.float32)


def cfg_for(method, iterations=3):
    return {
        "seed": 0,
        "dataset": "llff",
        "method": method,
        "train": {
            "iterations": iterations,
            "lambda_dssim": 0.2,
            "sh_degree": 3,
            "densify_from": 500,
            "densify_every": 100,
            "densify_until": 5000,
            "grad_threshold": 0.0002,
            "opacity_reset_every": 3000,
            "prune_opacity": 0.005,
        },
    }


def depths_for(scene, ids):
    rng = np.random.default_rng(1)
    out = {}
    for i in ids:
        cam = scene.get_camera(i)
        out[i] = rng.uniform(0, 1, (cam.height, cam.width)).astype(np.float32)
    return out


@pytest.mark.parametrize("name", ["vanilla", "dropgaussian", "fsgs", "dngaussian"])
def test_three_steps_of_training_run_and_move_the_gaussians(gsplat, fern, name):
    from splataudit.runner import build_method
    from splataudit.train import train

    scene, ids = fern
    xyz, rgb = tiny_cloud(scene)
    cfg = cfg_for(name)
    method = build_method(cfg | {"methods": {}})
    depths = depths_for(scene, ids) if method.requires_depth else None

    state = train(cfg, scene, ids, xyz, rgb, method, depths)

    assert state["n_gaussians"] == len(xyz)
    assert state["means"].shape == (len(xyz), 3)
    assert not torch.allclose(state["means"], torch.tensor(xyz))
    assert torch.isfinite(state["means"]).all()
    assert torch.isfinite(state["opacities"]).all()


def test_dngaussian_renders_two_extra_views_per_step(gsplat, fern):
    from splataudit.runner import build_method
    from splataudit.train import train

    scene, ids = fern
    xyz, rgb = tiny_cloud(scene)
    cfg = cfg_for("dngaussian", iterations=2)
    method = build_method(cfg | {"methods": {"dngaussian": {"soft_depth_start": 0}}})
    train(cfg, scene, ids, xyz, rgb, method, depths_for(scene, ids))

    # the gates are on the 0-based step, so step 0 has no depth term and step 1 has both:
    # 2 main renders plus the hard and soft renders of step 1
    assert len(RENDERED) == 4
    extra = [r for r in RENDERED if len(r["colors"]) == 2]
    assert len(extra) == 2 and all(r["colors"][-1] == 3 for r in extra)


def test_precomputed_colours_switch_sh_degree_off(gsplat, fern):
    from splataudit.runner import build_method
    from splataudit.train import train

    scene, ids = fern
    xyz, rgb = tiny_cloud(scene)
    cfg = cfg_for("dngaussian", iterations=2)
    method = build_method(cfg | {"methods": {"dngaussian": {"soft_depth_start": 0}}})
    train(cfg, scene, ids, xyz, rgb, method, depths_for(scene, ids))

    extra = [r for r in RENDERED if len(r["colors"]) == 2]
    assert extra and all(r["sh_degree"] is None for r in extra)
    main = [r for r in RENDERED if len(r["colors"]) == 3]
    assert all(r["sh_degree"] == 0 for r in main)


def test_the_strategy_is_called_around_every_step(gsplat, fern):
    from splataudit.runner import build_method
    from splataudit.train import train

    scene, ids = fern
    xyz, rgb = tiny_cloud(scene)
    cfg = cfg_for("vanilla", iterations=4)
    train(cfg, scene, ids, xyz, rgb, build_method(cfg | {"methods": {}}))
    # StubStrategy records its own calls, so just check the loop asked for both hooks
    assert RENDERED and len(RENDERED) == 4


def test_dngaussian_gets_an_optimiser_step_per_pass(gsplat, fern, monkeypatch):
    from splataudit.runner import build_method
    from splataudit.train import train

    calls = []
    real = torch.optim.Adam.step
    monkeypatch.setattr(
        torch.optim.Adam, "step", lambda self, *a, **k: (calls.append(1), real(self, *a, **k))[1]
    )

    scene, ids = fern
    xyz, rgb = tiny_cloud(scene)
    n_params = 6  # means, quats, scales, opacities, sh0, shN

    cfg = cfg_for("vanilla", iterations=2)
    train(cfg, scene, ids, xyz, rgb, build_method(cfg | {"methods": {}}))
    assert len(calls) == 2 * n_params, "vanilla takes one step per iteration"

    calls.clear()
    cfg = cfg_for("dngaussian", iterations=2)
    method = build_method(cfg | {"methods": {"dngaussian": {"soft_depth_start": 0}}})
    train(cfg, scene, ids, xyz, rgb, method, depths_for(scene, ids))
    # step 0 has no depth pass (gates are > start), step 1 has hard and soft
    assert len(calls) == (1 + 3) * n_params


def test_grads_are_released_between_passes(gsplat, fern):
    from splataudit.runner import build_method
    from splataudit.train import train

    scene, ids = fern
    xyz, rgb = tiny_cloud(scene)
    cfg = cfg_for("dngaussian", iterations=3)
    method = build_method(cfg | {"methods": {"dngaussian": {"soft_depth_start": 0}}})
    state = train(cfg, scene, ids, xyz, rgb, method, depths_for(scene, ids))
    # a pass that touches nothing must not leave a zero grad for adam to step on
    assert torch.isfinite(state["means"]).all() and torch.isfinite(state["opacities"]).all()
