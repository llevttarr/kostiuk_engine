import numpy as np
import pytest

torch = pytest.importorskip("torch")

from splataudit import methods as M  # noqa: E402

H, W = 32, 48


def gaussians(n=64, seed=0):
    """Activated gaussians with grad, as train.py hands them to a method."""
    g = torch.Generator().manual_seed(seed)
    return {
        "means": torch.rand(n, 3, generator=g, requires_grad=True),
        "quats": torch.rand(n, 4, generator=g),
        "scales": torch.rand(n, 3, generator=g).clamp_min(0.01).requires_grad_(True),
        "opacities": torch.rand(n, generator=g).requires_grad_(True),
        "colors": torch.rand(n, 16, 3, generator=g),
    }


def fake_render(act, mode="RGB+D"):
    """Stand-in for gsplat: a depth map that depends on the gaussians, so gradients flow."""
    base = torch.linspace(0, 1, H * W).reshape(H, W)
    depth = base * act["means"][:, 2].mean() + act["opacities"].mean()
    return {"depth": depth, "rgb": torch.zeros(1, 3, H, W), "alpha": torch.zeros(H, W)}


def batch(act, prior=None):
    g = torch.Generator().manual_seed(1)
    return {
        "act": act,
        "render": fake_render,
        "gt_image": torch.rand(1, 3, H, W, generator=g),
        "depth": torch.rand(H, W, generator=g) if prior is None else prior,
        "confidence": None,
    }


def test_pearson_is_one_for_increasing_affine():
    x = torch.rand(500)
    assert M.pearson(x, 2 * x + 3) == pytest.approx(1.0, abs=1e-4)
    assert M.pearson(x, -x) == pytest.approx(-1.0, abs=1e-4)


def test_norm01_spans_the_unit_interval():
    y = M.norm01(torch.tensor([-2.0, 0.0, 6.0]))
    assert y.min() == pytest.approx(0.0) and y.max() == pytest.approx(1.0)


def test_patchify_drops_the_remainder():
    p = M.patchify(torch.zeros(10, 10), 4)
    assert p.shape == (4, 16)


def test_margin_l2_ignores_errors_inside_the_margin():
    a = torch.zeros(10)
    assert M.margin_l2(a, a + 0.1, 0.2) == 0.0
    assert M.margin_l2(a, a + 0.5, 0.2) == pytest.approx(0.25)


def test_patch_depth_loss_is_invariant_to_rescaled_depth():
    g = torch.Generator().manual_seed(3)
    pred = torch.rand(H, W, generator=g)
    target = torch.rand(H, W, generator=g)
    for use_global in (False, True):
        a = M.patch_depth_loss(pred, target, 8, 0.1, use_global)
        b = M.patch_depth_loss(3 * pred + 5, target, 8, 0.1, use_global)
        assert a == pytest.approx(float(b), rel=1e-4)


def test_depth_smoothness_is_zero_on_flat_depth():
    flat = torch.ones(H, W)
    guide = torch.rand(H, W)
    assert M.depth_smoothness(flat, guide) == pytest.approx(0.0)


def test_vanilla_changes_nothing():
    act = gaussians()
    m = M.build("vanilla")
    assert m.pre_render(act, 0) is act
    assert m.extra_loss(fake_render(act), batch(act), 0) == {}
    assert m.densify_mask(torch.zeros(64, 2), act, 0) is None


def test_dropgaussian_rate_ramps_and_preserves_mean_opacity():
    m = M.build("dropgaussian", {"drop_rate": 0.2, "iterations": 10000})
    assert m.rate(0) == 0.0
    assert m.rate(10000) == pytest.approx(0.2)

    act = gaussians(n=20000)
    assert m.pre_render(dict(act), 0)["opacities"] is act["opacities"]
    out = m.pre_render(dict(act), 10000)["opacities"]
    dropped = (out == 0).float().mean()
    assert 0.15 < dropped < 0.25
    assert float(out.detach().mean()) == pytest.approx(
        float(act["opacities"].detach().mean()), rel=0.05
    )


def test_fsgs_depth_loss_vanishes_on_a_perfect_prior():
    act = gaussians()
    render = fake_render(act)
    prior = -render["depth"].detach()  # disparity is high where depth is low
    m = M.build("fsgs", {"depth_weight": 1.0})
    loss = m.extra_loss(render, batch(act, prior), 0)["fsgs_depth"]
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-3)


def test_fsgs_weight_drops_late_in_training():
    act = gaussians()
    render = fake_render(act)
    m = M.build(
        "fsgs", {"depth_weight": 0.05, "depth_weight_late": 0.001, "depth_weight_late_from": 5000}
    )
    b = batch(act)
    early = float(m.extra_loss(render, b, 100)["fsgs_depth"].detach())
    late = float(m.extra_loss(render, b, 5000)["fsgs_depth"].detach())
    assert early == pytest.approx(late * 50, rel=1e-3)


def test_fsgs_requires_a_prior():
    act = gaussians()
    b = batch(act)
    b["depth"] = None
    with pytest.raises(AssertionError):
        M.build("fsgs").extra_loss(fake_render(act), b, 0)


def test_dngaussian_terms_switch_on_at_their_gates():
    act = gaussians()
    m = M.build("dngaussian", {"dataset": "llff", "soft_depth_start": 1000})
    b = batch(act)
    assert set(m.extra_loss(fake_render(act), b, 1)) == {"dn_reg", "dn_hard"}
    assert set(m.extra_loss(fake_render(act), b, 2000)) == {"dn_reg", "dn_hard", "dn_soft"}


def test_dngaussian_gradients_reach_means_and_opacities():
    act = gaussians()
    m = M.build("dngaussian", {"dataset": "llff"})
    loss = sum(m.extra_loss(fake_render(act), batch(act), 2000).values())
    loss.backward()
    assert act["means"].grad is not None and act["means"].grad.abs().sum() > 0
    assert act["opacities"].grad is not None and act["opacities"].grad.abs().sum() > 0


def test_dngaussian_patch_range_follows_the_dataset():
    assert M.build("dngaussian", {"dataset": "llff"}).patch_range == (5, 17)
    assert M.build("dngaussian", {"dataset": "dtu"}).patch_range == (17, 53)


def test_dngaussian_penalties_are_zero_for_round_opaque_gaussians():
    n = 32
    act = {
        "means": torch.zeros(n, 3),
        "quats": torch.zeros(n, 4),
        "scales": torch.full((n, 3), 1e-6),
        "opacities": torch.ones(n),
        "colors": torch.zeros(n, 16, 3),
    }
    m = M.build("dngaussian", {"dataset": "llff"})
    # shape 1.0 and scale 0 leave only the shape penalty weight
    assert float(m.penalties(act)) == pytest.approx(0.001, abs=1e-6)


def test_every_method_in_the_sweep_is_implemented():
    for name in ("vanilla", "fsgs", "dngaussian", "dropgaussian"):
        assert M.build(name, {"dataset": "llff"}).implemented


def test_cross_view_gating_still_raises():
    with pytest.raises(NotImplementedError):
        M.build("cross_view_gating")


def test_depth_methods_declare_their_prior_needs():
    assert M.build("fsgs").requires_depth and not M.build("fsgs").requires_confidence
    assert M.build("dngaussian", {"dataset": "llff"}).requires_depth
    assert not M.build("vanilla").requires_depth
    assert not M.build("dropgaussian").requires_depth


def test_seeded_patch_sizes_are_reproducible():
    a = M.build("dngaussian", {"dataset": "llff", "seed": 7})
    b = M.build("dngaussian", {"dataset": "llff", "seed": 7})
    sizes_a = [int(a.rng.integers(*a.patch_range)) for _ in range(5)]
    sizes_b = [int(b.rng.integers(*b.patch_range)) for _ in range(5)]
    assert sizes_a == sizes_b
    assert all(5 <= s < 17 for s in sizes_a)


def test_patch_loss_handles_a_patch_larger_than_the_image():
    with pytest.raises(RuntimeError):
        M.patch_depth_loss(torch.rand(4, 4), torch.rand(4, 4), 8, 0.2)


def test_norm_patches_matches_a_hand_computation():
    p = torch.tensor([[1.0, 2.0, 3.0]])
    out = M.norm_patches(p, None, 0.0)
    expected = (p - 2.0) / p.std(dim=1, keepdim=True)
    assert torch.allclose(out, expected)
    assert np.isclose(float(out.detach().mean()), 0.0)
