import numpy as np
import torch
import torch.nn.functional as F

PATCH_RANGE = {"llff": (5, 17), "dtu": (17, 53), "mip360": (17, 53)}


class Method:
    """Few-shot training strategy. Subclasses override only the hooks they use."""

    requires_depth = False
    requires_confidence = False
    implemented = True

    def __init__(self, cfg=None):
        if not self.implemented:
            raise NotImplementedError(f"method {type(self).__name__} is not implemented yet")
        self.cfg = cfg or {}

    def pre_render(self, act, step):
        """Activated gaussians to rasterise this step, the optimiser keeps the originals."""
        return act

    def extra_loss(self, render, batch, step):
        """{name: scalar tensor} added to the photometric loss."""
        return {}

    def densify_mask(self, grads, state, step):
        """(N,) bool mask of gaussians allowed to densify, or None for all of them."""
        return None

    def post_step(self, state, step):
        """Side effects after the optimiser step."""
        pass


class Vanilla(Method):
    """Plain 3dgs, no extra regularisation (Kerbl et al. 2023)."""

    pass


class DropGaussian(Method):
    """Dropout on whole gaussians, rate ramps to drop_rate over training (Park et al. CVPR 2025)."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.drop_rate = self.cfg.get("drop_rate", 0.2)
        self.iterations = self.cfg.get("iterations", 10000)

    def rate(self, step):
        """Drop rate at this step, 0 at the start and drop_rate at the last iteration."""
        return self.drop_rate * step / self.iterations

    def pre_render(self, act, step):
        rate = self.rate(step)
        if rate <= 0:
            return act
        op = act["opacities"]
        # dropout scales survivors by 1/(1-rate), as the release does, so opacity can pass 1
        keep = F.dropout(torch.ones_like(op), p=rate, training=True)
        return {**act, "opacities": op * keep}


class FSGS(Method):
    """Fsgs depth term: 1 - pearson(rendered depth, mono prior) (Zhu et al. ECCV 2024)."""

    requires_depth = True

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.weight = self.cfg.get("depth_weight", 0.05)
        self.late_weight = self.cfg.get("depth_weight_late", 0.001)
        self.late_from = self.cfg.get("depth_weight_late_from", 9500)

    def extra_loss(self, render, batch, step):
        prior = batch["depth"]
        assert prior is not None, "fsgs needs a cached depth prior for this view"
        d = render["depth"].reshape(-1)
        p = norm01(prior).reshape(-1)
        # fsgs keeps whichever prior form correlates better with the render: negated
        # disparity, or its reciprocal. their offset is 200 on midas' 0-1000 range
        direct = 1 - pearson(-p, d)
        inverse = 1 - pearson(1 / (p + 0.2), d)
        w = self.weight if step < self.late_from else self.late_weight
        return {"fsgs_depth": w * torch.minimum(direct, inverse)}


class DNGaussian(Method):
    """Hard and soft depth losses on patch-normalised depth, plus shape penalties (Li et al. CVPR 2024)."""

    requires_depth = True

    def __init__(self, cfg=None):
        super().__init__(cfg)
        c = self.cfg
        self.local_w = c.get("local_weight", 0.1)
        self.global_w = c.get("global_weight", 1.0)
        self.smooth_w = c.get("smooth_weight", 0.1)
        self.smooth_from = c.get("smooth_start", 3000)
        self.margin = c.get("error_tolerance", 0.2)
        self.hard_from = c.get("hard_depth_start", 0)
        self.soft_from = c.get("soft_depth_start", 1000)
        self.hard_opacity = c.get("hard_opacity", 0.95)
        self.shape_pena = c.get("shape_pena", 0.001)
        self.scale_pena = c.get("scale_pena", 0.001)
        self.opa_pena = c.get("opa_pena", 0.01)
        self.patch_range = tuple(c.get("patch_range") or PATCH_RANGE[c.get("dataset", "llff")])
        self.rng = np.random.default_rng([c.get("seed", 0), 0xD9])

    def extra_loss(self, render, batch, step):
        prior = batch["depth"]
        assert prior is not None, "dngaussian needs a cached depth prior for this view"
        act = batch["act"]
        # patch normalisation cancels any affine change, so negating is their 255 - disparity
        target = -norm01(prior)
        out = {"dn_reg": self.penalties(act)}
        if step > self.hard_from:
            out["dn_hard"] = self.depth_loss(batch["render"](self.hard(act))["depth"], target, step)
        if step > self.soft_from:
            out["dn_soft"] = self.depth_loss(batch["render"](self.soft(act))["depth"], target, step)
        return out

    def hard(self, act):
        """Geometry-only view: opacity pinned high, white colours, scales and rotations detached."""
        return {
            "means": act["means"],
            "quats": act["quats"].detach(),
            "scales": act["scales"].detach(),
            "opacities": torch.full_like(act["opacities"], self.hard_opacity),
            "colors": torch.ones_like(act["means"]),
        }

    def soft(self, act):
        """Opacity-only view: learned opacity, geometry detached, white colours."""
        return {
            "means": act["means"].detach(),
            "quats": act["quats"].detach(),
            "scales": act["scales"].detach(),
            "opacities": act["opacities"],
            "colors": torch.ones_like(act["means"]),
        }

    def depth_loss(self, depth, target, step):
        """Local plus global patch-normalised depth error, with smoothness later in training."""
        size = int(self.rng.integers(*self.patch_range))
        loss = self.local_w * patch_depth_loss(depth, target, size, self.margin)
        loss = loss + self.global_w * patch_depth_loss(
            depth, target, size, self.margin, use_global=True
        )
        if step > self.smooth_from:
            loss = loss + self.smooth_w * depth_smoothness(depth, target)
        return loss

    def penalties(self, act):
        """Dngaussian's shape, scale and opacity penalties on the current gaussians."""
        s, op = act["scales"], act["opacities"]
        shape = (s.max(dim=1).values / s.min(dim=1).values.clamp_min(1e-8)).mean()
        scale = (s.max(dim=1).values ** 2).mean()
        opa = 1 - sq_mean(op[op > 0.2]) + sq_mean(1 - op[op < 0.2])
        return self.shape_pena * shape + self.scale_pena * scale + self.opa_pena * opa


class CrossViewGating(Method):
    """Ours: confidence from cross-view depth agreement gates densification and depth weight."""

    requires_depth = True
    requires_confidence = True
    implemented = False

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.threshold = self.cfg.get("confidence_threshold", 0.5)


METHODS = {
    "vanilla": Vanilla,
    "fsgs": FSGS,
    "dngaussian": DNGaussian,
    "dropgaussian": DropGaussian,
    "cross_view_gating": CrossViewGating,
}


def build(name, cfg=None):
    """Method instance by name, raises NotImplementedError for the stubs."""
    return METHODS[name](cfg)


def pearson(a, b):
    """Pearson correlation of two flat tensors, 1.0 when b is an increasing affine map of a."""
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / ((a * a).mean().sqrt() * (b * b).mean().sqrt() + 1e-8)


def norm01(x):
    """Tensor rescaled to [0, 1] by its own min and max."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def sq_mean(x):
    """Mean of squares, 0 for an empty selection."""
    return (x**2).mean() if x.numel() else x.sum()


def patchify(x, size):
    """(H, W) -> (P, size * size) non-overlapping patches, right and bottom remainder dropped."""
    return F.unfold(x[None, None], kernel_size=size, stride=size)[0].T


def norm_patches(p, std, stab):
    """Patch rows centred, then divided by their own std or by the std given."""
    p = p - p.mean(dim=1, keepdim=True)
    scale = p.std(dim=1, keepdim=True) if std is None else std
    return p / (scale + stab)


def patch_depth_loss(pred, target, size, margin, use_global=False):
    """Dngaussian's patch-normalised margin l2 on (H, W) depth maps."""
    p, t = patchify(pred, size), patchify(target, size)
    p = norm_patches(p, pred.std().detach() if use_global else None, 1e-2 * pred.std().detach())
    t = norm_patches(t, target.std() if use_global else None, 1e-2 * target.std())
    return margin_l2(p, t, margin)


def margin_l2(pred, target, margin):
    """Squared error over the elements that are off by more than margin, 0 if none are."""
    err = pred - target
    mask = err.abs() > margin
    return (err[mask] ** 2).mean() if bool(mask.any()) else (err * 0).sum()


def depth_smoothness(depth, guide):
    """Edge-aware smoothness of (H, W) depth, weighted down where guide has edges."""
    dx = (depth[:, :-1] - depth[:, 1:]).abs()
    dy = (depth[:-1] - depth[1:]).abs()
    wx = torch.exp(-(guide[:, :-1] - guide[:, 1:]).abs())
    wy = torch.exp(-(guide[:-1] - guide[1:]).abs())
    return ((dx * wx).sum() + (dy * wy).sum()) / (wx.sum() + wy.sum())
