from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Method:
    """
    Base class for few-shot 3DGS training strategies.

    A Method hooks into four points of the training loop:

    ::

        for step in range(iterations):
            state  = method.pre_render(state, step)        # 1
            render = gsplat_rasterize(state, camera)
            losses = photometric_loss(render, gt_image)
            losses.update(method.extra_loss(render, batch, step))  # 2
            backward(sum(losses))
            optimizer.step()
            if should_densify(step):
                mask = method.densify_mask(grads, state, step)     # 3
                densify(state, mask)
            method.post_step(state, step)                          # 4

    Subclasses override only the hooks they need.
    """

    #: Set to True for methods that require monocular depth priors.
    #: When True, `priors.depth_prior()` is called before training and
    #: the depth maps are passed into `extra_loss` via `batch["depth"]`.
    requires_depth: bool = False

    #: False until the method's hooks are actually written.  The runner refuses
    #: to train an unimplemented method: running it would silently train vanilla
    #: 3DGS under another name, exactly the "methods don't matter" result RQ1
    #: is testing for.
    implemented: bool = True

    def __init__(self) -> None:
        if not self.implemented:
            raise NotImplementedError(f"method {type(self).__name__} is not implemented yet")

    def pre_render(self, state: dict, step: int) -> dict:
        """
        Optionally transform the Gaussian state before rasterisation.

        Use this for stochastic augmentations applied per-render, e.g.:
        - DropGaussian: randomly zero out a fraction of Gaussians.

        The returned state is used for this render only; the original
        state is preserved for the optimiser update.
        """
        return state

    def extra_loss(
        self,
        render: dict,
        batch: dict,
        step: int,
    ) -> dict[str, torch.Tensor]:
        """
        Return auxiliary scalar loss terms as ``{name: tensor}`` pairs.

        ``render`` contains gsplat outputs (``"rgb"``, ``"depth"``, ``"alpha"``).
        ``batch``  contains the current camera and optional ``"depths"`` map.

        The runner sums all returned tensors into the total loss, so each
        tensor must be a zero-dimensional scalar.
        """
        return {}

    def densify_mask(
        self,
        grads: torch.Tensor,
        state: dict,
        step: int,
    ) -> torch.Tensor | None:
        """
        Boolean mask ``(N,)`` gating which Gaussians may be densified.

        Return ``None`` to fall back to the default gradient-based selection
        (all Gaussians above the grad threshold are eligible).

        Use this to suppress densification in geometrically unreliable
        regions, e.g. CrossViewGating uses the confidence map here.
        """
        return None

    def post_step(self, state: dict, step: int) -> None:
        """
        Arbitrary side effects executed after each optimiser step.

        Examples: EMA updates, opacity resets, logging.
        """
        pass


class Vanilla(Method):
    """Standard 3DGS with no additional regularisation (Kerbl et al., 2023)."""

    pass


class FSGS(Method):
    """
    Few-Shot Gaussian Splatting (Zhu et al., arXiv 2312.00451).

    Adds a depth-alignment loss that anchors the rendered depth to
    monocular depth-prior estimates, scaled via per-image affine fitting.
    Also supports proximity-guided Gaussian initialisation (handled in init).
    """

    requires_depth = True
    implemented = False

    def extra_loss(self, render, batch, step):
        # TODO: affine-invariant depth alignment loss
        #   L_depth = || s * D_render + b - D_prior ||_1   (robust per-image fit)
        return {}


class DNGaussian(Method):
    """
    Global-Local Depth-Normalised Gaussian Splatting (Li et al., CVPR 2024).

    Uses a two-level depth normalisation:
    - Global: scene-level scale/shift alignment.
    - Local: patch-level ranking loss to preserve ordinal structure.
    """

    requires_depth = True
    implemented = False

    def extra_loss(self, render, batch, step):
        # TODO: global depth alignment + local ranking loss
        return {}


class DropGaussian(Method):
    """
    Structural Dropout Regularisation (Park et al., CVPR 2025).

    Randomly drops a fraction of Gaussians before each render during
    training (remaining opacities compensated by 1/(1-p)), acting as a
    structural regulariser that discourages over-reliance on any single
    Gaussian.  Not to be confused with CoR-GS (Zhang et al., ECCV 2024),
    which co-regularises two Gaussian fields.
    """

    implemented = False

    def __init__(self, drop_rate: float = 0.1):
        super().__init__()
        self.drop_rate = drop_rate

    def pre_render(self, state: dict, step: int) -> dict:
        # TODO: randomly zero opacity of `drop_rate` fraction of Gaussians
        return state


class CrossViewGating(Method):
    """
    Cross-View Consistency Gating for Densification (ours).

    Estimates geometric reliability across sparse training views by
    measuring depth agreement when projecting each view's depth map
    into the other views.  The resulting confidence signal is used to:

    1. Weight depth supervision: penalise unreliable regions less.
    2. Gate densification: suppress floater growth in low-confidence
       regions (via ``densify_mask``).

    The confidence maps are computed once before training in
    ``priors.confidence()`` and stored in ``batch["confidence"]``.
    """

    requires_depth = True
    implemented = False

    def __init__(self, confidence_threshold: float = 0.5):
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def densify_mask(
        self,
        grads: torch.Tensor,
        state: dict,
        step: int,
    ) -> torch.Tensor | None:
        # TODO: project each Gaussian onto the confidence maps;
        #   allow densification only where confidence > threshold.
        #   Return shape (N,), one flag per Gaussian.
        return None

    def extra_loss(self, render, batch, step):
        # TODO: confidence-weighted depth supervision
        return {}
