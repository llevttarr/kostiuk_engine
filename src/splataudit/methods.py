from __future__ import annotations
class Method:
    n_models = 1
    def pre_render(self, state, step):
        return state
    def extra_loss(self, render, batch, step) -> dict:
        return {}
    def densify_mask(self, grads, state, step):
        return None
    def post_step(self, state, step) -> None:
        pass


class Vanilla(Method):
    pass
class FSGS(Method):
    pass
class DNGaussian(Method):
    pass
class DropGaussian(Method):
    pass
class InstantSplat(Method):
    pass
