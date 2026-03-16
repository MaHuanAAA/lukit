import math

from .base_method import UncertaintyMethod


class PTrue(UncertaintyMethod):
    name = "p_true"
    requires_data = ["p_true_prob"]
    tags = ["interaction"]

    def __init__(self, with_context: bool = False) -> None:
        self.with_context = with_context

    def _compute(self, stats):
        p_true = float(stats["p_true_prob"])
        p_true = min(max(p_true, 1e-12), 1.0)
        u = -math.log(p_true)
        return {"u": u, "p_true": p_true}
