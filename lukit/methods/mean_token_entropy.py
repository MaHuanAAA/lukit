import numpy as np

from .base_method import UncertaintyMethod


class MeanTokenEntropy(UncertaintyMethod):
    name = "mean_token_entropy"
    requires_data = ["logprob_stats"]
    tags = ["intrinsic"]

    def _compute(self, stats):
        entropies = np.asarray(stats["logprob_stats"]["token_entropies"], dtype=np.float64)
        if entropies.size == 0:
            return {"u": 0.0}
        return {"u": float(entropies.mean())}
