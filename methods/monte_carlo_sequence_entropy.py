import numpy as np

from .base_method import UncertaintyMethod


class MonteCarloSequenceEntropy(UncertaintyMethod):
    name = "monte_carlo_sequence_entropy"
    requires_data = ["sampling_stats"]
    tags = ["sampling"]

    def _compute(self, stats):
        nlls = np.asarray(stats["sampling_stats"]["sampled_sequence_nlls"], dtype=np.float64)
        if nlls.size == 0:
            return {"u": 0.0}
        return {"u": float(nlls.mean())}
