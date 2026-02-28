import numpy as np

from .base_method import UncertaintyMethod


class Perplexity(UncertaintyMethod):
    name = "perplexity"
    requires_data = ["logprob_stats"]
    tags = ["intrinsic"]

    def _compute(self, stats):
        log_probs = np.asarray(stats["logprob_stats"]["completion_log_probs"], dtype=np.float64)
        if log_probs.size == 0:
            return {"u": 0.0}
        return {"u": float(np.exp(-log_probs.mean()))}
