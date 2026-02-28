import numpy as np

from .base_method import UncertaintyMethod


class SelfCertainty(UncertaintyMethod):
    name = "self_certainty"
    requires_data = ["logprob_stats"]
    tags = ["intrinsic"]

    def _compute(self, stats):
        logprob_stats = stats["logprob_stats"]
        mean_logp_vocab = np.asarray(logprob_stats["mean_logp_vocab"], dtype=np.float64)
        vocab_size = int(logprob_stats["vocab_size"])
        if mean_logp_vocab.size == 0 or vocab_size <= 0:
            return {"u": 0.0}

        kl_uniform_to_p = -mean_logp_vocab - np.log(vocab_size)
        score = -np.mean(kl_uniform_to_p)
        return {"u": float(score)}
