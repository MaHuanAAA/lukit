import numpy as np
from .base_method import UncertaintyMethod


class NumSet(UncertaintyMethod):
    name = "numset"
    requires_data = ["sampling_stats", "similarity_matrix_stats"]
    tags = ["sampling", "semantic"]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def _compute(self, stats):
        sim_stats = stats["similarity_matrix_stats"]
        w = np.asarray(sim_stats.get("W", []), dtype=np.float32)
        
        if w.ndim != 2 or w.shape[0] == 0:
            return {"u": 0.0}

        n = w.shape[0]
        if n == 1:
            return {"u": 1.0}

        semantic_set_ids = list(range(n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if min(w[i, j], w[j, i]) > self.threshold:
                    set_i = semantic_set_ids[i]
                    set_j = semantic_set_ids[j]
                    if set_i != set_j:
                        for k in range(n):
                            if semantic_set_ids[k] == set_j:
                                semantic_set_ids[k] = set_i

        num_sets = len(set(semantic_set_ids))
        return {"u": float(num_sets)}
