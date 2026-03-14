import numpy as np
from .base_method import UncertaintyMethod


class Degree(UncertaintyMethod):
    name = "degree"
    requires_data = ["sampling_stats", "similarity_matrix_stats"]
    tags = ["sampling", "semantic"]

    def _compute(self, stats):
        sim_stats = stats["similarity_matrix_stats"]
        w = np.asarray(sim_stats.get("W", []), dtype=np.float32)
        
        if w.ndim != 2 or w.shape[0] == 0:
            return {"u": 0.0}
            
        if w.shape[0] == 1:
            return {"u": 0.0}


        disagreement = 1.0 - w
        degrees = np.sum(disagreement, axis=1)
        
        u = float(np.mean(degrees))
        return {"u": u}
