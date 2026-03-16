import numpy as np

from .base_method import UncertaintyMethod
from .semantic_graph_utils import build_affinity_matrix, get_sampled_texts


class DegMat(UncertaintyMethod):
    name = "deg_mat"
    requires_data = ["sampling_stats"]
    tags = ["sampling", "semantic", "graph"]
    requires_semantic_matrices = True

    def __init__(
        self,
        similarity_score: str = "nli",
        affinity: str = "disagreement_w",
        temperature: float = 3.0,
        jaccard_threshold: float = 0.5,
    ) -> None:
        self.similarity_score = str(similarity_score).lower()
        self.affinity = str(affinity).lower()
        self.temperature = float(temperature)
        self.jaccard_threshold = float(jaccard_threshold)

    def _compute(self, stats):
        sampling_stats = stats["sampling_stats"]
        texts = get_sampled_texts(sampling_stats)
        n = len(texts)
        if n == 0:
            return {"u": 0.0, "c": []}

        similarity = build_affinity_matrix(
            sampling_stats,
            similarity_score=self.similarity_score,
            affinity=self.affinity,
            temperature=self.temperature,
            symmetric=False,
        )
        if similarity.shape != (n, n):
            return {"u": 0.0, "c": []}

        distances = 1.0 - similarity
        per_sample = np.sum(distances, axis=1, dtype=np.float64)
        return {
            "u": float(np.mean(per_sample)),
            "c": per_sample.astype(float).tolist(),
        }
