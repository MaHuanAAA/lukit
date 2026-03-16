import numpy as np

from .base_method import UncertaintyMethod
from .semantic_graph_utils import (
    build_affinity_matrix,
    get_sampled_texts,
    normalized_graph_laplacian,
)


class Eccentricity(UncertaintyMethod):
    name = "eccentricity"
    requires_data = ["sampling_stats"]
    tags = ["sampling", "semantic", "graph", "spectral"]
    requires_semantic_matrices = True

    def __init__(
        self,
        similarity_score: str = "nli",
        affinity: str = "disagreement_w",
        temperature: float = 3.0,
        thres: float = 0.9,
        jaccard_threshold: float = 0.5,
    ) -> None:
        self.similarity_score = str(similarity_score).lower()
        self.affinity = str(affinity).lower()
        self.temperature = float(temperature)
        self.thres = None if thres is None else float(thres)
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
            symmetric=True,
        )
        if similarity.shape != (n, n):
            return {"u": 0.0, "c": []}

        laplacian = normalized_graph_laplacian(similarity)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        if self.thres is not None:
            keep_mask = eigenvalues < self.thres
            kept_vectors = eigenvectors[:, keep_mask]
        else:
            kept_vectors = eigenvectors

        if kept_vectors.size == 0:
            return {"u": 0.0, "c": [0.0] * n}

        centroid = np.mean(kept_vectors, axis=0)
        distances = np.linalg.norm(kept_vectors - centroid, ord=2, axis=1)
        score = np.linalg.norm(distances, ord=2)
        return {
            "u": float(score),
            "c": distances.astype(float).tolist(),
        }
