import numpy as np
from typing import Any, Dict

from .base_provider import DataProvider


def get_jaccard_matrix(texts):
    n = len(texts)
    ret = np.eye(n, dtype=np.float32)
    tokenized = [set(t.lower().split()) for t in texts]
    for i in range(n):
        for j in range(i + 1, n):
            set_i, set_j = tokenized[i], tokenized[j]
            inter = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))
            val = inter / max(union, 1)
            ret[i, j] = ret[j, i] = val
    return ret


class SimilarityMatrixProvider(DataProvider):
    """
    Builds the similarity matrix (W) using either Jaccard or NLI models.
    """
    provides = "similarity_matrix_stats"
    requires_from_model = ["model_inputs"]

    def __call__(self, model_outputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        backend = kwargs.get("backend")
        sampling_stats = kwargs.get("sampling_stats", {})
        texts = sampling_stats.get("sampled_texts", [])
        
        sim_metric = kwargs.get("similarity_metric", "jaccard").lower()
        if not texts:
            return {"W": np.zeros((0, 0), dtype=np.float32), "metric": sim_metric}

        n = len(texts)
        if n == 1:
            return {"W": np.ones((1, 1), dtype=np.float32), "metric": sim_metric}

        if sim_metric == "jaccard":
            W = get_jaccard_matrix(texts)
            return {"W": W, "metric": "jaccard"}
        elif sim_metric == "nli":
            nli_model_path = kwargs.get("nli_model_path", None)
            nli_affinity_mode = kwargs.get("nli_affinity_mode", "disagreement_w")
            nli_temperature = kwargs.get("nli_temperature", 3.0)
            nli_device = kwargs.get("nli_device", None)
            question = model_outputs.get("question", "")

            if not nli_model_path:
                raise ValueError(
                    "NLI model path is required when using expected_similarity_metric='nli'."
                )

            if backend and hasattr(backend, "compute_nli_affinity_matrix"):
                try:
                    W = backend.compute_nli_affinity_matrix(
                        question=question,
                        answers=texts,
                        nli_model_path=nli_model_path,
                        affinity_mode=nli_affinity_mode,
                        temperature=float(nli_temperature),
                        symmetric=True,
                        nli_device=nli_device
                    )
                    return {"W": W, "metric": "nli"}
                except Exception as exc:
                    return {
                        "W": np.zeros((n, n), dtype=np.float32),
                        "metric": "nli",
                        "__warning__": f"NLI matrix computation failed: {exc}"
                    }
            else:
                return {
                    "W": np.zeros((n, n), dtype=np.float32),
                    "metric": "nli",
                    "__warning__": "Backend does not support compute_nli_affinity_matrix"
                }
        else:
            return {
                "W": np.zeros((n, n), dtype=np.float32),
                "metric": sim_metric,
                "__warning__": f"Unknown similarity metric: {sim_metric}"
            }
