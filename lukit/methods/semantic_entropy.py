from .base_method import UncertaintyMethod
from .semantic_class_utils import semantic_entropy_from_logprobs


class SemanticEntropy(UncertaintyMethod):
    name = "semantic_entropy"
    requires_data = ["sampling_stats"]
    tags = ["sampling", "semantic", "entropy"]
    requires_semantic_classes = True

    def __init__(self, semantic_class_source: str = "nli") -> None:
        self.semantic_class_source = str(semantic_class_source).lower()

    def _compute(self, stats):
        sampling_stats = stats["sampling_stats"]
        score = semantic_entropy_from_logprobs(sampling_stats)
        semantic_classes = sampling_stats.get("semantic_classes") or {}
        return {
            "u": float(score),
            "n_classes": len(semantic_classes.get("class_to_sample") or []),
        }
