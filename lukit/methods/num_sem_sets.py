from .base_method import UncertaintyMethod
from .semantic_graph_utils import (
    build_num_sem_sets_adjacency,
    connected_components_count,
    get_sampled_texts,
)


class NumSemSets(UncertaintyMethod):
    name = "num_sem_sets"
    requires_data = ["sampling_stats"]
    tags = ["sampling", "semantic", "graph", "discrete"]
    requires_semantic_matrices = True

    def __init__(
        self,
        similarity_score: str = "nli",
        affinity: str = "disagreement_w",
        temperature: float = 3.0,
        strict_entailment: bool = False,
        jaccard_threshold: float = 0.5,
    ) -> None:
        self.similarity_score = str(similarity_score).lower()
        self.affinity = str(affinity).lower()
        self.temperature = float(temperature)
        self.strict_entailment = bool(strict_entailment)
        self.jaccard_threshold = float(jaccard_threshold)

    def _compute(self, stats):
        sampling_stats = stats["sampling_stats"]
        texts = get_sampled_texts(sampling_stats)
        if len(texts) == 0:
            return {"u": 0.0}

        adjacency = build_num_sem_sets_adjacency(
            sampling_stats,
            similarity_score=self.similarity_score,
            strict_entailment=self.strict_entailment,
            jaccard_threshold=self.jaccard_threshold,
        )
        score = float(connected_components_count(adjacency))
        return {"u": score}
