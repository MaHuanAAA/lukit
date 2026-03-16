from typing import Any, Dict

from .base_provider import DataProvider


class LogProbStatsProvider(DataProvider):
    provides = "logprob_stats"
    requires_from_model = ["prompt_ids", "completion_ids"]

    def __call__(self, model_outputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        backend = kwargs["backend"]
        return backend.compute_logprob_stats(
            prompt_ids=model_outputs["prompt_ids"],
            completion_ids=model_outputs["completion_ids"],
        )
