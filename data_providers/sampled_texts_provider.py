from typing import Any, Dict

import torch

from .base_provider import DataProvider


class SamplingStatsProvider(DataProvider):
    provides = "sampling_stats"
    requires_from_model = ["model_inputs"]

    def __call__(self, model_outputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        backend = kwargs["backend"]
        try:
            return backend.collect_sampling_stats(
                model_inputs=model_outputs["model_inputs"],
                max_new_tokens=int(kwargs.get("max_new_tokens", 64)),
                num_samples=int(kwargs.get("num_samples", 4)),
                sample_temperature=float(kwargs.get("sample_temperature", 0.8)),
                sample_top_p=float(kwargs.get("sample_top_p", 0.9)),
                need_eigenscore_embeddings=bool(kwargs.get("need_eigenscore_embeddings", False)),
            )
        except torch.OutOfMemoryError as exc:
            return {
                "sampled_texts": [],
                "sampled_sequence_nlls": [],
                "eigenscore_embeddings": [],
                "__warning__": f"Sampling skipped due to CUDA OOM: {exc}",
            }
