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
                question=str(model_outputs.get("question", "")),
                max_new_tokens=int(kwargs.get("max_new_tokens", 64)),
                num_samples=int(kwargs.get("num_samples", 4)),
                sample_temperature=float(kwargs.get("sample_temperature", 0.8)),
                sample_top_p=float(kwargs.get("sample_top_p", 0.9)),
                need_eigenscore_embeddings=bool(kwargs.get("need_eigenscore_embeddings", False)),
                need_semantic_matrices=bool(kwargs.get("need_semantic_matrices", False)),
                need_semantic_classes=bool(kwargs.get("need_semantic_classes", False)),
                semantic_similarity_score=str(kwargs.get("semantic_similarity_score", "nli")),
                semantic_affinity=str(kwargs.get("semantic_affinity", "disagreement_w")),
                semantic_temperature=float(kwargs.get("semantic_temperature", 3.0)),
                semantic_class_source=str(kwargs.get("semantic_class_source", "nli")),
                nli_model_path=str(kwargs.get("nli_model_path", "")),
                nli_device=str(kwargs.get("nli_device", "auto")),
                nli_torch_dtype=str(kwargs.get("nli_torch_dtype", "auto")),
                equivalence_judger_model_path=str(kwargs.get("equivalence_judger_model_path", "")),
                equivalence_judger_device=str(kwargs.get("equivalence_judger_device", "auto")),
                equivalence_judger_torch_dtype=str(kwargs.get("equivalence_judger_torch_dtype", "auto")),
                equivalence_judger_max_new_tokens=int(kwargs.get("equivalence_judger_max_new_tokens", 16)),
            )
        except torch.OutOfMemoryError as exc:
            return {
                "sampled_texts": [],
                "sampled_sequence_nlls": [],
                "sampled_sequence_logprobs": [],
                "eigenscore_embeddings": [],
                "similarity_matrix": [],
                "semantic_nli_logits": [],
                "semantic_matrix_entail": [],
                "semantic_matrix_contra": [],
                "semantic_classes": {"sample_to_class": [], "class_to_sample": []},
                "__warning__": f"Sampling skipped due to CUDA OOM: {exc}",
            }
