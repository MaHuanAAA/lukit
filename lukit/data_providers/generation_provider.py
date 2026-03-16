from typing import Any, Dict

from .base_provider import DataProvider


class GenerationArtifactsProvider(DataProvider):
    provides = "generation_artifacts"
    requires_from_model = ["model_inputs", "generated_ids", "answer_text", "prompt_ids", "completion_ids"]

    def __call__(self, model_outputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return {
            "model_inputs": model_outputs["model_inputs"],
            "generated_ids": model_outputs["generated_ids"],
            "prompt_ids": model_outputs["prompt_ids"],
            "completion_ids": model_outputs["completion_ids"],
            "answer_text": model_outputs["answer_text"],
        }


class PTrueProvider(DataProvider):
    provides = "p_true_prob"
    requires_from_model = ["question", "answer_text"]

    def __call__(self, model_outputs: Dict[str, Any], **kwargs) -> float:
        backend = kwargs["backend"]
        return float(
            backend.compute_p_true_prob(
                question=str(model_outputs.get("question", "")),
                answer_text=str(model_outputs.get("answer_text", "")),
                with_context=bool(kwargs.get("p_true_with_context", False)),
                extra_context=str(model_outputs.get("extra_context", "")),
            )
        )
