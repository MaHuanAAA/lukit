from typing import Any, Dict

from .base_provider import DataProvider


class PromptTextProvider(DataProvider):
    provides = "prompt_text"
    requires_from_model = ["prompt_text"]

    def __call__(self, model_outputs: Dict[str, Any], **kwargs) -> str:
        return str(model_outputs.get("prompt_text", ""))
