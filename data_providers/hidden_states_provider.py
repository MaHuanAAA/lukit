from typing import Any, Dict

from .base_provider import DataProvider


class HiddenStatesProvider(DataProvider):
    provides = "hidden_states"
    requires_from_model = ["generated_ids"]

    def __call__(self, model_outputs: Dict[str, Any], **kwargs) -> Any:
        return model_outputs.get("hidden_states")
