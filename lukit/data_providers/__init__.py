"""Data provider registry."""

import importlib
import inspect
from typing import Dict, Type

from .base_provider import DataProvider

_PROVIDER_MODULES = [
    "prompt_provider",
    "generation_provider",
    "log_probs_provider",
    "sampled_texts_provider",
    "hidden_states_provider",
]


def _discover_providers() -> Dict[str, Type[DataProvider]]:
    registry: Dict[str, Type[DataProvider]] = {}
    for module_name in _PROVIDER_MODULES:
        module = importlib.import_module(f"{__name__}.{module_name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, DataProvider) or cls is DataProvider:
                continue
            if cls.__module__ != module.__name__:
                continue
            if not cls.provides:
                continue
            registry[cls.provides] = cls
    return registry


PROVIDER_REGISTRY = _discover_providers()

__all__ = ["DataProvider", "PROVIDER_REGISTRY"]
