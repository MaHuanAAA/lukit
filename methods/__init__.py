"""Method registry with explicit module loading."""

import importlib
import inspect
from typing import Dict, List, Type

from .base_method import UncertaintyMethod

_METHOD_MODULES = [
    "sequence_log_probability",
    "perplexity",
    "mean_token_entropy",
    "self_certainty",
    "monte_carlo_sequence_entropy",
    "lexical_similarity",
    "p_true",
]


def _discover_methods() -> Dict[str, Type[UncertaintyMethod]]:
    registry: Dict[str, Type[UncertaintyMethod]] = {}
    for module_name in _METHOD_MODULES:
        module = importlib.import_module(f"{__name__}.{module_name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, UncertaintyMethod) or cls is UncertaintyMethod:
                continue
            if cls.__module__ != module.__name__:
                continue
            registry[cls.name] = cls
    return registry


METHOD_REGISTRY = _discover_methods()


def list_methods() -> List[str]:
    return sorted(METHOD_REGISTRY.keys())


def create_method(name: str, **kwargs) -> UncertaintyMethod:
    if name not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method: {name}")
    cls = METHOD_REGISTRY[name]
    if kwargs:
        return cls(**kwargs)
    return cls()


__all__ = ["UncertaintyMethod", "METHOD_REGISTRY", "list_methods", "create_method"]
