from abc import ABC, abstractmethod
from typing import Any, Dict, List


class DataProvider(ABC):
    """Base class for all declarative data providers."""

    provides: str = ""
    requires_from_model: List[str] = []

    @abstractmethod
    def __call__(self, model_outputs: Dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError
