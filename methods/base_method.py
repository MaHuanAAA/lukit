from abc import ABC, abstractmethod
from typing import Any, Dict, List


class UncertaintyMethod(ABC):
    """Base class for all uncertainty methods."""

    name: str = "base_method"
    requires_data: List[str] = []
    supported_backends: List[str] = ["huggingface"]
    tags: List[str] = []

    @abstractmethod
    def _compute(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def compute(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        result = self._compute(stats)
        if isinstance(result, (int, float)):
            result = {"u": float(result)}
        if not isinstance(result, dict):
            raise TypeError(f"{self.name} must return dict, got: {type(result)}")
        if "u" in result:
            result["u"] = float(result["u"])
        return result

    def __call__(
        self,
        model: Any,
        tokenizer: Any = None,
        prompt: str = "",
        backend_config: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        from ..engine.execution_engine import ExecutionEngine

        cfg = dict(backend_config or {})
        cfg.setdefault("type", "huggingface")
        engine = ExecutionEngine(
            model=model,
            tokenizer=tokenizer,
            backend_config=cfg,
        )
        record = engine.run_single(prompt=prompt, method=self, **kwargs)
        return record["u"][self.name]
