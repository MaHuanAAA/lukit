from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseBackend(ABC):
    """Backend abstraction for model inference and statistic extraction."""

    name: str = "base"

    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def apply_chat_template(self, question: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def format_chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def compute_logprob_stats(self, prompt_ids: Any, completion_ids: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def collect_sampling_stats(
        self,
        model_inputs: Dict[str, Any],
        max_new_tokens: int,
        num_samples: int,
        sample_temperature: float,
        sample_top_p: float,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def compute_p_true_prob(
        self,
        question: str,
        answer_text: str,
        with_context: bool = False,
        extra_context: str = "",
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        raise NotImplementedError
