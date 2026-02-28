import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_backend import BaseBackend

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _as_sequences(generated: Any) -> torch.Tensor:
    return generated.sequences if hasattr(generated, "sequences") else generated


class HFBackend(BaseBackend):
    name = "huggingface"

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_path: str = "",
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        chat_template_config: str = "",
    ) -> None:
        self._device = _resolve_device(device)

        if model is None and not model_path and isinstance(tokenizer, str):
            model_path = tokenizer
            tokenizer = None

        if isinstance(model, str) and not model_path:
            model_path = model
            model = None

        if model is None:
            if not model_path:
                raise ValueError("model_path is required when model is not provided.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
            )
            dtype_arg: Any = torch_dtype
            if isinstance(torch_dtype, str) and torch_dtype.lower() in _DTYPE_MAP:
                dtype_arg = _DTYPE_MAP[torch_dtype.lower()]
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype_arg,
                trust_remote_code=trust_remote_code,
            )
            self.model.to(self._device)
        else:
            self.model = model
            self.tokenizer = tokenizer
            if self.tokenizer is None:
                raise ValueError("tokenizer is required when model object is provided.")
            self._device = str(getattr(self.model, "device", self._device))

        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.chat_template_config = chat_template_config
        self.chat_template_messages = None
        if self.chat_template_config:
            self.chat_template_messages = self._load_chat_template_config(self.chat_template_config)

    @property
    def device(self) -> str:
        return self._device

    def format_chat(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        lines = []
        for msg in messages:
            lines.append(f"{msg['role'].upper()}: {msg['content']}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    @staticmethod
    def _default_generation_messages(question: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant. Give a concise factual answer.",
            },
            {"role": "user", "content": question},
        ]

    @staticmethod
    def _load_chat_template_config(path: str) -> List[Dict[str, str]]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Invalid chat template config in {path}: 'messages' must be a non-empty list.")

        normalized: List[Dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"Invalid chat template item in {path}: each item must be an object.")
            role = str(message.get("role", "")).strip()
            content = message.get("content", "")
            if not role or not isinstance(content, str):
                raise ValueError(
                    f"Invalid chat template item in {path}: each item needs string 'role' and string 'content'."
                )
            normalized.append({"role": role, "content": content})
        return normalized

    def _build_generation_messages(self, question: str) -> List[Dict[str, str]]:
        if not self.chat_template_messages:
            return self._default_generation_messages(question)

        rendered: List[Dict[str, str]] = []
        for message in self.chat_template_messages:
            rendered.append(
                {
                    "role": message["role"],
                    "content": message["content"].replace("{question}", question),
                }
            )
        return rendered

    def apply_chat_template(self, question: str) -> str:
        messages = self._build_generation_messages(question)
        return self.format_chat(messages)

    def _extract_prompt_and_completion_ids(
        self,
        model_inputs: Dict[str, torch.Tensor],
        generated_ids: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = _as_sequences(generated_ids)
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        prompt_ids = model_inputs["input_ids"]
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_len = prompt_ids.shape[1]
        completion_ids = seq[:, prompt_len:]
        return prompt_ids, completion_ids

    def _decode_completion(self, model_inputs: Dict[str, torch.Tensor], generated_ids: Any) -> str:
        seq = _as_sequences(generated_ids)
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        prompt_len = model_inputs["input_ids"].shape[1]
        completion = seq[0, prompt_len:]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Dict[str, Any]:
        model_inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        generated_ids = self.model.generate(**model_inputs, **kwargs)
        prompt_ids, completion_ids = self._extract_prompt_and_completion_ids(model_inputs, generated_ids)
        answer_text = self._decode_completion(model_inputs, generated_ids)
        return {
            "prompt_text": prompt_text,
            "model_inputs": model_inputs,
            "generated_ids": generated_ids,
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "answer_text": answer_text,
        }

    @torch.no_grad()
    def compute_logprob_stats(self, prompt_ids: Any, completion_ids: Any) -> Dict[str, Any]:
        if isinstance(prompt_ids, torch.Tensor) and prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if isinstance(completion_ids, torch.Tensor) and completion_ids.dim() == 1:
            completion_ids = completion_ids.unsqueeze(0)

        prompt_ids = prompt_ids.to(self.model.device)
        completion_ids = completion_ids.to(self.model.device)
        completion_len = completion_ids.shape[1]

        if completion_len == 0:
            return {
                "completion_log_probs": np.zeros((0,), dtype=np.float64),
                "token_entropies": np.zeros((0,), dtype=np.float64),
                "mean_logp_vocab": np.zeros((0,), dtype=np.float64),
                "vocab_size": int(getattr(self.model.config, "vocab_size", 0)),
            }

        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits = self.model(full_ids).logits
        logp = torch.log_softmax(logits, dim=-1)

        prompt_len = prompt_ids.shape[1]
        sliced_logp = logp[:, prompt_len - 1 : prompt_len - 1 + completion_len, :]

        token_logprobs = torch.gather(
            sliced_logp,
            dim=-1,
            index=completion_ids.unsqueeze(-1),
        ).squeeze(-1)
        sliced_p = torch.exp(sliced_logp)
        token_entropies = -(sliced_p * sliced_logp).sum(dim=-1)
        mean_logp_vocab = sliced_logp.mean(dim=-1)

        return {
            "completion_log_probs": token_logprobs[0].detach().float().cpu().numpy(),
            "token_entropies": token_entropies[0].detach().float().cpu().numpy(),
            "mean_logp_vocab": mean_logp_vocab[0].detach().float().cpu().numpy(),
            "vocab_size": int(sliced_logp.shape[-1]),
        }

    @torch.no_grad()
    def _sample(
        self,
        model_inputs: Dict[str, torch.Tensor],
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[Any]:
        if num_samples <= 0:
            return []
        sampled_ids = []
        for _ in range(num_samples):
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            sampled_ids.append(generated)
        return sampled_ids

    def _decode_sample_texts(
        self,
        model_inputs: Dict[str, torch.Tensor],
        sampled_generated_ids: List[Any],
    ) -> List[str]:
        prompt_len = model_inputs["input_ids"].shape[1]
        texts = []
        for generated in sampled_generated_ids:
            seq = _as_sequences(generated)
            if seq.dim() == 1:
                seq = seq.unsqueeze(0)
            completion = seq[0, prompt_len:]
            texts.append(self.tokenizer.decode(completion, skip_special_tokens=True).strip())
        return texts

    @torch.no_grad()
    def collect_sampling_stats(
        self,
        model_inputs: Dict[str, Any],
        max_new_tokens: int,
        num_samples: int,
        sample_temperature: float,
        sample_top_p: float,
    ) -> Dict[str, Any]:
        sampled = self._sample(
            model_inputs=model_inputs,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=sample_temperature if sample_temperature > 0 else 0.8,
            top_p=sample_top_p,
        )
        if not sampled:
            return {
                "sampled_texts": [],
                "sampled_sequence_nlls": [],
            }

        texts = self._decode_sample_texts(model_inputs, sampled)
        nlls = []
        for generated in sampled:
            prompt_ids, completion_ids = self._extract_prompt_and_completion_ids(model_inputs, generated)
            stats = self.compute_logprob_stats(prompt_ids, completion_ids)
            log_probs = stats["completion_log_probs"]
            nll = float(-np.mean(log_probs)) if len(log_probs) > 0 else 0.0
            nlls.append(nll)
        return {
            "sampled_texts": texts,
            "sampled_sequence_nlls": nlls,
        }

    def _get_single_token_id(self, text: str) -> int:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 1:
            return token_ids[0]
        token_ids = self.tokenizer.encode(" " + text, add_special_tokens=False)
        if len(token_ids) == 1:
            return token_ids[0]
        if not token_ids:
            raise ValueError(f"Unable to resolve token id for text: {text}")
        return token_ids[0]

    @torch.no_grad()
    def compute_p_true_prob(
        self,
        question: str,
        answer_text: str,
        with_context: bool = False,
        extra_context: str = "",
    ) -> float:
        system_prompt = "You are a careful judge. Answer ONLY with True or False."
        if with_context:
            user_prompt = (
                f"Context:\n{extra_context}\n\n"
                f"Question:\n{question}\n\n"
                f"Possible answer:\n{answer_text}\n\n"
                "Is the possible answer:\n(A) True\n(B) False\n\n"
                "The possible answer is:"
            )
        else:
            user_prompt = (
                f"Question:\n{question}\n\n"
                f"Possible answer:\n{answer_text}\n\n"
                "Is the possible answer:\n(A) True\n(B) False\n\n"
                "The possible answer is:"
            )

        prompt = self.format_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits
        next_token_logits = logits[0, -1, :]

        true_id = self._get_single_token_id("True")
        false_id = self._get_single_token_id("False")
        probs = torch.softmax(
            torch.stack([next_token_logits[true_id], next_token_logits[false_id]]),
            dim=0,
        )
        return float(probs[0].item())

    @torch.no_grad()
    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        prompt = self.format_chat(messages)
        result = self.generate(
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return result["answer_text"]
