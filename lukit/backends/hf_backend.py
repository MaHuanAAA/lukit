import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from ..methods.semantic_class_utils import (
    build_semantic_classes_from_incremental_judger,
    build_semantic_classes_from_mutual_entailment,
)
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
        self._nli_cache_key: Optional[Tuple[str, str, str]] = None
        self._nli_tokenizer = None
        self._nli_model = None
        self._nli_label_indices: Optional[Tuple[int, int, int]] = None
        self._equivalence_judger_cache_key: Optional[Tuple[str, str, str]] = None
        self._equivalence_judger_tokenizer = None
        self._equivalence_judger_model = None

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

    def _extract_eigenscore_embedding(
        self,
        model_inputs: Dict[str, torch.Tensor],
        generated_ids: Any,
    ) -> np.ndarray:
        seq = _as_sequences(generated_ids)
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)

        prompt_len = int(model_inputs["input_ids"].shape[1])
        seq_len = int(seq.shape[1])
        completion_len = max(seq_len - prompt_len, 0)

        if completion_len <= 0:
            selected_abs_idx = seq_len - 1
        else:
            selected_rel_idx = completion_len - 2
            if selected_rel_idx < 0:
                selected_rel_idx = completion_len - 1
            selected_abs_idx = min(prompt_len + selected_rel_idx, seq_len - 1)

        outputs = self.model(seq.to(self.model.device), output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if not hidden_states:
            return np.zeros((0,), dtype=np.float64)

        selected_layer = len(hidden_states) // 2
        layer_states = hidden_states[selected_layer]
        return layer_states[0, selected_abs_idx, :].detach().float().cpu().numpy()

    @staticmethod
    def _compute_jaccard_similarity_matrix(sampled_texts: List[str]) -> np.ndarray:
        n = len(sampled_texts)
        matrix = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                tokens_i = set(sampled_texts[i].lower().split())
                tokens_j = set(sampled_texts[j].lower().split())
                union = len(tokens_i | tokens_j)
                score = 0.0 if union == 0 else float(len(tokens_i & tokens_j) / union)
                matrix[i, j] = score
                matrix[j, i] = score
        return matrix

    @staticmethod
    def _resolve_nli_label_indices(model: Any) -> Tuple[int, int, int]:
        id2label = getattr(model.config, "id2label", {}) or {}
        normalized = {
            int(idx): str(label).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
            for idx, label in id2label.items()
        }

        contra_idx = None
        neutral_idx = None
        entail_idx = None
        for idx, label in normalized.items():
            if "contradict" in label or "contra" in label:
                contra_idx = idx
            elif "neutral" in label:
                neutral_idx = idx
            elif "entail" in label:
                entail_idx = idx

        if neutral_idx is None:
            remaining = [idx for idx in normalized if idx not in {contra_idx, entail_idx}]
            if remaining:
                neutral_idx = remaining[0]

        if entail_idx is None or contra_idx is None or neutral_idx is None:
            raise ValueError(
                "Unable to resolve contradiction/neutral/entailment labels from the NLI model config."
            )
        return contra_idx, neutral_idx, entail_idx

    def _ensure_nli_model(
        self,
        model_path: str,
        device: str,
        torch_dtype: str,
    ) -> Tuple[Any, Any, Tuple[int, int, int]]:
        if not model_path:
            raise ValueError(
                "NLI similarity requires --nli_model_path (or nli_model_path in Python API)."
            )

        resolved_device = _resolve_device(device)
        cache_key = (model_path, resolved_device, str(torch_dtype))
        if self._nli_cache_key == cache_key and self._nli_model is not None and self._nli_tokenizer is not None:
            return self._nli_tokenizer, self._nli_model, self._nli_label_indices  # type: ignore[return-value]

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
        )
        dtype_arg: Any = torch_dtype
        if isinstance(torch_dtype, str) and torch_dtype.lower() in _DTYPE_MAP:
            dtype_arg = _DTYPE_MAP[torch_dtype.lower()]
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=dtype_arg,
            trust_remote_code=True,
        )
        model.to(resolved_device)
        model.eval()
        label_indices = self._resolve_nli_label_indices(model)

        self._nli_cache_key = cache_key
        self._nli_tokenizer = tokenizer
        self._nli_model = model
        self._nli_label_indices = label_indices
        return tokenizer, model, label_indices

    def _ensure_equivalence_judger_model(
        self,
        model_path: str,
        device: str,
        torch_dtype: str,
    ) -> Tuple[Any, Any]:
        if not model_path:
            raise ValueError(
                "Semantic equivalence judger requires equivalence_judger_model_path."
            )

        resolved_device = _resolve_device(device)
        cache_key = (model_path, resolved_device, str(torch_dtype))
        if (
            self._equivalence_judger_cache_key == cache_key
            and self._equivalence_judger_model is not None
            and self._equivalence_judger_tokenizer is not None
        ):
            return self._equivalence_judger_tokenizer, self._equivalence_judger_model

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype_arg: Any = torch_dtype
        if isinstance(torch_dtype, str) and torch_dtype.lower() in _DTYPE_MAP:
            dtype_arg = _DTYPE_MAP[torch_dtype.lower()]
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype_arg,
            trust_remote_code=True,
        )
        model.to(resolved_device)
        model.eval()

        self._equivalence_judger_cache_key = cache_key
        self._equivalence_judger_tokenizer = tokenizer
        self._equivalence_judger_model = model
        return tokenizer, model

    @staticmethod
    def _build_equivalence_judger_prompt(
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> str:
        return (
            f"User: ### Question: {question}\n\n"
            f"### Reference Answer: {answer_a}\n\n"
            f"### Candidate Answer: {answer_b}\n\n"
            "For the above question, please verify if the candidate answer is "
            "semantically equivalent to the reference answer.\n"
            "Do not solve the question by yourself; only judge semantic equivalence.\n"
            'If the candidate answer is equivalent, output "Final Decision: Yes". '
            'If it is not equivalent, output "Final Decision: No". Assistant:'
        )

    @staticmethod
    def _parse_equivalence_judger_output(text: str) -> bool:
        lowered = str(text).strip().lower()
        if "final decision: yes" in lowered:
            return True
        if "final decision: no" in lowered:
            return False
        if lowered.endswith("yes") and "no" not in lowered:
            return True
        return False

    @torch.no_grad()
    def compute_equivalence_judger_classes(
        self,
        question: str,
        sampled_texts: List[str],
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 16,
    ) -> Dict[str, List]:
        texts = [str(text).strip() for text in sampled_texts]
        if not texts:
            return {
                "sample_to_class": [],
                "class_to_sample": [],
            }

        tokenizer, model = self._ensure_equivalence_judger_model(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
        )

        def is_equivalent(i: int, j: int) -> bool:
            prompt = self._build_equivalence_judger_prompt(question, texts[i], texts[j])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            generated = outputs[0, inputs["input_ids"].shape[1] :]
            decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
            return self._parse_equivalence_judger_output(decoded)

        return build_semantic_classes_from_incremental_judger(texts, is_equivalent)

    @torch.no_grad()
    def compute_semantic_graph_stats(
        self,
        sampled_texts: List[str],
        similarity_score: str = "nli",
        affinity: str = "disagreement_w",
        temperature: float = 3.0,
        nli_model_path: str = "",
        nli_device: str = "auto",
        nli_torch_dtype: str = "auto",
    ) -> Dict[str, Any]:
        texts = [str(text).strip() for text in sampled_texts]
        n = len(texts)
        empty = {
            "similarity_matrix": np.zeros((0, 0), dtype=np.float64),
            "semantic_nli_logits": np.zeros((0, 0, 3), dtype=np.float64),
            "semantic_matrix_entail": np.zeros((0, 0), dtype=np.float64),
            "semantic_matrix_contra": np.zeros((0, 0), dtype=np.float64),
        }
        if n == 0:
            return empty
        if n == 1:
            return {
                "similarity_matrix": np.eye(1, dtype=np.float64),
                "semantic_nli_logits": np.zeros((1, 1, 3), dtype=np.float64),
                "semantic_matrix_entail": np.eye(1, dtype=np.float64),
                "semantic_matrix_contra": np.zeros((1, 1), dtype=np.float64),
            }

        similarity_score = str(similarity_score).lower()
        affinity = str(affinity).lower()
        if similarity_score == "jaccard":
            similarity = self._compute_jaccard_similarity_matrix(texts)
            return {
                "similarity_matrix": similarity,
                "semantic_nli_logits": np.zeros((n, n, 3), dtype=np.float64),
                "semantic_matrix_entail": np.eye(n, dtype=np.float64),
                "semantic_matrix_contra": np.zeros((n, n), dtype=np.float64),
            }
        if similarity_score != "nli":
            raise ValueError(f"Unsupported semantic similarity score: {similarity_score}")

        tokenizer, model, (contra_idx, neutral_idx, entail_idx) = self._ensure_nli_model(
            model_path=nli_model_path,
            device=nli_device,
            torch_dtype=nli_torch_dtype,
        )

        nli_logits = np.zeros((n, n, 3), dtype=np.float64)
        entail = np.eye(n, dtype=np.float64)
        contra = np.zeros((n, n), dtype=np.float64)
        pair_indices: List[Tuple[int, int]] = []
        premise_texts: List[str] = []
        hypothesis_texts: List[str] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pair_indices.append((i, j))
                premise_texts.append(texts[i])
                hypothesis_texts.append(texts[j])

        batch_size = 16
        model_max_length = getattr(tokenizer, "model_max_length", 512)
        if not isinstance(model_max_length, int) or model_max_length <= 0 or model_max_length > 4096:
            model_max_length = 512
        for start in range(0, len(pair_indices), batch_size):
            batch_pairs = pair_indices[start : start + batch_size]
            inputs = tokenizer(
                premise_texts[start : start + batch_size],
                hypothesis_texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=model_max_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(model.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            logits = model(**inputs).logits.detach().float().cpu().numpy()
            probs = torch.softmax(torch.from_numpy(logits) / float(temperature), dim=-1).numpy()
            for row_idx, (i, j) in enumerate(batch_pairs):
                canonical_logits = np.asarray(
                    [
                        logits[row_idx, contra_idx],
                        logits[row_idx, neutral_idx],
                        logits[row_idx, entail_idx],
                    ],
                    dtype=np.float64,
                )
                nli_logits[i, j] = canonical_logits
                contra[i, j] = float(probs[row_idx, contra_idx])
                entail[i, j] = float(probs[row_idx, entail_idx])

        if affinity in {"agreement_w", "entail", "entailment"}:
            similarity = entail
        else:
            similarity = 1.0 - contra
        similarity = (similarity + similarity.T) / 2.0
        np.fill_diagonal(similarity, 1.0)
        return {
            "similarity_matrix": similarity,
            "semantic_nli_logits": nli_logits,
            "semantic_matrix_entail": entail,
            "semantic_matrix_contra": contra,
        }

    @torch.no_grad()
    def collect_sampling_stats(
        self,
        model_inputs: Dict[str, Any],
        question: str,
        max_new_tokens: int,
        num_samples: int,
        sample_temperature: float,
        sample_top_p: float,
        need_eigenscore_embeddings: bool = False,
        need_semantic_matrices: bool = False,
        need_semantic_classes: bool = False,
        semantic_similarity_score: str = "nli",
        semantic_affinity: str = "disagreement_w",
        semantic_temperature: float = 3.0,
        semantic_class_source: str = "nli",
        nli_model_path: str = "",
        nli_device: str = "auto",
        nli_torch_dtype: str = "auto",
        equivalence_judger_model_path: str = "",
        equivalence_judger_device: str = "auto",
        equivalence_judger_torch_dtype: str = "auto",
        equivalence_judger_max_new_tokens: int = 16,
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
                "sampled_sequence_logprobs": [],
                "eigenscore_embeddings": [],
            }

        texts = self._decode_sample_texts(model_inputs, sampled)
        nlls = []
        sequence_logprobs = []
        eigenscore_embeddings = []
        for generated in sampled:
            prompt_ids, completion_ids = self._extract_prompt_and_completion_ids(model_inputs, generated)
            stats = self.compute_logprob_stats(prompt_ids, completion_ids)
            log_probs = stats["completion_log_probs"]
            total_logprob = float(np.sum(log_probs)) if len(log_probs) > 0 else 0.0
            nll = float(-np.mean(log_probs)) if len(log_probs) > 0 else 0.0
            sequence_logprobs.append(total_logprob)
            nlls.append(nll)
            if need_eigenscore_embeddings:
                eigenscore_embeddings.append(
                    self._extract_eigenscore_embedding(model_inputs=model_inputs, generated_ids=generated)
                )
        output = {
            "sampled_texts": texts,
            "sampled_sequence_nlls": nlls,
            "sampled_sequence_logprobs": sequence_logprobs,
            "eigenscore_embeddings": eigenscore_embeddings,
        }
        if need_semantic_matrices:
            output.update(
                self.compute_semantic_graph_stats(
                    sampled_texts=texts,
                    similarity_score=semantic_similarity_score,
                    affinity=semantic_affinity,
                    temperature=semantic_temperature,
                    nli_model_path=nli_model_path,
                    nli_device=nli_device,
                    nli_torch_dtype=nli_torch_dtype,
                )
            )
        if need_semantic_classes:
            semantic_class_source = str(semantic_class_source).lower()
            if semantic_class_source == "nli":
                if np.asarray(output.get("semantic_nli_logits", [])).shape != (len(texts), len(texts), 3):
                    output.update(
                        self.compute_semantic_graph_stats(
                            sampled_texts=texts,
                            similarity_score="nli",
                            affinity=semantic_affinity,
                            temperature=semantic_temperature,
                            nli_model_path=nli_model_path,
                            nli_device=nli_device,
                            nli_torch_dtype=nli_torch_dtype,
                        )
                    )
                output["semantic_classes"] = build_semantic_classes_from_mutual_entailment(output)
            elif semantic_class_source == "equivalence_judger":
                output["semantic_classes"] = self.compute_equivalence_judger_classes(
                    question=question,
                    sampled_texts=texts,
                    model_path=equivalence_judger_model_path,
                    device=equivalence_judger_device,
                    torch_dtype=equivalence_judger_torch_dtype,
                    max_new_tokens=equivalence_judger_max_new_tokens,
                )
            else:
                raise ValueError(f"Unsupported semantic_class_source: {semantic_class_source}")
        return output

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
