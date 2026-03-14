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

    @torch.no_grad()
    def collect_sampling_stats(
        self,
        model_inputs: Dict[str, Any],
        max_new_tokens: int,
        num_samples: int,
        sample_temperature: float,
        sample_top_p: float,
        need_eigenscore_embeddings: bool = False,
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
                "eigenscore_embeddings": [],
            }

        texts = self._decode_sample_texts(model_inputs, sampled)
        nlls = []
        eigenscore_embeddings = []
        for generated in sampled:
            prompt_ids, completion_ids = self._extract_prompt_and_completion_ids(model_inputs, generated)
            stats = self.compute_logprob_stats(prompt_ids, completion_ids)
            log_probs = stats["completion_log_probs"]
            nll = float(-np.mean(log_probs)) if len(log_probs) > 0 else 0.0
            nlls.append(nll)
            if need_eigenscore_embeddings:
                eigenscore_embeddings.append(
                    self._extract_eigenscore_embedding(model_inputs=model_inputs, generated_ids=generated)
                )
        return {
            "sampled_texts": texts,
            "sampled_sequence_nlls": nlls,
            "eigenscore_embeddings": eigenscore_embeddings,
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

    def _lazy_load_nli_model(self, nli_model_path: str, nli_device: str = None):
        if not hasattr(self, "nli_model") or self._nli_model_path != nli_model_path:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
            
            # Use requested device or fallback to main device
            target_device = nli_device if nli_device else self.device
            
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_path, trust_remote_code=True)
            self.nli_is_generative = False
            
            if "verifier" in nli_model_path.lower() or "llm" in nli_model_path.lower() or "qwen" in nli_model_path.lower():
                self.nli_is_generative = True
                self.nli_model = AutoModelForCausalLM.from_pretrained(
                    nli_model_path, trust_remote_code=True, torch_dtype=torch.float16
                ).to(target_device).eval()
                if self.nli_tokenizer.pad_token is None:
                    self.nli_tokenizer.pad_token = self.nli_tokenizer.eos_token
                self.nli_tokenizer.padding_side = "left"
            else:
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                    nli_model_path, trust_remote_code=True
                ).to(target_device).eval()
                
            self._nli_model_path = nli_model_path
            self._nli_device = target_device

    @torch.no_grad()
    def compute_nli_affinity_matrix(
        self,
        question: str,
        answers: List[str],
        nli_model_path: str,
        affinity_mode: str = "disagreement_w",
        temperature: float = 3.0,
        symmetric: bool = True,
        nli_device: str = None
    ) -> np.ndarray:
        if not nli_model_path:
            raise ValueError("nli_model_path is required for NLI similarity matrix computation.")

        self._lazy_load_nli_model(nli_model_path, nli_device=nli_device)
        target_device = self._nli_device
        
        unique_ans = sorted(list(set(answers)))
        n_unique = len(unique_ans)
        
        if getattr(self, "nli_is_generative", False):
            # Generative Verifier (e.g. TIGER-Lab/general-verifier) Mode
            sim_mat_unique = torch.zeros((n_unique, n_unique), device=target_device)
            pairs_to_compute = []
            indices = []
            
            for i in range(n_unique):
                for j in range(n_unique):
                    if i == j: 
                        sim_mat_unique[i, j] = 1.0
                        continue
                    prompt = f"Question: {question}\nReference Answer: {unique_ans[i]}\nStudent Answer: {unique_ans[j]}\nIs the student answer correct?"
                    try:
                        messages = [{"role": "user", "content": prompt}]
                        chat_prompt = self.nli_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        chat_prompt = prompt
                    pairs_to_compute.append(chat_prompt)
                    indices.append((i, j))
                    
            if pairs_to_compute:
                batch_size = 16 
                try:
                    true_id = self.nli_tokenizer.encode("True", add_special_tokens=False)[-1]
                    false_id = self.nli_tokenizer.encode("False", add_special_tokens=False)[-1]
                except Exception:
                    true_id, false_id = 0, 1
                    
                for start_idx in range(0, len(pairs_to_compute), batch_size):
                    batch_pairs = pairs_to_compute[start_idx : start_idx + batch_size]
                    inputs = self.nli_tokenizer(batch_pairs, padding=True, truncation=True, return_tensors="pt").to(target_device)
                    
                    with torch.no_grad():
                        logits = self.nli_model(**inputs).logits[:, -1, :]
                        
                    for b_idx in range(len(batch_pairs)):
                        t_logit = logits[b_idx, true_id]
                        f_logit = logits[b_idx, false_id]
                        # We use temperature scaling on logits 
                        prob_true = float(torch.softmax(torch.stack([t_logit / temperature, f_logit / temperature]), dim=0)[0].item())
                        
                        actual_idx = start_idx + b_idx
                        i, j = indices[actual_idx]
                        sim_mat_unique[i, j] = prob_true
                        
            W_unique = sim_mat_unique
            if symmetric:
                W_unique = (W_unique + W_unique.permute(1, 0)) / 2
                
        else:
            # Traditional Sequence Classification Mode (DeBERTa-MNLI)
            num_labels = getattr(self.nli_model.config, "num_labels", 3)
            sim_mat_unique = torch.zeros((n_unique, n_unique, num_labels), device=target_device)
            pairs_to_compute = []
            indices = []
            
            for i in range(n_unique):
                for j in range(n_unique):
                    if i == j: continue
                    # Input format for DeBERTa MNLI: premise [SEP] hypothesis
                    pairs_to_compute.append(f"{question} {unique_ans[i]} [SEP] {question} {unique_ans[j]}")
                    indices.append((i, j))
                    
            if pairs_to_compute:
                batch_size = 64
                all_logits = []
                for start_idx in range(0, len(pairs_to_compute), batch_size):
                    batch_pairs = pairs_to_compute[start_idx : start_idx + batch_size]
                    inputs = self.nli_tokenizer(batch_pairs, padding=True, truncation=True, return_tensors="pt").to(target_device)
                    logits = self.nli_model(**inputs).logits
                    all_logits.append(logits)
                
                all_logits = torch.cat(all_logits, dim=0)
                for idx, (i, j) in enumerate(indices):
                    sim_mat_unique[i, j] = all_logits[idx]
                    
            # Fetch the exact indices for contradiction and entailment from the model config
            label2id = getattr(self.nli_model.config, "label2id", {})
            
            # Fallback mappings
            contradiction_id = 0
            entailment_id = 2 if num_labels > 2 else 1
            
            if label2id:
                for label, idx in label2id.items():
                    if "contradiction" in label.lower():
                        contradiction_id = idx
                    elif "entailment" in label.lower() or "entail" in label.lower():
                        entailment_id = idx
            
            # Calculate affinity from logits based on mode
            if affinity_mode == 'disagreement':
                sim_mat_unique = (sim_mat_unique + sim_mat_unique.permute(1, 0, 2)) / 2
                W_unique = (sim_mat_unique.argmax(-1) != contradiction_id).float()
            elif affinity_mode == 'disagreement_w':
                probs = torch.softmax(sim_mat_unique / temperature, dim=-1)
                # Probability of contradiction
                W_unique = probs[:, :, contradiction_id]
                if symmetric:
                    W_unique = (W_unique + W_unique.permute(1, 0)) / 2
                # Affinity is 1 - disagreement
                W_unique = 1.0 - W_unique
            elif affinity_mode == 'agreement':
                sim_mat_unique = (sim_mat_unique + sim_mat_unique.permute(1, 0, 2)) / 2
                W_unique = (sim_mat_unique.argmax(-1) == entailment_id).float()
            elif affinity_mode == 'agreement_w':
                probs = torch.softmax(sim_mat_unique / temperature, dim=-1)
                # Probability of entailment
                W_unique = probs[:, :, entailment_id]
                if symmetric:
                    W_unique = (W_unique + W_unique.permute(1, 0)) / 2
            else:
                raise ValueError(f"Unknown affinity_mode: {affinity_mode}")
            
        W_unique = W_unique.cpu().numpy()
        np.fill_diagonal(W_unique, 1.0)
        
        # Map back from unique to original answers
        n_orig = len(answers)
        W = np.zeros((n_orig, n_orig), dtype=np.float32)
        ans_to_idx = {ans: i for i, ans in enumerate(unique_ans)}
        
        for i in range(n_orig):
            for j in range(n_orig):
                W[i, j] = W_unique[ans_to_idx[answers[i]], ans_to_idx[answers[j]]]
                
        return W
