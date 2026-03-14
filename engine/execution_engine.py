from typing import Any, Dict, Iterable, List

from ..backends import HFBackend
from ..methods.base_method import UncertaintyMethod
from ..progress import wrap_progress
from .dependency_resolver import DependencyResolver


def _extract_reference_answer(sample: Dict[str, Any]) -> str:
    answer = sample.get("answer")
    if isinstance(answer, dict):
        for key in ("value", "normalized_value"):
            value = answer.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("aliases", "normalized_aliases"):
            value = answer.get(key)
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(sample.get("a_gold"), str):
        return str(sample["a_gold"]).strip()
    return ""


def _extract_aliases(sample: Dict[str, Any]) -> List[str]:
    answer = sample.get("answer", {})
    aliases: List[str] = []
    if isinstance(answer, dict):
        for key in ("value", "normalized_value"):
            value = answer.get(key)
            if isinstance(value, str) and value.strip():
                aliases.append(value.strip())
        for key in ("aliases", "normalized_aliases"):
            values = answer.get(key)
            if isinstance(values, list):
                aliases.extend([x.strip() for x in values if isinstance(x, str) and x.strip()])
    elif isinstance(answer, str) and answer.strip():
        aliases.append(answer.strip())
    if not aliases and isinstance(sample.get("a_gold"), str):
        aliases = [str(sample["a_gold"]).strip()]
    dedup: List[str] = []
    seen = set()
    for alias in aliases:
        key = alias.lower()
        if key and key not in seen:
            seen.add(key)
            dedup.append(alias)
    return dedup


class ExecutionEngine:
    """Dependency-aware execution engine for UQ methods."""

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        backend_config: Dict[str, Any] = None,
        cache_path: str = "",
    ) -> None:
        cfg = dict(backend_config or {})
        backend_type = cfg.pop("type", "huggingface")
        if backend_type != "huggingface":
            raise NotImplementedError(f"Unsupported backend type: {backend_type}")

        if model is None:
            model = cfg.pop("model_path", None)
        self.backend = HFBackend(model=model, tokenizer=tokenizer, **cfg)
        self.cache_path = cache_path
        self._resolver = DependencyResolver()

    def _validate_methods(self, methods: Iterable[UncertaintyMethod]) -> None:
        for method in methods:
            if method.supported_backends and self.backend.name not in method.supported_backends:
                raise ValueError(
                    f"Method {method.name} does not support backend {self.backend.name}. "
                    f"Supported: {method.supported_backends}"
                )

    def _run_one_sample(
        self,
        question: str,
        answer_gold: str,
        answer_aliases: List[str],
        methods: List[UncertaintyMethod],
        providers: List[Any],
        **kwargs,
    ) -> Dict[str, Any]:
        runtime_kwargs = dict(kwargs)
        if "p_true_with_context" not in runtime_kwargs:
            for method in methods:
                if method.name == "p_true" and hasattr(method, "with_context"):
                    runtime_kwargs["p_true_with_context"] = bool(getattr(method, "with_context"))
                    break
        if "need_eigenscore_embeddings" not in runtime_kwargs:
            runtime_kwargs["need_eigenscore_embeddings"] = any(
                method.name == "eigenscore" for method in methods
            )

        prompt_text = self.backend.apply_chat_template(question)
        model_outputs = self.backend.generate(
            prompt_text=prompt_text,
            max_new_tokens=int(runtime_kwargs.get("max_new_tokens", 64)),
            temperature=float(runtime_kwargs.get("temperature", 0.0)),
            top_p=float(runtime_kwargs.get("top_p", 0.9)),
        )
        model_outputs["question"] = question
        model_outputs["extra_context"] = str(runtime_kwargs.get("extra_context", ""))

        stats: Dict[str, Any] = {}
        warnings: List[str] = []
        for provider in providers:
            # pass accumulated stats so downstream providers can use upstream results e.g. sampling_stats
            provided = provider(model_outputs, backend=self.backend, **runtime_kwargs, **stats)
            if isinstance(provided, dict) and "__warning__" in provided:
                warnings.append(str(provided["__warning__"]))
                provided = {k: v for k, v in provided.items() if k != "__warning__"}
            stats[provider.provides] = provided

        method_outputs: Dict[str, Dict[str, Any]] = {}
        for method in methods:
            local_stats = {k: stats[k] for k in method.requires_data}
            method_outputs[method.name] = method.compute(local_stats)

        output = {
            "device": self.backend.device,
            "q": question,
            "a_gold": answer_gold,
            "a_gold_aliases": answer_aliases,
            "a_model": model_outputs["answer_text"],
            "u": method_outputs,
        }
        if warnings:
            output["warnings"] = warnings
        return output

    def run(
        self,
        dataset: Iterable[Dict[str, Any]],
        methods: List[UncertaintyMethod],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        runtime_kwargs = dict(kwargs)
        show_progress = bool(runtime_kwargs.pop("show_progress", False))
        progress_desc = str(runtime_kwargs.pop("progress_desc", "Generation"))

        methods = list(methods)
        if not methods:
            return []
        self._validate_methods(methods)

        required_data = set()
        for method in methods:
            required_data.update(method.requires_data)
        providers = self._resolver.resolve(required_data)

        total = None
        if show_progress:
            try:
                total = len(dataset)  # type: ignore[arg-type]
            except Exception:
                total = None

        sample_iter = dataset
        if show_progress:
            sample_iter = wrap_progress(sample_iter, desc=progress_desc, total=total)

        outputs: List[Dict[str, Any]] = []
        for idx, sample in enumerate(sample_iter):
            question = str(sample.get("question", "")).strip()
            answer_gold = _extract_reference_answer(sample)
            aliases = _extract_aliases(sample)
            record = self._run_one_sample(
                question=question,
                answer_gold=answer_gold,
                answer_aliases=aliases,
                methods=methods,
                providers=providers,
                **runtime_kwargs,
            )
            record["sample_idx"] = int(sample.get("sample_idx", idx))
            outputs.append(record)
        return outputs

    def run_single(
        self,
        prompt: str,
        method: UncertaintyMethod,
        answer_gold: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        sample = {"question": prompt, "answer": answer_gold, "sample_idx": 0}
        records = self.run([sample], [method], **kwargs)
        return records[0]
