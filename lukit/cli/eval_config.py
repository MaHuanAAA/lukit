from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from ..methods import METHOD_REGISTRY, list_methods
from ..methods.semantic_graph_utils import canonicalize_affinity

SEMANTIC_GRAPH_METHODS = {
    "deg_mat",
    "eccentricity",
    "eig_val_laplacian",
    "num_sem_sets",
}

SEMANTIC_ENTROPY_METHODS = {
    "semantic_entropy",
    "semantic_entropy_empirical",
}


@dataclass(frozen=True)
class GenerationConfig:
    model_path: str
    device: str
    torch_dtype: str
    chat_template_config: str
    max_new_tokens: int
    temperature: float
    top_p: float

    def backend_config(self) -> Dict[str, Any]:
        return {
            "type": "huggingface",
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "chat_template_config": self.chat_template_config,
        }

    def runtime_kwargs(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    name: str
    mode: str
    directory: str
    start_idx: int
    num_samples_eval: int


@dataclass(frozen=True)
class SamplingConfig:
    num_samples: int
    temperature: float
    top_p: float

    def runtime_kwargs(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "sample_temperature": self.temperature,
            "sample_top_p": self.top_p,
        }


@dataclass(frozen=True)
class LexicalSimilarityConfig:
    metric: str

    def method_kwargs(self) -> Dict[str, Any]:
        return {"metric": self.metric}


@dataclass(frozen=True)
class PTrueConfig:
    with_context: bool

    def method_kwargs(self) -> Dict[str, Any]:
        return {"with_context": self.with_context}

    def runtime_kwargs(self) -> Dict[str, Any]:
        return {"p_true_with_context": self.with_context}


@dataclass(frozen=True)
class SemanticGraphConfig:
    similarity_score: str
    affinity: str
    temperature: float
    jaccard_threshold: float

    def method_kwargs(self) -> Dict[str, Any]:
        return {
            "similarity_score": self.similarity_score,
            "affinity": self.affinity,
            "temperature": self.temperature,
            "jaccard_threshold": self.jaccard_threshold,
        }

    def runtime_kwargs(self) -> Dict[str, Any]:
        return {
            "semantic_similarity_score": self.similarity_score,
            "semantic_affinity": self.affinity,
            "semantic_temperature": self.temperature,
            "semantic_jaccard_threshold": self.jaccard_threshold,
        }


@dataclass(frozen=True)
class SemanticEntropyConfig:
    class_source: str

    def method_kwargs(self) -> Dict[str, Any]:
        return {"semantic_class_source": self.class_source}

    def runtime_kwargs(self) -> Dict[str, Any]:
        return {"semantic_class_source": self.class_source}


@dataclass(frozen=True)
class NLIConfig:
    model_path: str
    device: str
    torch_dtype: str

    def runtime_kwargs(self) -> Dict[str, Any]:
        return {
            "nli_model_path": self.model_path,
            "nli_device": self.device,
            "nli_torch_dtype": self.torch_dtype,
        }

    def validate_required(self, message: str) -> None:
        if not self.model_path:
            raise ValueError(message)


@dataclass(frozen=True)
class EquivalenceJudgerConfig:
    model_path: str
    device: str
    torch_dtype: str
    max_new_tokens: int

    def runtime_kwargs(self) -> Dict[str, Any]:
        return {
            "equivalence_judger_model_path": self.model_path,
            "equivalence_judger_device": self.device,
            "equivalence_judger_torch_dtype": self.torch_dtype,
            "equivalence_judger_max_new_tokens": self.max_new_tokens,
        }

    def validate_required(self, message: str) -> None:
        if not self.model_path:
            raise ValueError(message)


@dataclass(frozen=True)
class JudgeConfig:
    model_path: str
    device: str
    max_new_tokens: int
    mode: str


@dataclass(frozen=True)
class OutputConfig:
    out_jsonl: str
    out_metrics: str


@dataclass(frozen=True)
class EvalConfig:
    list_methods_flag: bool
    methods_raw: str
    generation: GenerationConfig
    dataset: DatasetConfig
    sampling: SamplingConfig
    lexical_similarity: LexicalSimilarityConfig
    p_true: PTrueConfig
    semantic_graph: SemanticGraphConfig
    semantic_entropy: SemanticEntropyConfig
    nli: NLIConfig
    equivalence_judger: EquivalenceJudgerConfig
    judge: JudgeConfig
    output: OutputConfig

    @classmethod
    def from_args(cls, args: Any) -> "EvalConfig":
        return cls(
            list_methods_flag=bool(args.list_methods),
            methods_raw=str(args.methods),
            generation=GenerationConfig(
                model_path=str(args.gm_model_path),
                device=str(args.gm_device),
                torch_dtype=str(args.torch_dtype),
                chat_template_config=str(args.chat_template_config),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            ),
            dataset=DatasetConfig(
                source=str(args.dataset_source),
                name=str(args.dataset_name),
                mode=str(args.dataset_mode),
                directory=str(args.dataset_dir),
                start_idx=int(args.start_idx),
                num_samples_eval=int(args.num_samples_eval),
            ),
            sampling=SamplingConfig(
                num_samples=int(args.num_samples),
                temperature=float(args.sample_temperature),
                top_p=float(args.sample_top_p),
            ),
            lexical_similarity=LexicalSimilarityConfig(metric=str(args.lexical_metric)),
            p_true=PTrueConfig(with_context=bool(args.p_true_with_context)),
            semantic_graph=SemanticGraphConfig(
                similarity_score=str(args.semantic_similarity_score).lower(),
                affinity=canonicalize_affinity(str(args.semantic_affinity)),
                temperature=float(args.semantic_temperature),
                jaccard_threshold=float(args.semantic_jaccard_threshold),
            ),
            semantic_entropy=SemanticEntropyConfig(
                class_source=str(args.semantic_class_source).lower(),
            ),
            nli=NLIConfig(
                model_path=str(args.nli_model_path),
                device=str(args.nli_device),
                torch_dtype=str(args.nli_torch_dtype),
            ),
            equivalence_judger=EquivalenceJudgerConfig(
                model_path=str(args.equivalence_judger_model_path),
                device=str(args.equivalence_judger_device),
                torch_dtype=str(args.equivalence_judger_torch_dtype),
                max_new_tokens=int(args.equivalence_judger_max_new_tokens),
            ),
            judge=JudgeConfig(
                model_path=str(args.jm_model_path),
                device=str(args.jm_device),
                max_new_tokens=int(args.judge_max_new_tokens),
                mode=str(args.judge_mode),
            ),
            output=OutputConfig(
                out_jsonl=str(args.out_jsonl),
                out_metrics=str(args.out_metrics),
            ),
        )

    def selected_method_names(self) -> List[str]:
        selected = [item.strip() for item in self.methods_raw.split(",") if item.strip()]
        if not selected or selected == ["all"]:
            return list_methods()
        for name in selected:
            if name not in METHOD_REGISTRY:
                raise KeyError(f"Unknown method: {name}")
        return selected

    def method_kwargs(self, method_name: str) -> Dict[str, Any]:
        if method_name == "lexical_similarity":
            return self.lexical_similarity.method_kwargs()
        if method_name == "p_true":
            return self.p_true.method_kwargs()
        if method_name in SEMANTIC_GRAPH_METHODS:
            return self.semantic_graph.method_kwargs()
        if method_name in SEMANTIC_ENTROPY_METHODS:
            return self.semantic_entropy.method_kwargs()
        return {}

    def validate_for_methods(self, method_names: Sequence[str]) -> None:
        if any(name in SEMANTIC_GRAPH_METHODS for name in method_names):
            if self.semantic_graph.similarity_score == "nli":
                self.nli.validate_required(
                    "--nli_model_path is required when semantic graph methods use "
                    "--semantic_similarity_score nli."
                )

        if any(name in SEMANTIC_ENTROPY_METHODS for name in method_names):
            if self.semantic_entropy.class_source == "nli":
                self.nli.validate_required(
                    "--nli_model_path is required when semantic entropy uses "
                    "--semantic_class_source nli."
                )
            if self.semantic_entropy.class_source == "equivalence_judger":
                self.equivalence_judger.validate_required(
                    "--equivalence_judger_model_path is required when semantic entropy uses "
                    "--semantic_class_source equivalence_judger."
                )

    def runtime_kwargs(self) -> Dict[str, Any]:
        runtime: Dict[str, Any] = {}
        runtime.update(self.generation.runtime_kwargs())
        runtime.update(self.sampling.runtime_kwargs())
        runtime.update(self.p_true.runtime_kwargs())
        runtime.update(self.semantic_graph.runtime_kwargs())
        runtime.update(self.semantic_entropy.runtime_kwargs())
        runtime.update(self.nli.runtime_kwargs())
        runtime.update(self.equivalence_judger.runtime_kwargs())
        return runtime
