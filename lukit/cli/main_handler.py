import argparse
import json
import math
import os
import re
import string
from typing import Dict, List, Optional

from ..backends import HFBackend
from ..engine import ExecutionEngine
from ..methods import METHOD_REGISTRY, create_method, list_methods
from ..progress import wrap_progress
from .eval_config import EvalConfig, JudgeConfig

_DTYPE_MAP = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}

_JSON_JUDGE_TORCH_DTYPE = "auto"
_DEFAULT_JSON_JUDGE_MODEL_PATH = "/data1/chenjingdong/ms/Qwen__Qwen3-4B-Instruct-2507"
_CHATGLM_JUDGE_TORCH_DTYPE = "bfloat16"
_CHATGLM_HTTP_PROXY = "http://agent.baidu.com:8188"
_CHATGLM_HTTPS_PROXY = "http://agent.baidu.com:8188"
_CHATGLM_NO_PROXY = "baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com"

_HF_DATASET_NAMES = {"trivia_qa_split", "simple_qa"}
_JSONL_DATASET_NAMES = {
    "chinese_simpleqa",
    "hotpot_qa",
    "nq_open",
    "simpleqa_verified",
    "triviaqa_validation",
    "webqa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lukit batch evaluation CLI.")
    action_group = parser.add_argument_group("Actions")
    generation_group = parser.add_argument_group("Generation")
    dataset_group = parser.add_argument_group("Dataset")
    method_group = parser.add_argument_group("Methods")
    sampling_group = parser.add_argument_group("Sampling Family")
    semantic_graph_group = parser.add_argument_group("Semantic Graph Family")
    semantic_entropy_group = parser.add_argument_group("Semantic Entropy Family")
    judge_group = parser.add_argument_group("Judge")
    output_group = parser.add_argument_group("Output")

    action_group.add_argument("--list-methods", action="store_true", help="List all registered UQ methods.")

    generation_group.add_argument(
        "--gm_model_path",
        "--model_path",
        dest="gm_model_path",
        type=str,
        default="",
        help="Generation model path. Alias: --model_path",
    )
    generation_group.add_argument(
        "--gm_device",
        "--device",
        dest="gm_device",
        type=str,
        default="cuda:0",
        help="Generation model device. Alias: --device",
    )
    generation_group.add_argument("--torch_dtype", type=str, default="auto")
    generation_group.add_argument(
        "--chat_template_config",
        type=str,
        default="./configs/chat_template.json",
        help="Path to chat template config JSON for generation prompts.",
    )

    dataset_group.add_argument(
        "--dataset_name",
        type=str,
        default="trivia_qa_split",
        help=(
            "HF source: trivia_qa_split/simple_qa. "
            "JSONL source: chinese_simpleqa/hotpot_qa/nq_open/"
            "simpleqa_verified/triviaqa_validation/webqa/all(all scans dataset_dir/*.jsonl)"
        ),
    )
    dataset_group.add_argument(
        "--dataset_source",
        type=str,
        default="hf",
        choices=["hf", "jsonl"],
        help="Dataset source backend.",
    )
    dataset_group.add_argument(
        "--dataset_mode",
        type=str,
        default="original",
        choices=["original", "augment"],
        help="Only used when --dataset_source jsonl.",
    )
    dataset_group.add_argument(
        "--dataset_dir",
        type=str,
        default="./augmented_benchmark",
        help="Folder containing jsonl datasets when --dataset_source jsonl.",
    )
    dataset_group.add_argument("--start_idx", type=int, default=0)
    dataset_group.add_argument(
        "--num_samples_eval",
        type=int,
        default=100,
        help="Number of samples to evaluate. Use -1 to evaluate all remaining samples.",
    )

    method_group.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated method names, or 'all'.",
    )
    method_group.add_argument("--max_new_tokens", type=int, default=64)
    method_group.add_argument("--temperature", type=float, default=0.0)
    method_group.add_argument("--top_p", type=float, default=0.9)

    sampling_group.add_argument("--num_samples", type=int, default=4)
    sampling_group.add_argument("--sample_temperature", type=float, default=0.8)
    sampling_group.add_argument("--sample_top_p", type=float, default=0.9)
    sampling_group.add_argument("--lexical_metric", type=str, default="rougeL")
    sampling_group.add_argument("--p_true_with_context", action="store_true")
    semantic_graph_group.add_argument(
        "--semantic_similarity_score",
        type=str,
        default="nli",
        choices=["nli", "jaccard"],
        help="Shared similarity source for semantic graph methods.",
    )
    semantic_graph_group.add_argument(
        "--semantic_affinity",
        type=str,
        default="disagreement_w",
        choices=["disagreement_w", "agreement_w", "contra", "entail"],
        help="Shared NLI affinity mode for semantic graph methods; contra/entail are kept as aliases.",
    )
    semantic_graph_group.add_argument(
        "--semantic_temperature",
        type=float,
        default=3.0,
        help="Temperature used to turn NLI logits into probabilities for graph affinities.",
    )
    semantic_graph_group.add_argument(
        "--semantic_jaccard_threshold",
        type=float,
        default=0.5,
        help="Edge threshold used by num_sem_sets when semantic_similarity_score=jaccard.",
    )
    semantic_entropy_group.add_argument(
        "--semantic_class_source",
        type=str,
        default="nli",
        choices=["nli", "equivalence_judger"],
        help="Semantic class source used by semantic entropy methods.",
    )
    semantic_graph_group.add_argument(
        "--nli_model_path",
        type=str,
        default="",
        help="NLI model path used when semantic_similarity_score=nli.",
    )
    semantic_graph_group.add_argument(
        "--nli_device",
        type=str,
        default="cuda:2",
        help="Device for the NLI model when semantic_similarity_score=nli.",
    )
    semantic_graph_group.add_argument(
        "--nli_torch_dtype",
        type=str,
        default="auto",
        help="Torch dtype for the NLI model.",
    )
    semantic_entropy_group.add_argument(
        "--equivalence_judger_model_path",
        type=str,
        default="",
        help="Model path for the semantic equivalence judger used by semantic entropy.",
    )
    semantic_entropy_group.add_argument(
        "--equivalence_judger_device",
        type=str,
        default="cuda:0",
        help="Device for the semantic equivalence judger.",
    )
    semantic_entropy_group.add_argument(
        "--equivalence_judger_torch_dtype",
        type=str,
        default="auto",
        help="Torch dtype for the semantic equivalence judger.",
    )
    semantic_entropy_group.add_argument(
        "--equivalence_judger_max_new_tokens",
        type=int,
        default=16,
        help="Max new tokens for semantic equivalence judger decoding.",
    )

    judge_group.add_argument(
        "--jm_model_path",
        "--judge_model_path",
        dest="jm_model_path",
        type=str,
        default=_DEFAULT_JSON_JUDGE_MODEL_PATH,
        help="Judge model path. Alias: --judge_model_path",
    )
    judge_group.add_argument(
        "--jm_device",
        "--judge_device",
        dest="jm_device",
        type=str,
        default="cuda:1",
        help="Judge model device. Alias: --judge_device",
    )
    judge_group.add_argument("--judge_max_new_tokens", type=int, default=16)
    judge_group.add_argument(
        "--judge_mode",
        type=str,
        default="json",
        choices=["json", "chatglm"],
        help="Judge mode: json uses strict JSON prompt; chatglm uses reasoning-style prompt.",
    )
    output_group.add_argument("--out_jsonl", type=str, default="./results_lukit.jsonl")
    output_group.add_argument("--out_metrics", type=str, default="./metrics_lukit.json")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def load_dataset_split(dataset_name: str, start_idx: int, num_samples: int) -> List[Dict]:
    from datasets import load_dataset

    if dataset_name == "trivia_qa_split":
        split = load_dataset("trivia_qa", "rc.nocontext")["validation"]
        field_question = "question"
        field_answer = "answer"
    elif dataset_name == "simple_qa":
        split = load_dataset("basicv8vc/SimpleQA")["test"]
        field_question = "problem"
        field_answer = "answer"
    else:
        raise ValueError(
            f"Unsupported dataset_name: {dataset_name}. "
            "Supported: trivia_qa_split, simple_qa"
        )

    if num_samples < 0:
        end_idx = len(split)
    else:
        end_idx = min(start_idx + num_samples, len(split))
    records = []
    for idx in range(start_idx, end_idx):
        sample = split[idx]
        records.append(
            {
                "sample_idx": idx,
                "question": str(sample.get(field_question, "")).strip(),
                "answer": sample.get(field_answer, {}),
            }
        )
    return records


def _extract_answer_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("answer", "text", "value", "normalized_value", "gold", "label"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
        for key in ("answers", "aliases", "normalized_aliases"):
            inner = value.get(key)
            if isinstance(inner, list) and inner:
                for item in inner:
                    parsed = _extract_answer_text(item)
                    if parsed:
                        return parsed
    if isinstance(value, list):
        for item in value:
            parsed = _extract_answer_text(item)
            if parsed:
                return parsed
    return ""


def _extract_non_empty_str(sample: Dict, keys: List[str]) -> str:
    for key in keys:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_jsonl_record(sample: Dict, dataset_mode: str) -> Dict[str, str]:
    original_question = _extract_non_empty_str(
        sample,
        ["original_question", "question", "problem", "q"],
    )

    if dataset_mode == "augment":
        question = _extract_non_empty_str(
            sample,
            ["augmented_question", "rewrite_question", "question_aug", "question"],
        )
        if not question:
            questions = sample.get("questions", [])
            if isinstance(questions, list):
                best = ""
                for item in questions:
                    if isinstance(item, str) and item.strip():
                        item_clean = item.strip()
                        if original_question and item_clean != original_question:
                            best = item_clean
                            break
                        if not best:
                            best = item_clean
                question = best
        if not question:
            question = original_question

        answer = _extract_answer_text(
            sample.get("correct_answers")
            or sample.get("augmented_answer")
            or sample.get("answer")
            or sample.get("original_answer")
        )
    else:
        question = original_question
        answer = _extract_answer_text(
            sample.get("original_answer")
            or sample.get("answer")
            or sample.get("reference_answer")
            or sample.get("gold_answer")
        )

    if not question:
        question = _extract_non_empty_str(sample, ["question", "problem", "q"])
    if not answer:
        answer = _extract_answer_text(sample.get("answer"))
    return {"question": question, "answer": answer}


def load_jsonl_dataset(
    dataset_name: str,
    dataset_mode: str,
    dataset_dir: str,
    start_idx: int,
    num_samples: int,
) -> List[Dict]:
    selected_names: List[str]
    if dataset_name == "all":
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")
        selected_names = sorted(
            filename[: -len(".jsonl")]
            for filename in os.listdir(dataset_dir)
            if filename.endswith(".jsonl")
        )
        if not selected_names:
            raise FileNotFoundError(f"No .jsonl files found under dataset_dir: {dataset_dir}")
    elif dataset_name in _JSONL_DATASET_NAMES:
        selected_names = [dataset_name]
    else:
        raise ValueError(
            f"Unsupported dataset_name for jsonl source: {dataset_name}. "
            "Supported: chinese_simpleqa, hotpot_qa, nq_open, simpleqa_verified, "
            "triviaqa_validation, webqa, all"
        )

    records_all: List[Dict] = []
    for name in selected_names:
        path = os.path.join(dataset_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            if dataset_name == "all":
                continue
            raise FileNotFoundError(f"jsonl dataset file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL line in {path}:{line_no}: {exc}") from exc
                mapped = _build_jsonl_record(sample, dataset_mode)
                if not mapped["question"]:
                    continue
                records_all.append(
                    {
                        "question": mapped["question"],
                        "answer": mapped["answer"],
                        "dataset_name": name,
                    }
                )

    if not records_all:
        raise ValueError("No jsonl records loaded from dataset_dir.")

    if num_samples < 0:
        end_idx = len(records_all)
    else:
        end_idx = min(start_idx + num_samples, len(records_all))
    records = records_all[start_idx:end_idx]
    for idx, record in enumerate(records, start=start_idx):
        record["sample_idx"] = idx
    return records


def load_input_dataset(
    dataset_source: str,
    dataset_name: str,
    dataset_mode: str,
    dataset_dir: str,
    start_idx: int,
    num_samples: int,
) -> List[Dict]:
    if dataset_source == "hf":
        if dataset_name not in _HF_DATASET_NAMES:
            raise ValueError(
                f"Unsupported dataset_name for hf source: {dataset_name}. "
                "Supported: trivia_qa_split, simple_qa"
            )
        return load_dataset_split(dataset_name, start_idx, num_samples)

    return load_jsonl_dataset(
        dataset_name=dataset_name,
        dataset_mode=dataset_mode,
        dataset_dir=dataset_dir,
        start_idx=start_idx,
        num_samples=num_samples,
    )


def build_methods(config: EvalConfig):
    selected = config.selected_method_names()
    config.validate_for_methods(selected)
    return [create_method(name, **config.method_kwargs(name)) for name in selected]


def safe_float(value):
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return None


def compute_au_metrics(records: List[Dict], method_names: List[str]) -> Dict[str, Dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_error_all = [1 - int(r["correct"]) for r in records]
    summary: Dict[str, Dict] = {}
    for name in method_names:
        y_true = []
        y_score = []
        for rec, err in zip(records, y_error_all):
            method_obj = rec["u"].get(name)
            if isinstance(method_obj, dict):
                score = safe_float(method_obj.get("u"))
            else:
                score = safe_float(method_obj)
            if score is None:
                continue
            y_true.append(err)
            y_score.append(score)

        result = {
            "n_valid": len(y_true),
            "n_error": int(sum(y_true)),
            "n_correct": int(len(y_true) - sum(y_true)),
            "auroc": None,
            "auprc": None,
        }
        if len(y_true) >= 2 and len(set(y_true)) == 2:
            try:
                result["auroc"] = float(roc_auc_score(y_true, y_score))
            except Exception:
                result["auroc"] = None
            try:
                result["auprc"] = float(average_precision_score(y_true, y_score))
            except Exception:
                result["auprc"] = None
        summary[name] = result
    return summary


def build_judge_prompt(question: str, ground_truth: str, answer: str) -> List[Dict[str, str]]:
    system = (
        "You are a strict QA judge. Decide whether the candidate answer is factually correct "
        "given the question and ground-truth answer. "
        "Output ONLY JSON: {\"correct\": 1} or {\"correct\": 0}."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Ground Truth:\n{ground_truth}\n\n"
        f"Candidate Answer:\n{answer}\n\n"
        "Rules:\n"
        "1) Semantic equivalence counts as correct.\n"
        "2) Extra non-conflicting details are allowed.\n"
        "3) Contradiction with ground truth is incorrect.\n"
        "4) If core fact is missing, incorrect.\n\n"
        "Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_chatglm_judge_prompt(question: str, ground_truth: str, answer: str) -> str:
    return (
        "According to the reference answer, determine whether the student's answer is correct.\n\n"
        f"Question: {question}\n"
        f"Reference Answer: {ground_truth}\n"
        f"Student Answer: {answer}\n\n"
        "Analyze the reasoning and give your judgment.\n"
        "Output your final decision in the end as one word: Yes or No."
    )


def parse_judge_response(text: str) -> Optional[int]:
    text = text.strip()
    match = re.search(r"\{[\s\S]*?\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            value = obj.get("correct")
            if str(value) in {"0", "1"}:
                return int(value)
        except Exception:
            pass
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))
    lowered = text.lower()
    if re.search(r"\b(yes|correct)\b", lowered) and not re.search(r"\b(no|incorrect)\b", lowered):
        return 1
    if re.search(r"\b(no|incorrect)\b", lowered) and not re.search(r"\b(yes|correct)\b", lowered):
        return 0
    if "正确" in text and "错误" not in text:
        return 1
    if "错误" in text and "正确" not in text:
        return 0
    if "true" in lowered and "false" not in lowered:
        return 1
    if "false" in lowered and "true" not in lowered:
        return 0
    return None


def parse_chatglm_judge_response(text: str) -> Optional[int]:
    lowered = text.lower().strip()
    # Prefer explicit final decision sections.
    section = re.search(r"###\s*correct\s*([\s\S]+)$", lowered)
    if section:
        sec = section.group(1)
        if re.search(r"\b(yes|true)\b", sec) or "正确" in sec:
            return 1
        if re.search(r"\b(no|false)\b", sec) or "错误" in sec:
            return 0

    # Common decision phrases.
    if re.search(r"\b(final decision|judgment|answer)\s*[:：]\s*(yes|true)\b", lowered):
        return 1
    if re.search(r"\b(final decision|judgment|answer)\s*[:：]\s*(no|false)\b", lowered):
        return 0

    # Last non-empty line fallback.
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        if line in {"yes", "true", "correct"}:
            return 1
        if line in {"no", "false", "incorrect"}:
            return 0
    return None


class ChatGLMJudgeRunner:
    def __init__(self, config: JudgeConfig) -> None:
        if _CHATGLM_HTTP_PROXY:
            os.environ["http_proxy"] = _CHATGLM_HTTP_PROXY
        if _CHATGLM_HTTPS_PROXY:
            os.environ["https_proxy"] = _CHATGLM_HTTPS_PROXY
        if _CHATGLM_NO_PROXY:
            os.environ["no_proxy"] = _CHATGLM_NO_PROXY

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        dtype = _CHATGLM_JUDGE_TORCH_DTYPE.lower()
        torch_dtype = (
            getattr(torch, _DTYPE_MAP[dtype])
            if dtype in _DTYPE_MAP
            else _CHATGLM_JUDGE_TORCH_DTYPE
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.model.to(config.device)
        self.model.eval()
        self.max_new_tokens = config.max_new_tokens

    def judge(self, question: str, ground_truth: str, answer: str) -> str:
        prompt = build_chatglm_judge_prompt(
            question=question,
            ground_truth=ground_truth,
            answer=answer,
        )
        if hasattr(self.tokenizer, "build_chat_input"):
            inputs = self.tokenizer.build_chat_input(prompt, history=[])
            if isinstance(inputs, dict):
                inputs = {
                    key: value.to(self.model.device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
            else:
                raise TypeError("tokenizer.build_chat_input returned unsupported type.")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def heuristic_correct(record: Dict) -> int:
    pred_norm = normalize_text(str(record.get("a_model", "")))
    aliases = record.get("a_gold_aliases") or [record.get("a_gold", "")]
    alias_norms = [normalize_text(str(alias)) for alias in aliases if str(alias).strip()]
    for alias_norm in alias_norms:
        if not alias_norm:
            continue
        if alias_norm in pred_norm or pred_norm in alias_norm:
            return 1
    return 0


def annotate_correctness(
    records: List[Dict],
    judge_config: JudgeConfig,
    show_progress: bool = True,
) -> None:
    if not judge_config.model_path:
        record_iter = records
        if show_progress:
            record_iter = wrap_progress(
                record_iter,
                desc="Judge (heuristic)",
                total=len(records),
            )
        for record in record_iter:
            record["correct"] = heuristic_correct(record)
            record["judge_mode"] = "heuristic"
        return

    if judge_config.mode == "chatglm":
        judge_runner = ChatGLMJudgeRunner(judge_config)
        record_iter = records
        if show_progress:
            record_iter = wrap_progress(
                record_iter,
                desc="Judge (chatglm)",
                total=len(records),
            )
        for record in record_iter:
            raw = judge_runner.judge(
                question=record["q"],
                ground_truth=record["a_gold"],
                answer=record["a_model"],
            )
            parsed = parse_chatglm_judge_response(raw)
            if parsed is None:
                parsed = heuristic_correct(record)
            record["judge_raw"] = raw
            record["correct"] = int(parsed)
            record["judge_mode"] = "chatglm"
        return

    judge_backend = HFBackend(
        model=judge_config.model_path,
        device=judge_config.device,
        torch_dtype=_JSON_JUDGE_TORCH_DTYPE,
    )
    record_iter = records
    if show_progress:
        record_iter = wrap_progress(
            record_iter,
            desc="Judge (json)",
            total=len(records),
        )
    for record in record_iter:
        prompt = build_judge_prompt(
            question=record["q"],
            ground_truth=record["a_gold"],
            answer=record["a_model"],
        )
        raw = judge_backend.generate_from_messages(
            prompt,
            max_new_tokens=judge_config.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        parsed = parse_judge_response(raw)
        if parsed is None:
            parsed = heuristic_correct(record)
        record["judge_raw"] = raw
        record["correct"] = int(parsed)
        record["judge_mode"] = "json"


def save_jsonl(path: str, rows: List[Dict]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str, payload: Dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_methods() -> None:
    print("Registered methods:")
    for name in list_methods():
        cls = METHOD_REGISTRY[name]
        tags = ",".join(cls.tags) if getattr(cls, "tags", None) else "-"
        print(f"- {name} [{tags}]")


def main() -> None:
    args = parse_args()
    config = EvalConfig.from_args(args)
    if config.list_methods_flag:
        print_methods()
        return

    if not config.generation.model_path:
        raise ValueError("--gm_model_path is required unless --list-methods is used.")

    methods = build_methods(config)
    method_names = [method.name for method in methods]
    dataset = load_input_dataset(
        dataset_source=config.dataset.source,
        dataset_name=config.dataset.name,
        dataset_mode=config.dataset.mode,
        dataset_dir=config.dataset.directory,
        start_idx=config.dataset.start_idx,
        num_samples=config.dataset.num_samples_eval,
    )

    engine = ExecutionEngine(
        model=config.generation.model_path,
        backend_config=config.generation.backend_config(),
    )

    records = engine.run(
        dataset=dataset,
        methods=methods,
        **config.runtime_kwargs(),
        show_progress=True,
        progress_desc="Generation",
    )

    annotate_correctness(records, config.judge, show_progress=True)
    metrics = compute_au_metrics(records, method_names)

    save_jsonl(config.output.out_jsonl, records)
    save_json(
        config.output.out_metrics,
        {
            "dataset_source": config.dataset.source,
            "dataset_name": config.dataset.name,
            "dataset_mode": config.dataset.mode if config.dataset.source == "jsonl" else None,
            "dataset_dir": config.dataset.directory if config.dataset.source == "jsonl" else None,
            "start_idx": config.dataset.start_idx,
            "num_samples_eval": len(records),
            "methods": method_names,
            "metrics": metrics,
        },
    )

    print(
        json.dumps(
            {
                "n_records": len(records),
                "out_jsonl": config.output.out_jsonl,
                "out_metrics": config.output.out_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
