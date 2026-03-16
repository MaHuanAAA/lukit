from typing import Any, Callable, Dict, List

import numpy as np

from .semantic_graph_utils import get_sampled_texts, resolve_nli_logits


def empty_semantic_classes() -> Dict[str, List]:
    return {
        "sample_to_class": [],
        "class_to_sample": [],
    }


def build_semantic_classes_from_mutual_entailment(
    sampling_stats: Dict[str, Any],
) -> Dict[str, List]:
    texts = get_sampled_texts(sampling_stats)
    n = len(texts)
    if n == 0:
        return empty_semantic_classes()

    logits = resolve_nli_logits(sampling_stats)
    if logits.shape != (n, n, 3):
        return empty_semantic_classes()

    relation = np.argmax(logits, axis=-1)
    sample_to_class = [-1] * n
    class_to_sample: List[List[int]] = []

    for i in range(n):
        if sample_to_class[i] != -1:
            continue
        class_id = len(class_to_sample)
        sample_to_class[i] = class_id
        members = [i]
        for j in range(i + 1, n):
            if sample_to_class[j] != -1:
                continue
            if int(relation[i, j]) == 2 and int(relation[j, i]) == 2:
                sample_to_class[j] = class_id
                members.append(j)
        class_to_sample.append(members)

    return {
        "sample_to_class": sample_to_class,
        "class_to_sample": class_to_sample,
    }


def build_semantic_classes_from_incremental_judger(
    sampled_texts: List[str],
    is_equivalent: Callable[[int, int], bool],
) -> Dict[str, List]:
    n = len(sampled_texts)
    if n == 0:
        return empty_semantic_classes()

    sample_to_class = [-1] * n
    class_to_sample: List[List[int]] = []

    for i in range(n):
        if sample_to_class[i] != -1:
            continue
        class_id = len(class_to_sample)
        sample_to_class[i] = class_id
        members = [i]
        for j in range(i + 1, n):
            if sample_to_class[j] != -1:
                continue
            if is_equivalent(i, j):
                sample_to_class[j] = class_id
                members.append(j)
        class_to_sample.append(members)

    return {
        "sample_to_class": sample_to_class,
        "class_to_sample": class_to_sample,
    }


def semantic_entropy_from_logprobs(
    sampling_stats: Dict[str, Any],
) -> float:
    semantic_classes = sampling_stats.get("semantic_classes") or {}
    sample_to_class = semantic_classes.get("sample_to_class") or []
    class_to_sample = semantic_classes.get("class_to_sample") or []
    logprobs = np.asarray(
        sampling_stats.get("sampled_sequence_logprobs", []),
        dtype=np.float64,
    )

    if not sample_to_class or not class_to_sample or logprobs.size == 0:
        return 0.0
    if len(sample_to_class) != logprobs.shape[0]:
        return 0.0

    class_logprobs = []
    for sample_indices in class_to_sample:
        if not sample_indices:
            continue
        class_logprobs.append(np.logaddexp.reduce(logprobs[np.asarray(sample_indices, dtype=np.int64)]))
    if len(class_logprobs) != len(class_to_sample):
        return 0.0

    per_sample_class_logprobs = np.asarray(
        [class_logprobs[int(class_id)] for class_id in sample_to_class],
        dtype=np.float64,
    )
    return float(-np.mean(per_sample_class_logprobs))


def semantic_entropy_empirical(
    sampling_stats: Dict[str, Any],
) -> float:
    semantic_classes = sampling_stats.get("semantic_classes") or {}
    sample_to_class = semantic_classes.get("sample_to_class") or []
    class_to_sample = semantic_classes.get("class_to_sample") or []

    n = len(sample_to_class)
    if n == 0 or not class_to_sample:
        return 0.0

    probs = np.asarray([len(indices) / float(n) for indices in class_to_sample], dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())
