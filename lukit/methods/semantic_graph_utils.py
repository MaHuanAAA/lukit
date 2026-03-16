from typing import Any, Dict, List, Tuple

import numpy as np

_SEMANTIC_JACCARD_THRESHOLD = 0.5
_SEMANTIC_TEMPERATURE = 3.0

_AFFINITY_ALIASES = {
    "agreement_w": "agreement_w",
    "entail": "agreement_w",
    "entailment": "agreement_w",
    "disagreement_w": "disagreement_w",
    "contra": "disagreement_w",
    "contradiction": "disagreement_w",
}


def canonicalize_affinity(affinity: str) -> str:
    key = str(affinity).strip().lower()
    if key not in _AFFINITY_ALIASES:
        raise ValueError(f"Unsupported semantic affinity: {affinity}")
    return _AFFINITY_ALIASES[key]


def resolve_similarity_matrix(sampling_stats: Dict[str, Any]) -> np.ndarray:
    matrix = sampling_stats.get("similarity_matrix")
    if matrix is None:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def resolve_nli_logits(sampling_stats: Dict[str, Any]) -> np.ndarray:
    logits = sampling_stats.get("semantic_nli_logits")
    if logits is None:
        return np.zeros((0, 0, 3), dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 3 or logits.shape[-1] != 3:
        return np.zeros((0, 0, 3), dtype=np.float64)
    return logits


def resolve_semantic_matrix_entail(sampling_stats: Dict[str, Any]) -> np.ndarray:
    matrix = sampling_stats.get("semantic_matrix_entail")
    if matrix is None:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def resolve_semantic_matrix_contra(sampling_stats: Dict[str, Any]) -> np.ndarray:
    matrix = sampling_stats.get("semantic_matrix_contra")
    if matrix is None:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def get_sampled_texts(sampling_stats: Dict[str, Any]) -> List[str]:
    texts = sampling_stats.get("sampled_texts", [])
    if not isinstance(texts, list):
        return []
    return [str(text) for text in texts]


def normalize_exact_match_text(text: str) -> str:
    return str(text).strip().lower()


def deduplicate_sampled_texts(texts: List[str]) -> Tuple[List[str], List[int], List[int]]:
    normalized_texts: List[str] = []
    mapping: List[int] = []
    first_indices: List[int] = []
    seen: Dict[str, int] = {}

    for idx, text in enumerate(texts):
        normalized = normalize_exact_match_text(text)
        mapped = seen.get(normalized)
        if mapped is None:
            mapped = len(normalized_texts)
            seen[normalized] = mapped
            normalized_texts.append(normalized)
            first_indices.append(idx)
        mapping.append(mapped)
    return normalized_texts, mapping, first_indices


def nli_probs_from_logits(nli_logits: np.ndarray, temperature: float = _SEMANTIC_TEMPERATURE) -> np.ndarray:
    logits = np.asarray(nli_logits, dtype=np.float64)
    if logits.ndim != 3 or logits.shape[-1] != 3:
        return np.zeros((0, 0, 3), dtype=np.float64)

    temp = float(temperature)
    if temp <= 0:
        raise ValueError("semantic temperature must be > 0.")
    scaled = logits / temp
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(scaled)
    denom = np.sum(exp, axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return exp / denom


def build_affinity_matrix_from_logits(
    nli_logits: np.ndarray,
    affinity: str,
    temperature: float = _SEMANTIC_TEMPERATURE,
    symmetric: bool = True,
) -> np.ndarray:
    probs = nli_probs_from_logits(nli_logits, temperature=temperature)
    if probs.size == 0:
        return np.zeros((0, 0), dtype=np.float64)

    mode = canonicalize_affinity(affinity)
    if mode == "disagreement_w":
        matrix = 1.0 - probs[:, :, 0]
    else:
        matrix = probs[:, :, 2]

    if symmetric:
        matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    return np.asarray(matrix, dtype=np.float64)


def build_affinity_matrix(
    sampling_stats: Dict[str, Any],
    similarity_score: str,
    affinity: str,
    temperature: float = _SEMANTIC_TEMPERATURE,
    symmetric: bool = True,
) -> np.ndarray:
    texts = get_sampled_texts(sampling_stats)
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if n == 1:
        return np.eye(1, dtype=np.float64)

    score_name = str(similarity_score).strip().lower()
    if score_name == "jaccard":
        similarity = resolve_similarity_matrix(sampling_stats)
        if similarity.shape != (n, n):
            return np.zeros((0, 0), dtype=np.float64)
        if symmetric:
            similarity = (similarity + similarity.T) / 2.0
        np.fill_diagonal(similarity, 1.0)
        return similarity
    if score_name != "nli":
        raise ValueError(f"Unsupported semantic similarity score: {similarity_score}")

    logits = resolve_nli_logits(sampling_stats)
    if logits.shape != (n, n, 3):
        return np.zeros((0, 0), dtype=np.float64)
    return build_affinity_matrix_from_logits(
        logits,
        affinity=affinity,
        temperature=temperature,
        symmetric=symmetric,
    )


def build_num_sem_sets_adjacency(
    sampling_stats: Dict[str, Any],
    similarity_score: str,
    strict_entailment: bool = False,
    jaccard_threshold: float = _SEMANTIC_JACCARD_THRESHOLD,
) -> np.ndarray:
    texts = get_sampled_texts(sampling_stats)
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64)

    score_name = str(similarity_score).strip().lower()
    if score_name == "nli":
        entail = resolve_semantic_matrix_entail(sampling_stats)
        contra = resolve_semantic_matrix_contra(sampling_stats)
        if entail.shape != (n, n) or contra.shape != (n, n):
            return np.zeros((0, 0), dtype=np.int64)

        if strict_entailment:
            logits = resolve_nli_logits(sampling_stats)
            if logits.shape != (n, n, 3):
                return np.zeros((0, 0), dtype=np.int64)
            relation = np.argmax(logits, axis=-1)
            adjacency = np.zeros((n, n), dtype=np.int64)
            for i in range(n):
                for j in range(i + 1, n):
                    equivalent = int(relation[i, j]) == 2 and int(relation[j, i]) == 2
                    if equivalent:
                        adjacency[i, j] = 1
                        adjacency[j, i] = 1
            return adjacency

        adjacency = (entail > contra).astype(np.int64)
        adjacency = adjacency * adjacency.T
        adjacency = np.triu(adjacency, k=1)
        adjacency = adjacency + adjacency.T
        np.fill_diagonal(adjacency, 0)
        return adjacency

    similarity = resolve_similarity_matrix(sampling_stats)
    if similarity.shape != (n, n):
        return np.zeros((0, 0), dtype=np.int64)
    adjacency = (similarity >= float(jaccard_threshold)).astype(np.int64)
    adjacency = np.triu(adjacency, k=1)
    adjacency = adjacency + adjacency.T
    np.fill_diagonal(adjacency, 0)
    return adjacency


def connected_components_count(adjacency: np.ndarray) -> int:
    n = int(adjacency.shape[0])
    if n == 0:
        return 0

    visited = np.zeros((n,), dtype=bool)
    count = 0
    for start in range(n):
        if visited[start]:
            continue
        count += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            neighbors = np.where(adjacency[node] != 0)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
    return count


def normalized_graph_laplacian(similarity: np.ndarray) -> np.ndarray:
    similarity = np.asarray(similarity, dtype=np.float64)
    n = int(similarity.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    degree = np.sum(similarity, axis=1)
    degree_inv_sqrt = np.zeros((n,), dtype=np.float64)
    non_zero = degree > 1e-12
    degree_inv_sqrt[non_zero] = 1.0 / np.sqrt(degree[non_zero])
    degree_inv_sqrt = np.diag(degree_inv_sqrt)
    degree_matrix = np.diag(degree)
    return degree_inv_sqrt @ (degree_matrix - similarity) @ degree_inv_sqrt
