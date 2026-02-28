from difflib import SequenceMatcher

import numpy as np

from .base_method import UncertaintyMethod

try:
    from nltk.translate.bleu_score import sentence_bleu

    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

try:
    from rouge_score import rouge_scorer

    _HAS_ROUGE = True
except Exception:
    _HAS_ROUGE = False


class LexicalSimilarity(UncertaintyMethod):
    name = "lexical_similarity"
    requires_data = ["sampling_stats"]
    tags = ["sampling", "semantic"]

    def __init__(self, metric: str = "rougeL") -> None:
        self.metric = metric
        if metric.startswith("rouge") and _HAS_ROUGE:
            self._scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
        else:
            self._scorer = None

    def _pairwise_similarity(self, text_a: str, text_b: str) -> float:
        if self.metric.startswith("rouge") and self._scorer is not None:
            return float(self._scorer.score(text_a, text_b)[self.metric].fmeasure)
        if self.metric.upper() == "BLEU" and _HAS_NLTK:
            tokens_a = text_a.split()
            tokens_b = text_b.split()
            if not tokens_a or not tokens_b:
                return 0.0
            min_len = min(len(tokens_a), len(tokens_b))
            if min_len <= 1:
                weights = [1.0, 0.0, 0.0, 0.0]
            elif min_len == 2:
                weights = [0.5, 0.5, 0.0, 0.0]
            elif min_len == 3:
                weights = [0.33, 0.33, 0.33, 0.0]
            else:
                weights = [0.25, 0.25, 0.25, 0.25]
            return float(sentence_bleu([tokens_a], tokens_b, weights=weights))
        return float(SequenceMatcher(None, text_a, text_b).ratio())

    def _compute(self, stats):
        texts = stats["sampling_stats"]["sampled_texts"]
        if len(texts) < 2:
            return {"u": 0.0}

        sims = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims.append(self._pairwise_similarity(texts[i], texts[j]))
        return {"u": float(-np.mean(sims))}
