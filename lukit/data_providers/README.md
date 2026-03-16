# Data Providers Module

The `data_providers` module provides declarative data providers for extracting various types of data from model inference.

## Overview

Data providers are modular components that extract specific types of information during model inference. They work together to provide the statistics needed by different uncertainty methods.

## Available Providers

| Provider Key | Description | Built-in Methods Using It |
|--------------|-------------|---------------------------|
| `logprob_stats` | Token log-probabilities, token entropies, vocab statistics | `sequence_log_probability`, `perplexity`, `mean_token_entropy`, `self_certainty` |
| `sampling_stats` | Sampled texts, sampled NLLs, exact sampled sequence logprobs, optional EigenScore embeddings, optional semantic graph matrices/classes | `monte_carlo_sequence_entropy`, `semantic_entropy`, `semantic_entropy_empirical`, `lexical_similarity`, `eigenscore`, `deg_mat`, `eccentricity`, `eig_val_laplacian`, `num_sem_sets` |
| `p_true_prob` | Binary True/False probability from the backend | `p_true` |
| `prompt_text` | Rendered prompt string | Reserved extension point |
| `generation_artifacts` | Prompt/completion ids and generated text | Reserved extension point |
| `hidden_states` | Pass-through hidden states entry from model outputs | Registered but not used by built-in methods |

## Provider Architecture

### DataProvider

Abstract base class defining the provider interface:

- `provides: str` - The stats key produced by this provider
- `requires_from_model: List[str]` - Fields expected inside `model_outputs`
- `__call__(model_outputs, **kwargs)` - Extract the provider payload

## Data Flow

```
Prompt -> Backend -> Provider(s) -> Statistics -> Method -> Uncertainty Score
```

## Using Data Providers

### In a Method

```python
class MyMethod(UncertaintyMethod):
    name = "my_method"
    requires_data = ["logprob_stats"]

    def _compute(self, stats):
        logprobs = stats["logprob_stats"]["completion_log_probs"]
        return {"u": float(-sum(logprobs) / len(logprobs))}
```

## Adding a New Provider

1. Create a new file in `data_providers/`
2. Inherit from `BaseProvider`
3. Define provider attributes
4. Implement extraction logic
5. Register in `data_providers/__init__.py`

### Example

```python
from .base_provider import DataProvider

class CustomProvider(DataProvider):
    provides = "custom_stats"
    requires_from_model = ["answer_text"]

    def __call__(self, model_outputs, **kwargs):
        return {"answer": model_outputs["answer_text"]}
```

## Stats Dictionary Structure

The stats dictionary passed to methods contains:

```python
{
    "logprob_stats": {
        "completion_log_probs": [...],
        "token_entropies": [...],
        "mean_logp_vocab": [...],
        "vocab_size": 32000,
    },
    "sampling_stats": {
        "sampled_texts": ["text1", "text2", ...],
        "sampled_sequence_nlls": [...],
        "sampled_sequence_logprobs": [...],
        "eigenscore_embeddings": [...],
        "similarity_matrix": [...],
        "semantic_nli_logits": [...],
        "semantic_matrix_entail": [...],
        "semantic_matrix_contra": [...],
        "semantic_classes": {
            "sample_to_class": [...],
            "class_to_sample": [...],
        },
    },
    "p_true_prob": 0.87
}
```
