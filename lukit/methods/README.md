# Methods Module

The `methods` module contains implementations of various uncertainty quantification (UQ) methods for LLM responses.

## Overview

Each uncertainty method is implemented as a class that inherits from `UncertaintyMethod`. The module provides:

- A base class `UncertaintyMethod` for defining new methods
- A registry for tracking all available methods
- A factory function for creating method instances

## Available Methods

| Method | Description | Type | Requires Sampling |
|--------|-------------|------|-------------------|
| `sequence_log_probability` | Log probability of the generated sequence | Intrinsic | No |
| `perplexity` | Perplexity of the generated text | Intrinsic | No |
| `mean_token_entropy` | Average entropy across tokens | Intrinsic | No |
| `self_certainty` | Self-certainty score | Intrinsic | No |
| `monte_carlo_sequence_entropy` | Monte Carlo estimate of sequence entropy | Sampling | Yes |
| `semantic_entropy` | Likelihood-weighted semantic entropy over semantic classes | Sampling | Yes |
| `semantic_entropy_empirical` | Empirical semantic entropy over semantic classes | Sampling | Yes |
| `lexical_similarity` | Similarity between sampled responses | Sampling | Yes |
| `p_true` | Probability that answer is true | Intrinsic | No |
| `eigenscore` | EigenScore-based uncertainty | Representation | Yes |
| `deg_mat` | Degree-matrix score over sampled answer graph | Graph | Yes |
| `eccentricity` | Spectral embedding dispersion over sampled answer graph | Graph | Yes |
| `eig_val_laplacian` | Sum of graph-Laplacian eigenvalue contributions | Graph | Yes |
| `num_sem_sets` | Number of semantic connected components | Graph | Yes |

## Adding a New Method

To add a new uncertainty method:

1. Create a new Python file in the `methods/` directory
2. Define a class that inherits from `UncertaintyMethod`
3. Set the class attributes: `name`, `requires_data`, and optionally `tags`
4. Implement the `_compute(self, stats: Dict[str, Any]) -> Dict[str, Any]` method
5. Add your module name to `_METHOD_MODULES` in `methods/__init__.py`

### Example

```python
from .base_method import UncertaintyMethod

class MyMethod(UncertaintyMethod):
    name = "my_method"
    requires_data = ["logprob_stats"]
    tags = ["intrinsic"]

    def _compute(self, stats):
        x = stats["logprob_stats"]["completion_log_probs"]
        if not x:
            return {"u": 0.0}
        return {"u": float(-sum(x) / len(x))}
```

## API Reference

### UncertaintyMethod (Base Class)

- `name: str` - Unique identifier for the method
- `requires_data: List[str]` - Data providers needed by this method
- `supported_backends: List[str]` - Compatible backends
- `tags: List[str]` - Method categorization tags

#### Methods

- `_compute(stats: Dict) -> Dict` - Core computation, must return a dict with `u` key
- `compute(stats: Dict) -> Dict` - Wrapper that validates and normalizes output
- `__call__(model, tokenizer, prompt, **kwargs) -> Dict` - Direct evaluation interface

### Registry Functions

- `list_methods() -> List[str]` - Get list of all registered method names
- `create_method(name: str, **kwargs) -> UncertaintyMethod` - Create a method instance
- `METHOD_REGISTRY: Dict[str, Type[UncertaintyMethod]]` - Global registry of methods
