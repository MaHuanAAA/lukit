# Engine Module

The `engine` module contains the core execution engine that orchestrates model inference and uncertainty computation.

## Overview

The execution engine is responsible for:

- Loading and managing model backends
- Coordinating data providers
- Running single-sample and batch evaluations
- Managing execution dependencies

## Components

### ExecutionEngine

Main class for running uncertainty evaluations.

#### Key Methods

- `run_single(prompt: str, method: UncertaintyMethod, **kwargs) -> Dict`
  - Run evaluation on a single prompt
  - Returns a record with generation and uncertainty scores

- `run(dataset: List[Dict], methods: List[UncertaintyMethod], **kwargs) -> List[Dict]`
  - Run batch evaluation on a dataset
  - Returns list of result records

#### Constructor Parameters

```python
ExecutionEngine(
    model: str,                          # Model path/name
    backend_config: Dict = None,         # Backend configuration
    tokenizer: Any = None,               # Optional pre-loaded tokenizer
)
```

## Usage Examples

### Single Prompt Evaluation

```python
from lukit.engine import ExecutionEngine
from lukit.methods import create_method

engine = ExecutionEngine(
    model="meta-llama/Llama-2-7b-hf",
    backend_config={"device": "cuda:0"}
)

method = create_method("p_true")
record = engine.run_single(
    prompt="What is the capital of France?",
    method=method,
    max_new_tokens=32
)

print(f"Answer: {record['a_model']}")
print(f"Uncertainty: {record['u']['p_true']['u']}")
```

### Batch Evaluation

```python
dataset = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "Capital of Japan?", "answer": "Tokyo"},
]

methods = [
    create_method("p_true"),
    create_method("mean_token_entropy"),
]

records = engine.run(
    dataset=dataset,
    methods=methods,
    max_new_tokens=64
)
```

## Record Format

Each result record contains:

```python
{
    "q": "Original question/prompt",
    "a_gold": "Ground truth answer",
    "a_model": "Model's answer",
    "u": {
        "method_name": {
            "u": float,  # Uncertainty score
        },
        ...
    },
    "generation_stats": {
        "num_tokens": int,
        "completion": str,
    }
}
```
