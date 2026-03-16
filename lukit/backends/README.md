# Backends Module

The `backends` module provides model backend implementations for different model serving frameworks.

## Overview

The backend system abstracts model loading and inference, allowing LUKIT to work with various model frameworks through a unified interface.

## Available Backends

### HFBackend (HuggingFace)

The HuggingFace backend provides support for:
- AutoModelForCausalLM models from HuggingFace Hub
- Local model checkpoints
- Configurable device placement (CPU/GPU)
- Automatic dtype handling

### BaseBackend

Abstract base class that defines the interface all backends must implement.

## API Reference

### BaseBackend

Abstract base class with the following methods:

- `__init__(model: str, device: str = "cuda:0", torch_dtype: str = "auto", **kwargs)`
- `load_model()` - Load the model and tokenizer
- `generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str`
- `generate_from_messages(messages: List[Dict], **kwargs) -> str`
- `get_log_probs(prompt: str, completion: str) -> List[float]`
- `get_hidden_states(prompt: str, completion: str) -> np.ndarray`

### HFBackend

HuggingFace-specific implementation with additional features:

- `chat_template_config: Optional[str]` - Path to chat template JSON
- Automatic chat template application
- Support for model-specific tokenization

## Configuration

### Backend Config Format

```python
{
    "type": "huggingface",
    "device": "cuda:0",
    "torch_dtype": "auto",
    "chat_template_config": "./configs/chat_template.json"
}
```

### Supported dtypes

- `"auto"` - Automatic selection based on model
- `"float16"` - FP16 (recommended for GPU inference)
- `"bfloat16"` - BF16 (recommended for newer GPUs)
- `"float32"` - FP32 (full precision)

## Adding a New Backend

To add support for a new backend:

1. Create a new file in `backends/` directory
2. Define a class that inherits from `BaseBackend`
3. Implement all required methods
4. Add your backend to the factory in `engine/dependency_resolver.py`

### Example

```python
from .base_backend import BaseBackend

class CustomBackend(BaseBackend):
    def __init__(self, model: str, device: str = "cuda:0", **kwargs):
        super().__init__(model, device, **kwargs)
        # Custom initialization

    def load_model(self):
        # Implement model loading
        pass

    # Implement other required methods...
```
