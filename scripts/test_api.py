import sys
import os


sys.path.append("/data1/kongxinke")
from lukit.engine import ExecutionEngine
from lukit.methods import create_method

engine = ExecutionEngine(
    model="/data1/chenjingdong/ms/meta-llama__Llama-3.1-8B-Instruct",
    backend_config={"type": "huggingface", "device": "cuda:4"},
)
method = create_method("p_true")
record = engine.run_single(prompt="What is 2+2?", method=method, max_new_tokens=32)

print("a_model:", record["a_model"])
print("p_true_result:", record["u"]["p_true"])
