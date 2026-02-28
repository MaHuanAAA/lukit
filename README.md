# lukit

`lukit` (LLM Uncertainty Kit) 用于评估 LLM 回答的不确定性（UQ），支持批量评测与判别（judge）。

## 1. 安装

```bash
conda create -n lukit python=3.10
conda activate lukit
pip install -r requirements.txt
```

## 2. 当前支持的方法

- `sequence_log_probability`
- `perplexity`
- `mean_token_entropy`
- `self_certainty`
- `monte_carlo_sequence_entropy`
- `lexical_similarity`
- `p_true`

查看当前注册方法：

```bash
python scripts/run_lukit.py --list-methods
```

## 3. 快速开始（CLI）

运行时会自动显示两个进度条：`Generation`（生成+UQ）和 `Judge`（正确性判别）。

全参数命令（显式，Qwen judge）：

```bash
python scripts/run_lukit.py \
  --model_path /data1/chenjingdong/ms/meta-llama__Llama-3.1-8B-Instruct \
  --device cuda:4 \
  --torch_dtype auto \
  --chat_template_config ./configs/chat_template.json \
  --dataset_source jsonl \
  --dataset_dir ./augmented_benchmark \
  --dataset_mode original \
  --dataset_name all \
  --start_idx 0 \
  --num_samples_eval 20 \
  --methods all \
  --max_new_tokens 64 \
  --temperature 0.0 \
  --top_p 0.9 \
  --num_samples 4 \
  --sample_temperature 0.8 \
  --sample_top_p 0.9 \
  --lexical_metric rougeL \
  --judge_model_path /data1/chenjingdong/ms/Qwen__Qwen3-4B-Instruct-2507 \
  --judge_device cuda:5 \
  --judge_max_new_tokens 16 \
  --judge_mode json \
  --out_jsonl ./lukit_judged_qwen.jsonl \
  --out_metrics ./lukit_judged_qwen_metrics.json
```

## 4. 参数说明

可选开关（默认关闭）：

- `--p_true_with_context`：启用 `p_true` 的上下文判别版本。

独立命令（只列方法，不跑评测）：

```bash
python scripts/run_lukit.py --list-methods
```

参数逐一说明：

- `--list-methods`：列出当前注册方法并退出。
- `--model_path`：被评测生成模型路径（必填，除非使用 `--list-methods`）。
- `--device`：生成模型设备；可选如 `cuda:0`、`cuda:1`、`cpu`。
- `--torch_dtype`：生成模型 dtype；可选 `auto`、`float16`、`bfloat16`、`float32`。
- `--chat_template_config`：生成 prompt 模板 JSON 路径（默认 `./configs/chat_template.json`）。
- `--dataset_source`：数据源；可选 `hf` 或 `jsonl`。
- `--dataset_dir`：本地 JSONL 目录（仅 `jsonl` 模式生效）。
- `--dataset_mode`：JSONL 字段选择；可选 `original` 或 `augment`（仅 `jsonl` 生效）。
- `--dataset_name`：数据集名称。
- `--dataset_name`（`hf`）：`trivia_qa_split`、`simple_qa`。
- `--dataset_name`（`jsonl`）：`chinese_simpleqa`、`hotpot_qa`、`nq_open`、`simpleqa_verified`、`triviaqa_validation`、`webqa`、`all`（扫描目录下全部 `.jsonl`）。
- `--start_idx`：从第几个样本开始评测。
- `--num_samples_eval`：本次评测样本数。
- `--methods`：运行的方法；`all` 或逗号分隔列表（如 `p_true,mean_token_entropy`）。
- `--max_new_tokens`：主模型单次生成的最大新 token 数。
- `--temperature`：主模型生成温度；`0.0` 表示贪心解码。
- `--top_p`：主模型 nucleus 采样阈值；通常与 `temperature > 0` 一起使用。
- `--num_samples`：采样类方法的采样次数（如 lexical/sampling 类方法）。
- `--sample_temperature`：采样类方法的温度。
- `--sample_top_p`：采样类方法的 top-p。
- `--lexical_metric`：`lexical_similarity` 使用的相似度指标；常用 `rougeL`、`rouge1`、`rouge2`、`BLEU`。
- `--p_true_with_context`：布尔开关；写上即启用，不写即关闭。
- `--judge_model_path`：judge 模型路径（你当前使用 Qwen：`/data1/chenjingdong/ms/Qwen__Qwen3-4B-Instruct-2507`）。
- `--judge_device`：judge 模型设备；可与生成模型分卡运行。
- `--judge_max_new_tokens`：judge 输出最大 token 数。
- `--judge_mode`：judge 模式；`json`（JSON 优先解析）或 `chatglm`（ChatGLM 风格解析）。
- `--out_jsonl`：逐样本输出文件路径。
- `--out_metrics`：汇总指标输出文件路径。

## 5. 生成 Prompt 模板配置

生成阶段使用独立配置文件（默认）：`./configs/chat_template.json`

格式如下：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant. Give a concise factual answer."},
    {"role": "user", "content": "{question}"}
  ]
}
```

其中 `{question}` 会在运行时替换为样本问题。

如果要指定其他模板文件：

```bash
python scripts/run_lukit.py \
  --model_path /path/to/model \
  --chat_template_config /path/to/chat_template.json \
  --methods all
```

## 6. 输出文件

- `--out_jsonl`：逐样本结果，包含 `q`、`a_gold`、`a_model`、`u`、`correct`、`judge_mode` 等。
- `--out_metrics`：汇总指标，包含每个方法的 `n_valid`、`n_error`、`n_correct`、`auroc`、`auprc`。

## 7. Python API 示例

`scripts/test_api.py` 是可直接运行的最小示例：

```bash
python scripts/test_api.py
```

核心调用方式：

```python
from lukit.engine import ExecutionEngine
from lukit.methods import create_method

engine = ExecutionEngine(
    model="/data1/chenjingdong/ms/meta-llama__Llama-3.1-8B-Instruct",
    backend_config={"type": "huggingface", "device": "cuda:4"},
)
method = create_method("p_true")
record = engine.run_single(prompt="What is 2+2?", method=method, max_new_tokens=32)

print(record["a_model"])
print(record["u"]["p_true"])
```

## 8. 目录

```text
lukit/
├── backends/                  # 模型后端实现（当前是 HuggingFace）
├── data_providers/            # 声明式数据提供者（如 logprob/sampling）
├── methods/                   # 各种 UQ 方法实现与注册
├── engine/                    # 调度与执行主流程
├── cli/                       # 命令行参数与主入口逻辑
├── configs/chat_template.json # 生成 prompt 模板配置
└── scripts/
    ├── run_lukit.py           # CLI 入口脚本
    └── test_api.py            # Python API 最小示例
```

## 9. 如何新增一个方法

新增方法推荐按下面 4 步：

1. 在 `methods/` 新建方法文件（例如 `methods/my_method.py`），继承 `UncertaintyMethod`。
2. 声明方法名和依赖数据（`name`、`requires_data`），实现 `_compute(stats)` 并返回 `{"u": float}`。
3. 在 `methods/__init__.py` 的 `_METHOD_MODULES` 中注册模块名（例如 `"my_method"`）。
4. 用 CLI 做最小验证：`--methods my_method --num_samples_eval 2`。

最小模板：

```python
import numpy as np
from .base_method import UncertaintyMethod


class MyMethod(UncertaintyMethod):
    name = "my_method"
    requires_data = ["logprob_stats"]
    tags = ["intrinsic"]

    def _compute(self, stats):
        x = np.asarray(stats["logprob_stats"]["completion_log_probs"], dtype=np.float64)
        if x.size == 0:
            return {"u": 0.0}
        return {"u": float(-x.mean())}
```

如果现有 `stats` 不够用，再在 `data_providers/` 新增 provider，并在 `data_providers/__init__.py` 注册。
