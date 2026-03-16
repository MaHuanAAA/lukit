# lukit

`lukit` (LLM Uncertainty Kit) 是一个用于评估 LLM 回答不确定性（UQ）的 Python 工具包，支持批量评测与判别（judge），并提供论文写作所需的 LaTeX 表格和 matplotlib 图片生成功能。

## 特性

- **多方法支持**: 内置 14 种不确定性评估方法
- **批量评测**: 支持 HuggingFace 和本地 JSONL 数据集
- **自动判别**: 支持模型判别和启发式判别
- **可视化输出**: 一键生成论文级 LaTeX 表格和 matplotlib 图片
- **榜单展示**: 清晰展示各方法性能对比
- **易扩展**: 简单的插件式方法注册机制

## 安装

### 从源码安装

```bash
git clone https://github.com/baidu/lukit.git
cd lukit
pip install -e .
```

### 依赖要求

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0

## 快速开始

### 1. 列出可用方法

```bash
lukit list-methods
```

输出:
```
Registered methods:
- deg_mat [sampling,semantic,graph]
- eccentricity [sampling,semantic,graph,spectral]
- eig_val_laplacian [sampling,semantic,graph,spectral]
- eigenscore [sampling,representation]
- lexical_similarity [sampling,semantic]
- mean_token_entropy [intrinsic]
- monte_carlo_sequence_entropy [sampling]
- num_sem_sets [sampling,semantic,graph,discrete]
- p_true [interaction]
- perplexity [intrinsic]
- self_certainty [intrinsic]
- semantic_entropy [sampling,semantic,entropy]
- semantic_entropy_empirical [sampling,semantic,entropy]
- sequence_log_probability [intrinsic]
```

### 2. 运行评测

```bash
lukit eval \
  --gm_model_path /path/to/model \
  --gm_device cuda:0 \
  --dataset_name trivia_qa_split \
  --methods all \
  --num_samples 4 \
  --nli_model_path /path/to/nli-model \
  --nli_device cuda:1 \
  --num_samples_eval 100 \
  --out_metrics ./metrics.json
```

等价的独立命令仍然可用:

```bash
lukit-eval --gm_model_path /path/to/model --dataset_name trivia_qa_split --methods all
```

完整的命令行参数可以通过以下命令查看：

```bash
lukit eval --help
```

`lukit eval` 的参数现在按用途分组展示，便于区分通用参数和方法族参数：

- `Generation`: 生成模型与解码参数，如 `--gm_model_path`、`--gm_device`、`--max_new_tokens`
- `Dataset`: 数据来源与采样范围，如 `--dataset_source`、`--dataset_name`、`--num_samples_eval`
- `Methods`: 方法选择与通用生成控制，如 `--methods`、`--temperature`、`--top_p`
- `Sampling Family`: 所有依赖多次采样的方法共享，如 `--num_samples`、`--sample_temperature`、`--lexical_metric`、`--p_true_with_context`
- `Semantic Graph Family`: `deg_mat`、`eccentricity`、`eig_val_laplacian`、`num_sem_sets` 共享，如 `--semantic_similarity_score`、`--semantic_affinity`、`--nli_model_path`
- `Semantic Entropy Family`: `semantic_entropy`、`semantic_entropy_empirical` 共享，如 `--semantic_class_source`、`--equivalence_judger_model_path`
- `Judge`: 正确性判别相关，如 `--jm_model_path`、`--judge_mode`
- `Output`: 输出路径，如 `--out_jsonl`、`--out_metrics`

常见约束：

- `--num_samples_eval -1` 表示从 `--start_idx` 开始评估剩余全部样本
- 使用 semantic graph 方法且 `--semantic_similarity_score nli` 时，需要提供 `--nli_model_path`
- 使用 semantic entropy 方法且 `--semantic_class_source nli` 时，需要提供 `--nli_model_path`
- 使用 semantic entropy 方法且 `--semantic_class_source equivalence_judger` 时，需要提供 `--equivalence_judger_model_path`

### 3. 查看榜单

```bash
lukit-leaderboard ./metrics.json
```

输出示例:
```
============================================================
LUKIT UNCERTAINTY METHOD LEADERBOARD
============================================================
Dataset: trivia_qa_split
Source: hf
Samples: 100
============================================================

+------+------------------------+--------+--------+----------+-------+---------+--------+
| Rank | Method                 | AUROC  | AUPRC  | Accuracy | Valid | Correct | Errors |
+------+------------------------+--------+--------+----------+-------+---------+--------+
| #1   | * p_true               | 0.8523 | 0.7891 | 0.6633   | 98    | 65      | 33     |
| #2   | perplexity             | 0.8102 | 0.7543 | 0.6633   | 98    | 65      | 33     |
| #3   | mean_token_entropy     | 0.7856 | 0.7234 | 0.6633   | 98    | 65      | 33     |
+------+------------------------+--------+--------+----------+-------+---------+--------+
```

### 4. 生成论文图表

```bash
lukit-visualize \
  --metrics ./metrics.json \
  --output_dir ./paper_assets \
  --table_type both \
  --plot_type all \
  --plot_format pdf
```

生成的文件:
- `./paper_assets/uq_results.tex` - LaTeX 表格
- `./paper_assets/uq_full_metrics.tex` - 完整指标 LaTeX 表格
- `./paper_assets/uq_bar_plot_auroc.pdf` - 柱状图
- `./paper_assets/uq_multi_plot.pdf` - 多指标对比图

## Python API

### 基本使用

```python
from lukit import ExecutionEngine, create_method

# 创建引擎
engine = ExecutionEngine(
    model="/path/to/model",
    backend_config={"device": "cuda:0"}
)

# 创建方法
method = create_method("p_true")

# 单样本评测
record = engine.run_single(
    prompt="What is 2+2?",
    method=method,
    max_new_tokens=32
)

print(f"Answer: {record['a_model']}")
print(f"Uncertainty: {record['u']['p_true']['u']}")
```

### 批量评测

```python
from lukit import ExecutionEngine, create_method, list_methods

# 准备数据
dataset = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "Capital of Japan?", "answer": "Tokyo"},
]

# 创建所有方法
methods = [create_method(name) for name in list_methods()]

# 运行评测
records = engine.run(
    dataset=dataset,
    methods=methods,
    max_new_tokens=64
)
```

### 生成可视化

```python
from lukit.visualization import LatexTableGenerator, PlotGenerator, load_metrics

# 加载评测结果
metrics = load_metrics("./metrics.json")

# 生成 LaTeX 表格
latex_gen = LatexTableGenerator(metrics)
latex_table = latex_gen.generate_performance_table(sort_by="auroc")
latex_gen.save_to_file(latex_table, "./results.tex")

# 生成 matplotlib 图片
plot_gen = PlotGenerator(metrics)
fig = plot_gen.generate_bar_comparison_plot(metric="auroc")
plot_gen.save_figure(fig, "./figure1.pdf", dpi=300)
```

## 支持的方法

| 方法 | 描述 | 类型 | 需要采样 |
|------|------|------|----------|
| `sequence_log_probability` | 生成序列的对数概率 | 内在 | 否 |
| `perplexity` | 生成文本的困惑度 | 内在 | 否 |
| `mean_token_entropy` | Token 平均熵 | 内在 | 否 |
| `self_certainty` | 自确定性分数 | 内在 | 否 |
| `monte_carlo_sequence_entropy` | 蒙特卡洛序列熵 | 采样 | 是 |
| `lexical_similarity` | 采样响应间的词汇相似度 | 采样 | 是 |
| `p_true` | 回答为真的概率 | 交互式 | 否 |
| `eigenscore` | 基于采样表示的特征值分数 | 表征 | 是 |
| `deg_mat` | 基于回答图平均度的语义图方法 | 图方法 | 是 |
| `eccentricity` | 基于图拉普拉斯谱嵌入离散度的语义图方法 | 图方法 | 是 |
| `eig_val_laplacian` | 基于图拉普拉斯特征值和的语义图方法 | 图方法 | 是 |
| `num_sem_sets` | 基于语义连通分量个数的语义图方法 | 图方法 | 是 |

## 添加自定义方法

1. 在 `lukit/methods/` 创建新文件:

```python
# lukit/methods/my_method.py
from .base_method import UncertaintyMethod

class MyMethod(UncertaintyMethod):
    name = "my_method"
    requires_data = ["logprob_stats"]
    tags = ["intrinsic"]

    def _compute(self, stats):
        logprobs = stats["logprob_stats"]["completion_log_probs"]
        if not logprobs:
            return {"u": 0.0}
        return {"u": float(-sum(logprobs) / len(logprobs))}
```

2. 在 `lukit/methods/__init__.py` 注册:

```python
_METHOD_MODULES = [
    # ... existing methods
    "my_method",
]
```

3. 验证:

```bash
lukit list-methods
```

## 项目结构

```
lukit/
├── lukit/                    # 主包
│   ├── backends/             # 模型后端实现
│   ├── bin/                  # CLI 可执行脚本
│   ├── cli/                  # 命令行接口
│   ├── data_providers/       # 数据提供者
│   ├── engine/               # 执行引擎
│   ├── methods/              # 不确定性方法
│   └── visualization/        # 可视化工具
├── configs/                  # 配置文件
│   └── chat_template.json    # 生成 prompt 模板
├── pyproject.toml            # 包配置
└── README.md                 # 本文档
```

## CLI 命令参考

### lukit

统一 CLI，支持以下子命令:

```bash
lukit list-methods
lukit eval --gm_model_path <path> --dataset_name <name> --methods all
lukit leaderboard <metrics_path> --sort_by auroc
lukit visualize --metrics <path> --output_dir <dir> --plot_type all
```

`lukit --list-methods`、`--model_path`、`--device`、`--judge_model_path`、`--judge_device` 也保留兼容。

### lukit-eval

运行不确定性评测。它与 `lukit eval` 使用同一套批量评测参数。

```bash
lukit-eval \
  --gm_model_path <path> \
  --dataset_name <name> \
  --methods all
```

建议直接用 `lukit eval --help` 查看完整参数列表。帮助信息会按以下分组展示：

- `Generation`: `--gm_model_path` `--gm_device` `--torch_dtype` `--chat_template_config`
- `Dataset`: `--dataset_source` `--dataset_name` `--dataset_mode` `--dataset_dir` `--start_idx` `--num_samples_eval`（设为 `-1` 表示评估全部剩余样本）
- `Methods`: `--methods` `--max_new_tokens` `--temperature` `--top_p`
- `Sampling Family`: `--num_samples` `--sample_temperature` `--sample_top_p` `--lexical_metric` `--p_true_with_context`
- `Semantic Graph Family`: `--semantic_similarity_score` `--semantic_affinity` `--semantic_temperature` `--semantic_jaccard_threshold` `--nli_model_path` `--nli_device` `--nli_torch_dtype`
- `Semantic Entropy Family`: `--semantic_class_source` `--equivalence_judger_model_path` `--equivalence_judger_device` `--equivalence_judger_torch_dtype` `--equivalence_judger_max_new_tokens`
- `Judge`: `--jm_model_path` `--jm_device` `--judge_max_new_tokens` `--judge_mode`
- `Output`: `--out_jsonl` `--out_metrics`

一个较完整的示例：

```bash
lukit-eval \
  --gm_model_path /path/to/model \
  --gm_device cuda:0 \
  --dataset_source hf \
  --dataset_name trivia_qa_split \
  --methods deg_mat,eccentricity,semantic_entropy,p_true \
  --num_samples 4 \
  --sample_temperature 0.8 \
  --semantic_similarity_score nli \
  --nli_model_path /path/to/nli-model \
  --nli_device cuda:1 \
  --semantic_class_source nli \
  --jm_model_path /path/to/judge-model \
  --jm_device cuda:2 \
  --num_samples_eval 100 \
  --out_jsonl ./results.jsonl \
  --out_metrics ./metrics.json
```

### lukit-leaderboard

显示结果榜单。

```bash
lukit-leaderboard <metrics_path> \
  --sort_by <metric>           # 排序指标 (auroc/auprc/accuracy)
```

### lukit-visualize

生成 LaTeX 表格和 matplotlib 图片。

```bash
lukit-visualize \
  --metrics <path>             # 指标文件路径 (必填)
  --output_dir <dir>           # 输出目录
  --table_type <type>          # 表格类型 (performance/full/both/none)
  --plot_type <type>           # 图片类型 (none/bar/multi/all)
  --plot_format <format>       # 图片格式 (png/pdf/svg)
  --bar_metric <metric>        # 单图指标 (auroc/auprc/accuracy)
  --multi_metrics <a,b>        # 组合图指标列表
```

## 输出文件格式

### metrics.json

```json
{
  "dataset_name": "trivia_qa_split",
  "dataset_source": "hf",
  "num_samples_eval": 100,
  "methods": ["p_true", "perplexity", ...],
  "metrics": {
    "p_true": {
      "auroc": 0.8523,
      "auprc": 0.7891,
      "n_valid": 98,
      "n_correct": 65,
      "n_error": 33
    }
  }
}
```

### results.jsonl

每行一个 JSON 对象:

```json
{
  "sample_idx": 0,
  "q": "Question text",
  "a_gold": "Ground truth",
  "a_model": "Model answer",
  "u": {"p_true": {"u": 0.85}, "perplexity": {"u": 12.3}},
  "correct": 1,
  "judge_mode": "json"
}
```

## 许可证

Apache License 2.0

## 贡献

欢迎提交 Issue 和 Pull Request！

## 引用

如果您在研究中使用了 LUKIT，请引用:

```bibtex
@software{lukit2024,
  title = {LUKIT: LLM Uncertainty Kit},
  author = {LUKIT Contributors},
  year = {2024},
  url = {https://github.com/baidu/lukit}
}
```
