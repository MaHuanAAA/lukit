# lukit Methods Tutorial


## 1. 运行流程（代码级）

1. `ExecutionEngine.run(...)` 接收样本与方法列表（`engine/execution_engine.py`）。
2. 引擎收集所有方法的 `requires_data`，通过 `DependencyResolver` 选择对应 `DataProvider`。
3. 每个样本先做一次主模型生成：得到 `answer_text`、`prompt_ids`、`completion_ids`、`model_inputs`。
4. 依次执行 provider，得到 `stats`（如 `logprob_stats`、`sampling_stats`、`p_true_prob`）。
5. 每个方法只读取自己声明的 `stats` 子集，调用 `_compute` 返回结果。
6. 输出结构中 `u` 是各方法结果字典，键为方法名。
7. 若采样阶段触发 CUDA OOM，`SamplingStatsProvider` 会返回空采样并附带 warning；引擎把 warning 写入样本输出的 `warnings` 字段。

## 2. 底层统计量定义

来自 `HFBackend.compute_logprob_stats(...)` 与 `collect_sampling_stats(...)`：

- 设生成 token 序列长度为 $T$，第 $t$ 个 token 条件对数概率为
  $$
  \ell_t=\log p(y_t\mid x,y_{<t}).
  $$
- `completion_log_probs`：$[\ell_1,\dots,\ell_T]$。
- `token_entropies`：每个位置的词表熵
  $$
  H_t=-\sum_{v}p_t(v)\log p_t(v).
  $$
- `mean_logp_vocab`：每个位置上词表 log-prob 均值
  $$
  m_t=\frac{1}{|V|}\sum_v \log p_t(v).
  $$
- `sampled_sequence_nlls`：对每条问题采样回答 $k$，
  $$
  \text{NLL}_k=-\frac{1}{T_k}\sum_{t=1}^{T_k}\ell_{k,t}.
  $$

## 3. 方法逐个解析

### 3.1 `sequence_log_probability`

- 依赖：`logprob_stats`
- 实现：
  $$
  u=-\frac{1}{T}\sum_{t=1}^{T}\ell_t.
  $$
- 代码行为：若空输出（`T=0`），返回 `0.0`。

### 3.2 `perplexity`

- 依赖：`logprob_stats`
- 实现：
  $$
  u=\exp\!\left(-\frac{1}{T}\sum_{t=1}^{T}\ell_t\right).
  $$
- 即 token-level 平均负对数似然的指数形式。空输出返回 `0.0`。

### 3.3 `mean_token_entropy`

- 依赖：`logprob_stats`
- 实现：
  $$
  u=\frac{1}{T}\sum_{t=1}^{T}H_t.
  $$
- 空输出返回 `0.0`。

### 3.4 `self_certainty`

- 依赖：`logprob_stats`
- 中间量（代码名 `kl_uniform_to_p`）：
  $$
  \mathrm{KL}(U\|p_t)= -m_t-\log|V|.
  $$
- 输出：
  $$
  u=-\frac{1}{T}\sum_{t=1}^{T}\mathrm{KL}(U\|p_t).
  $$
- 注意：实现返回的是负 KL，常为非正值；数值越负通常表示分布越尖锐（更“确定”）。

### 3.5 `monte_carlo_sequence_entropy`

- 依赖：`sampling_stats`
- 实现（代码直接平均采样 NLL）：
  $$
  u=\frac{1}{K}\sum_{k=1}^{K}\text{NLL}_k.
  $$
- 若无采样（`K=0`），返回 `0.0`。
- 若采样 OOM，被 provider 降级为空采样，同样返回 `0.0`（并在样本里有 warning）。
- 说明：名称是 “sequence_entropy”，但当前实现是 Monte Carlo 的平均序列 NLL。

### 3.6 `lexical_similarity`

- 依赖：`sampling_stats`
- 对采样回答两两计算相似度 $s_{ij}$，输出
  $$
  u=-\frac{1}{\binom{K}{2}}\sum_{i<j}s_{ij}.
  $$
- 相似度计算顺序：
  1. `metric` 以 `rouge` 开头且安装 `rouge-score`：用 ROUGE F1；
  2. `metric == BLEU` 且安装 `nltk`：用句级 BLEU（权重随最短句长调整）；
  3. 否则回退 `difflib.SequenceMatcher(...).ratio()`。
- 若采样文本少于 2 条，返回 `0.0`。
- 若采样 OOM，等价于采样文本为空，也会返回 `0.0`（并带 warning）。

### 3.7 `p_true`

- 依赖：`p_true_prob`（来自 `PTrueProvider`）。
- `p_true_prob` 的计算：构造 True/False 判别提示词，只取“下一 token”在 `True` 与 `False` 上的 logits，做二元 softmax 得到
  $$
  p_{\text{true}}.
  $$
- 当前实现中，`p_true` 使用的是同一个主生成模型后端（不是独立 judge 模型）。
- 方法输出：
  $$
  u=-\log\big(\mathrm{clip}(p_{\text{true}},10^{-12},1)\big).
  $$
- 返回字段为 `{"u": u, "p_true": p_true}`。

## 4. 参数如何影响方法

- `--num_samples` / `--sample_temperature` / `--sample_top_p`：
  影响 `sampling_stats`，因此影响 `monte_carlo_sequence_entropy` 与 `lexical_similarity`。
- `--lexical_metric`：仅影响 `lexical_similarity`。
- `--p_true_with_context`：仅影响 `p_true` provider 的提示词（是否加入 `extra_context`）。
- `--temperature` / `--top_p`：影响主回答 `a_model`，从而间接影响所有方法。
- 当 `--temperature 0.0` 时主生成走贪心，`top_p` 在主生成中不会生效（代码只在 `do_sample=True` 时传入 `top_p`）。

## 5. 一个最小调用示意

```python
from lukit.methods import create_method
from lukit.engine import ExecutionEngine

engine = ExecutionEngine(
    model="/path/to/model",
    backend_config={"type": "huggingface", "device": "cuda:0"},
)
methods = [create_method("perplexity"), create_method("p_true", with_context=False)]
records = engine.run([{"question": "What is 2+2?", "answer": "4"}], methods)
print(records[0]["u"])
```

`records[0]["u"]` 会包含每个方法的输出字典（例如 `{"perplexity": {"u": ...}, "p_true": {"u": ..., "p_true": ...}}`）。
