# lukit Methods Tutorial

本文严格对应当前代码实现，逐步解释 `lukit` 中当前已实现的不确定性方法（`methods/`）。


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
- `sampled_sequence_logprobs`：对每条采样回答 $k$ 的精确序列对数概率，
  $$
  \log p(y_k\mid x)=\sum_{t=1}^{T_k}\ell_{k,t}.
  $$
  当前 `semantic_entropy` 使用的是这个量，而不是 `sampled_sequence_nlls`。
- `similarity_matrix`：采样回答两两相似度矩阵。
  - 当 `--semantic_similarity_score jaccard` 时，使用词级 Jaccard 相似度。
  - 当 `--semantic_similarity_score nli` 时，使用 NLI 模型打分后按 `affinity` 派生的对称相似度矩阵。
- `semantic_matrix_entail` / `semantic_matrix_contra`：仅在 NLI 语义模式下提供的有向 entail / contradiction 概率矩阵，用于 `num_sem_sets` 等图方法。
- `semantic_classes`：语义类划分结果，包含：
  - `sample_to_class`
  - `class_to_sample`
  当前 `semantic_entropy` / `semantic_entropy_empirical` 直接使用该字段。

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

### 3.8 `eigenscore`

- 依赖：`sampling_stats`。
- 采样阶段会为每个 sample 额外提取一个向量 `eigenscore_embeddings[i]\in\mathbb{R}^d`：
  - hidden states 取中间层（`len(hidden_states)//2`）；
  - token 索引优先取 completion 的 `$T_k-2$`，过短时回退到最后一个 completion token。
- 令矩阵 $H\in\mathbb{R}^{N\times d}$ 为所有采样 embedding 堆叠（$N$ 为采样数），实现中计算：
  $$
  C=\mathrm{cov}(H)\in\mathbb{R}^{N\times N},
  \qquad
  C\leftarrow C+\alpha I,\ \alpha=10^{-3}.
  $$
- 特征值分解后输出：
  $$
  u=\frac{1}{N}\sum_{i=1}^{N}\log_{10}(\lambda_i).
  $$
- 数值稳定：特征值会被裁剪到至少 $10^{-12}$ 再取对数。

### 3.9 `deg_mat`

- 依赖：`sampling_stats`
- 构图：对采样回答构造相似度矩阵 $W$，来源由 `--semantic_similarity_score` 决定。
- 实现中保留方向性，不做对称化。
- 定义距离矩阵：
  $$
  \mathrm{Dist}_{ij}=1-W_{ij}.
  $$
- 个体分数：
  $$
  c_i=\sum_j \mathrm{Dist}_{ij}.
  $$
- 整体分数：
  $$
  u=\frac{1}{N}\sum_i c_i.
  $$
- 返回字段为 `{"u": u, "c": [...]}`。
- 直觉：每个回答与其它回答越“不一致”，其反向度越高，整体不确定性越高。

### 3.10 `eig_val_laplacian`

- 依赖：`sampling_stats`
- 使用归一化图拉普拉斯：
  $$
  L = I - D^{-1/2}WD^{-1/2}.
  $$
- 输出：
  $$
  u=\sum_k \max(0, 1-\lambda_k),
  $$
  其中 $\lambda_k$ 是 $L$ 的特征值。
- 直觉：这是 `num_sem_sets` 的连续化谱版本；图中越像有多个语义簇，越会出现更多较小特征值，分数越高。

### 3.11 `eccentricity`

- 依赖：`sampling_stats`
- 同样基于归一化图拉普拉斯 $L$ 的特征分解。
- 仅保留满足 $\lambda_k < \texttt{thres}$ 的特征向量（默认 `0.9`）。
- 设保留后的特征向量组成谱嵌入矩阵 $V\in\mathbb{R}^{N\times K}$，其中第 $i$ 行 $V_i$ 是第 $i$ 条采样回答的谱坐标。
- 先计算质心
  $$
  \mu=\frac{1}{N}\sum_i V_i,
  $$
  再计算每个样本到质心的距离
  $$
  c_i=\|V_i-\mu\|_2.
  $$
- 最终输出
  $$
  u=\left\|\left[c_i\right]_i\right\|_2.
  $$
- 返回字段为 `{"u": u, "c": [...]}`。
- 直觉：如果低频谱嵌入更分散，说明回答图的簇结构更明显，不确定性更高。

### 3.12 `num_sem_sets`

- 依赖：`sampling_stats`
- NLI 模式：
  - 对有向回答对 $(a_i, a_j)$，若 `entail > contra`，记为一条候选边；
  - 只有 $i \to j$ 与 $j \to i$ 都满足时，才保留无向边。
- Jaccard 模式：
  - 当两条回答的 Jaccard 相似度 $\ge 0.5$ 时连边。
- 最终输出图的连通分量个数：
  $$
  u = \#\text{connected components}.
  $$
- 直觉：回答如果分裂成多个互不支持的语义簇，连通分量更多，不确定性更高。

### 3.13 `semantic_entropy`

- 依赖：`sampling_stats`
- 需要 `semantic_classes` 与 `sampled_sequence_logprobs`。
- 先把采样回答按语义类聚合。设第 $c$ 个语义类包含的 sample 下标集合为 $\mathcal{C}_c$，则类级对数概率为
  $$
  \log p(c)=\log \sum_{j\in\mathcal{C}_c}\exp(\log p(y_j\mid x)).
  $$
- 对每个 sample，取其所属语义类的类级对数概率，再做负均值：
  $$
  u=-\frac{1}{N}\sum_{i=1}^{N}\log p(c(i)).
  $$
- 返回字段为 `{"u": u, "n_classes": ...}`。
- `semantic_class_source` 决定语义类来源：
  - `nli`：按 mutual entailment 划分类别
  - `equivalence_judger`：用独立语义等价判别模型做增量聚类

### 3.14 `semantic_entropy_empirical`

- 依赖：`sampling_stats`
- 需要 `semantic_classes`。
- 若共有 $M$ 个语义类，第 $c$ 类包含样本数 $|\mathcal{C}_c|$，经验类概率为
  $$
  p_c=\frac{|\mathcal{C}_c|}{N}.
  $$
- 输出经验语义熵：
  $$
  u=-\sum_{c=1}^{M} p_c\log p_c.
  $$
- 返回字段为 `{"u": u, "n_classes": ...}`。

## 4. 参数如何影响方法

- `--num_samples` / `--sample_temperature` / `--sample_top_p`：
  影响 `sampling_stats`，因此影响 `monte_carlo_sequence_entropy`、`lexical_similarity`、`eigenscore`、`deg_mat`、`eig_val_laplacian`、`eccentricity`、`num_sem_sets`、`semantic_entropy`、`semantic_entropy_empirical`。
- `--lexical_metric`：仅影响 `lexical_similarity`。
- `--p_true_with_context`：仅影响 `p_true` provider 的提示词（是否加入 `extra_context`）。
- `--semantic_similarity_score`：
  控制 4 个语义图方法是使用 `nli` 还是 `jaccard` 相似度。
- `--semantic_affinity`：
  在 `nli` 模式下，控制 `deg_mat`、`eig_val_laplacian`、`eccentricity` 这 3 个图方法的边权是使用 entail 概率，还是使用 `1 - contradiction`。`num_sem_sets` 的 NLI 连边规则固定为双向 `entail > contra`。
- `--semantic_jaccard_threshold`：
  在 `jaccard` 模式下，仅影响 `num_sem_sets` 的连边阈值。
- `--semantic_class_source`：
  控制 `semantic_entropy` 与 `semantic_entropy_empirical` 的语义类来源，可选 `nli` 或 `equivalence_judger`。
- `--nli_model_path` / `--nli_device` / `--nli_torch_dtype`：
  在 `--semantic_similarity_score nli` 或 `--semantic_class_source nli` 时生效，控制独立 NLI 模型。
- `--equivalence_judger_model_path` / `--equivalence_judger_device` / `--equivalence_judger_torch_dtype` / `--equivalence_judger_max_new_tokens`：
  仅在 `--semantic_class_source equivalence_judger` 时生效，控制语义等价判别模型。
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
