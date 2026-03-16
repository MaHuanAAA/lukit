# CLI Module

The `cli` module provides command-line interfaces for interacting with LUKIT.

## Overview

The CLI module offers multiple commands for:

- Running uncertainty evaluations
- Displaying result leaderboards
- Generating visualizations
- Listing available methods

## Commands

### Main CLI (`lukit`)

The main entry point provides a unified interface to evaluation, leaderboard, and visualization.

```bash
# Show help
lukit --help

# List available methods
lukit list-methods

# Run evaluation
lukit eval \
  --gm_model_path /path/to/model \
  --gm_device cuda:0 \
  --methods all \
  --nli_model_path /path/to/nli-model \
  --nli_device cuda:1

# View leaderboard
lukit leaderboard ./metrics_lukit.json --sort_by auroc

# Generate tables and plots
lukit visualize --metrics ./metrics_lukit.json --output_dir ./lukit_output --plot_type all
```

Compatibility shortcuts are also supported:

```bash
lukit --list-methods
lukit --model_path /path/to/model --methods all
```

### Individual CLI Tools

#### `lukit-eval`

Run uncertainty evaluation on a model.

```bash
lukit-eval \
  --gm_model_path /path/to/model \
  --gm_device cuda:0 \
  --jm_model_path /path/to/judge-model \
  --jm_device cuda:1 \
  --methods all \
  --nli_model_path /path/to/nli-model \
  --nli_device cuda:2 \
  --num_samples_eval 100
```

**Key Arguments:**

- `--gm_model_path`: Generation model path (required)
- `--gm_device`: Generation model device (default: cuda:0)
- `--jm_model_path`: Judge model path
- `--jm_device`: Judge model device (default: cuda:1)
- `--semantic_similarity_score`: `nli` or `jaccard` for the 4 semantic graph methods
- `--nli_model_path`: Required when semantic graph methods use `nli`
- `--nli_device`: Device for the NLI model
- `--methods`: Methods to run (default: all)
- `--num_samples_eval`: Number of samples to evaluate

#### `lukit-leaderboard`

Display results in a leaderboard format.

```bash
lukit-leaderboard ./metrics_lukit.json --sort_by auroc
```

**Key Arguments:**

- `metrics`: Path to metrics JSON file (required)
- `--sort_by`: Sort by auroc, auprc, or accuracy

#### `lukit-visualize`

Generate LaTeX tables and matplotlib plots.

```bash
lukit-visualize \
  --metrics ./metrics_lukit.json \
  --output_dir ./lukit_output \
  --table_type both \
  --plot_type all
```

**Key Arguments:**

- `--metrics`: Path to metrics JSON file (required)
- `--output_dir`: Directory for output files
- `--table_type`: `performance`, `full`, `both`, or `none`
- `--plot_type`: `none`, `bar`, `multi`, or `all`
- `--plot_format`: Output format such as `pdf`, `png`, or `svg`

## Installation and Usage

After installing LUKIT:

```bash
# Install in development mode
pip install -e .

# Use the CLI
lukit --help
```
