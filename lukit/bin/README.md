# Bin Module

The `bin` module contains executable scripts and command-line tools for LUKIT.

## Overview

This module provides standalone CLI tools that can be invoked as separate commands:

- `lukit-eval` - Run uncertainty evaluation
- `lukit-leaderboard` - Display results leaderboard
- `lukit-visualize` - Generate LaTeX tables and plots

## Scripts

### evaluate.py

Batch evaluation script for uncertainty quantification on QA datasets.

**Usage:**

```bash
lukit-eval \
  --gm_model_path /path/to/model \
  --gm_device cuda:0 \
  --methods all \
  --nli_model_path /path/to/nli-model \
  --nli_device cuda:1
```

**Key Features:**

- Batch evaluation on datasets
- Multiple uncertainty methods
- Configurable inference parameters
- JSON and JSONL output formats

### leaderboard.py

Display uncertainty method results in a tabular leaderboard format.

**Usage:**

```bash
lukit-leaderboard ./metrics_lukit.json --sort_by auroc
```

**Key Features:**

- Sortable by different metrics
- Summary statistics

### visualize.py

Generate LaTeX tables and matplotlib plots for research papers.

**Usage:**

```bash
lukit-visualize \
  --metrics ./metrics_lukit.json \
  --output_dir ./output \
  --table_type both \
  --plot_type all
```

**Key Features:**

- LaTeX table generation
- Multiple plot types (`bar`, `multi`, `all`)
- Configurable output formats (PNG, PDF, SVG)

## Installation

These scripts are automatically installed as entry points when you install LUKIT:

```bash
pip install -e .
```

After installation, the commands are available directly:

```bash
lukit-eval --help
lukit-leaderboard --help
lukit-visualize --help
```

## Usage Patterns

### Complete Evaluation Workflow

```bash
# 1. Run evaluation
lukit-eval \
  --gm_model_path /path/to/model \
  --gm_device cuda:0 \
  --methods all \
  --nli_model_path /path/to/nli-model \
  --num_samples_eval 100 \
  --out_metrics ./metrics.json

# 2. View leaderboard
lukit-leaderboard ./metrics.json

# 3. Generate paper figures
lukit-visualize \
  --metrics ./metrics.json \
  --output_dir ./paper_assets
```
