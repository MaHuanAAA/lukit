# Visualization Module

The `visualization` module provides tools for generating LaTeX tables and matplotlib plots from uncertainty evaluation results.

## Overview

The visualization module includes:

- **LatexTableGenerator**: Create publication-ready LaTeX tables
- **PlotGenerator**: Generate various matplotlib plots
- Utility functions for data formatting and loading

## Components

### LatexTableGenerator

Generate LaTeX tables for research papers.

#### Methods

- `generate_performance_table(sort_by="auroc")` - Basic performance comparison
- `generate_full_metrics_table()` - Complete metrics including sample counts
- `save_to_file(latex_content, output_path)` - Save LaTeX to file

#### Example

```python
from lukit.visualization import LatexTableGenerator, load_metrics

metrics = load_metrics("./metrics_lukit.json")
generator = LatexTableGenerator(metrics)

# Generate performance table
latex_table = generator.generate_performance_table(
    sort_by="auroc",
    caption="Uncertainty Method Performance Comparison",
    label="tab:uq_performance"
)

# Save to file
generator.save_to_file(latex_table, "./results.tex")
```

### PlotGenerator

Generate matplotlib plots for analysis and presentation.

#### Methods

- `generate_bar_comparison_plot(metric="auroc")` - Bar chart comparing methods
- `generate_multi_metric_comparison(metrics=["auroc", "auprc"])` - Grouped bar chart
- `save_figure(fig, output_path, dpi=300)` - Save figure to file

#### Example

```python
from lukit.visualization import PlotGenerator, load_metrics

metrics = load_metrics("./metrics_lukit.json")
generator = PlotGenerator(metrics)

# Generate bar plot
fig = generator.generate_bar_comparison_plot(
    metric="auroc",
    title="Uncertainty Method Comparison"
)

# Save as PDF
generator.save_figure(fig, "./uq_bar_plot.pdf", dpi=300)
```

## CLI Usage

### Generate Visualizations

```bash
lukit-visualize \
  --metrics ./metrics_lukit.json \
  --output_dir ./lukit_output \
  --table_type both \
  --plot_type all \
  --plot_format pdf
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--metrics` | Path to metrics JSON file (required) | - |
| `--output_dir` | Output directory | `./lukit_output` |
| `--table_type` | Table output: performance, full, both, none | `performance` |
| `--plot_type` | Plot type: none, bar, multi, all | `none` |
| `--plot_format` | Figure format | `pdf` |
| `--sort_by` | Sort column: auroc, auprc, accuracy | `auroc` |

## Python API

### Loading Data

```python
from lukit.visualization import load_metrics

metrics = load_metrics("./metrics_lukit.json")
```

### Complete Workflow

```python
import os
from lukit.visualization import (
    LatexTableGenerator,
    PlotGenerator,
    format_metrics_for_latex,
    load_metrics
)

# Load data
metrics = load_metrics("./metrics_lukit.json")

# Create output directory
os.makedirs("./paper_assets", exist_ok=True)

# Generate LaTeX tables
lat_gen = LatexTableGenerator(metrics)
lat_gen.save_to_file(
    lat_gen.generate_performance_table(),
    "./paper_assets/performance_table.tex"
)

# Or use the convenience helper
latex = format_metrics_for_latex(metrics, sort_by="auroc")

# Generate plots
plot_gen = PlotGenerator(metrics)
fig = plot_gen.generate_bar_comparison_plot(metric="auroc")
plot_gen.save_figure(fig, "./paper_assets/figure1.pdf", dpi=300)
```

## Best Practices

1. **For Papers**: Use PDF format and high DPI (300+)
2. **For Presentations**: Use PNG format with moderate DPI (150)
3. Use consistent naming conventions for output files
