"""Visualization module for generating LaTeX tables and matplotlib plots."""

import json
import os
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "LatexTableGenerator",
    "PlotGenerator",
    "format_metrics_for_latex",
    "load_metrics",
]


class LatexTableGenerator:
    """Generate LaTeX tables for uncertainty metrics."""

    def __init__(self, metrics_data: Dict):
        """Initialize with metrics data.

        Args:
            metrics_data: Dictionary containing metrics output from lukit evaluation
        """
        self.metrics_data = metrics_data
        self.metrics = metrics_data.get("metrics", {})

    def generate_performance_table(
        self,
        sort_by: str = "auroc",
        include_all: bool = False,
        caption: str = "Uncertainty Method Performance Comparison",
        label: str = "tab:uq_performance",
    ) -> str:
        """Generate a LaTeX table comparing method performance.

        Args:
            sort_by: Column to sort by (auroc, auprc, n_correct)
            include_all: Whether to include all methods or only top performers
            caption: Table caption
            label: Table label for referencing

        Returns:
            LaTeX table string
        """
        method_stats = []
        for method_name, method_metrics in self.metrics.items():
            auroc = method_metrics.get("auroc", None)
            auprc = method_metrics.get("auprc", None)
            n_correct = method_metrics.get("n_correct", 0)
            n_valid = method_metrics.get("n_valid", 0)
            accuracy = n_correct / n_valid if n_valid > 0 else 0.0

            method_stats.append({
                "method": method_name,
                "auroc": auroc,
                "auprc": auprc,
                "accuracy": accuracy,
                "n_correct": n_correct,
                "n_valid": n_valid,
            })

        reverse = True
        method_stats.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)

        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{" + caption + "}",
            "\\label{" + label + "}",
            "\\begin{tabular}{lccc}",
            "\\hline",
            "Method & AUROC & AUPRC & Accuracy \\\\",
            "\\hline",
        ]

        for i, stat in enumerate(method_stats):
            auroc_str = f"{stat['auroc']:.4f}" if stat['auroc'] is not None else "N/A"
            auprc_str = f"{stat['auprc']:.4f}" if stat['auprc'] is not None else "N/A"
            acc_str = f"{stat['accuracy']:.4f}"

            if i == 0:
                latex_lines.append(f"\\textbf{{{stat['method']}}} & \\textbf{{{auroc_str}}} & \\textbf{{{auprc_str}}} & \\textbf{{{acc_str}}} \\\\")
            else:
                latex_lines.append(f"{stat['method']} & {auroc_str} & {auprc_str} & {acc_str} \\\\")

        latex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(latex_lines)

    def generate_full_metrics_table(
        self,
        caption: str = "Complete Uncertainty Metrics",
        label: str = "tab:uq_metrics_full",
    ) -> str:
        """Generate a comprehensive LaTeX table with all metrics."""
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{" + caption + "}",
            "\\label{" + label + "}",
            "\\begin{tabular}{lccccc}",
            "\\hline",
            "Method & Valid & Correct & Errors & AUROC & AUPRC \\\\",
            "\\hline",
        ]

        for method_name, method_metrics in self.metrics.items():
            n_valid = method_metrics.get("n_valid", 0)
            n_correct = method_metrics.get("n_correct", 0)
            n_error = method_metrics.get("n_error", 0)
            auroc = method_metrics.get("auroc", None)
            auprc = method_metrics.get("auprc", None)

            auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
            auprc_str = f"{auprc:.4f}" if auprc is not None else "N/A"

            latex_lines.append(
                f"{method_name} & {n_valid} & {n_correct} & {n_error} & {auroc_str} & {auprc_str} \\\\"
            )

        latex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(latex_lines)

    def save_to_file(self, latex_content: str, output_path: str) -> None:
        """Save LaTeX content to file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_content)


class PlotGenerator:
    """Generate matplotlib plots for uncertainty metrics."""

    def __init__(self, metrics_data: Dict, records: Optional[List[Dict]] = None):
        """Initialize with metrics data."""
        self.metrics_data = metrics_data
        self.metrics = metrics_data.get("metrics", {})
        self.records = records

    @staticmethod
    def _resolve_metric_value(method_metrics: Dict, metric: str) -> Optional[float]:
        if metric == "accuracy":
            n_valid = method_metrics.get("n_valid", 0)
            if not n_valid:
                return None
            return float(method_metrics.get("n_correct", 0)) / float(n_valid)
        value = method_metrics.get(metric, None)
        if value is None:
            return None
        return float(value)

    def generate_bar_comparison_plot(
        self,
        metric: str = "auroc",
        title: str = "Uncertainty Method Comparison",
        figsize: tuple = (10, 6),
        color: str = "#1f77b4",
    ):
        """Generate a bar chart comparing methods on a specific metric."""
        import matplotlib.pyplot as plt

        methods = []
        values = []
        for method_name, method_metrics in self.metrics.items():
            value = self._resolve_metric_value(method_metrics, metric)
            if value is not None:
                methods.append(method_name)
                values.append(value)

        if not methods:
            raise ValueError(f"No valid data for metric: {metric}")

        sorted_pairs = sorted(zip(methods, values), key=lambda x: x[1], reverse=True)
        methods, values = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(methods, values, color=color)
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.set_ylim(0, 1.0)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

        if len(methods) > 5:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        return fig

    def generate_multi_metric_comparison(
        self,
        metrics: List[str] = ["auroc", "auprc"],
        title: str = "Uncertainty Method Multi-Metric Comparison",
        figsize: tuple = (12, 6),
    ):
        """Generate a grouped bar chart comparing methods on multiple metrics."""
        import matplotlib.pyplot as plt

        all_methods = set(self.metrics.keys())
        method_data = {metric: {} for metric in metrics}

        for method_name, method_metrics in self.metrics.items():
            for metric in metrics:
                value = self._resolve_metric_value(method_metrics, metric)
                if value is not None:
                    method_data[metric][method_name] = value

        common_methods = set.intersection(*[set(m.keys()) for m in method_data.values()])
        if not common_methods:
            raise ValueError(f"No common methods have valid values for metrics: {metrics}")
        common_methods = sorted(common_methods, key=lambda x: sum(method_data[m].get(x, 0) for m in metrics), reverse=True)

        n_methods = len(common_methods)
        n_metrics = len(metrics)
        x = np.arange(n_methods)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=figsize)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, metric in enumerate(metrics):
            values = [method_data[metric].get(method, 0) for method in common_methods]
            offset = (i - n_metrics / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=metric.upper(), color=colors[i % len(colors)])

        ax.set_xlabel("Method")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(common_methods, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        return fig

    def save_figure(self, fig, output_path: str, dpi: int = 300) -> None:
        """Save matplotlib figure to file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)


def load_metrics(metrics_path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_metrics_for_latex(
    metrics_data: Dict,
    sort_by: str = "auroc",
    full: bool = False,
) -> str:
    """Convenience helper for generating a LaTeX table from metrics."""
    generator = LatexTableGenerator(metrics_data)
    if full:
        return generator.generate_full_metrics_table()
    return generator.generate_performance_table(sort_by=sort_by)
