#!/usr/bin/env python3
"""Lukit visualization CLI - Generate LaTeX tables and matplotlib plots."""

import argparse
import os
import sys

from ..visualization import LatexTableGenerator, PlotGenerator, load_metrics


def main() -> None:
    """Main entry point for the visualization CLI."""
    parser = argparse.ArgumentParser(
        description="lukit visualization CLI - Generate LaTeX tables and plots"
    )

    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to metrics JSON file (output from lukit-eval)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lukit_output",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--latex_file",
        type=str,
        default="uq_results.tex",
        help="Name for LaTeX output file",
    )
    parser.add_argument(
        "--table_type",
        type=str,
        default="performance",
        choices=["performance", "full", "both", "none"],
        help="Which LaTeX table(s) to generate",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="auroc",
        choices=["auroc", "auprc", "accuracy"],
        help="Metric to sort table by",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="none",
        choices=["none", "bar", "multi", "all"],
        help="Which plot(s) to generate",
    )
    parser.add_argument(
        "--plot_format",
        type=str,
        default="pdf",
        help="Matplotlib output format, e.g. pdf/png/svg",
    )
    parser.add_argument(
        "--bar_metric",
        type=str,
        default="auroc",
        choices=["auroc", "auprc", "accuracy"],
        help="Metric for the single bar plot",
    )
    parser.add_argument(
        "--multi_metrics",
        type=str,
        default="auroc,auprc",
        help="Comma-separated metrics for the grouped comparison plot",
    )

    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        print(f"Error: Metrics file not found: {args.metrics}")
        sys.exit(1)

    print(f"Loading metrics from: {args.metrics}")
    metrics_data = load_metrics(args.metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    generated_files = []

    latex_gen = LatexTableGenerator(metrics_data)
    if args.table_type in {"performance", "both"}:
        print("Generating performance LaTeX table...")
        latex_path = os.path.join(args.output_dir, args.latex_file)
        latex_gen.save_to_file(
            latex_gen.generate_performance_table(sort_by=args.sort_by),
            latex_path,
        )
        generated_files.append(latex_path)
    if args.table_type in {"full", "both"}:
        print("Generating full metrics LaTeX table...")
        full_latex_path = os.path.join(args.output_dir, "uq_full_metrics.tex")
        latex_gen.save_to_file(
            latex_gen.generate_full_metrics_table(),
            full_latex_path,
        )
        generated_files.append(full_latex_path)

    if args.plot_type != "none":
        plot_gen = PlotGenerator(metrics_data)
        plot_format = args.plot_format.lstrip(".")
        multi_metrics = [item.strip() for item in args.multi_metrics.split(",") if item.strip()]

        try:
            if args.plot_type in {"bar", "all"}:
                print(f"Generating {args.bar_metric} bar plot...")
                fig = plot_gen.generate_bar_comparison_plot(metric=args.bar_metric)
                bar_path = os.path.join(args.output_dir, f"uq_bar_plot_{args.bar_metric}.{plot_format}")
                plot_gen.save_figure(fig, bar_path)
                generated_files.append(bar_path)

            if args.plot_type in {"multi", "all"}:
                print(f"Generating multi-metric plot for: {', '.join(multi_metrics)}")
                fig = plot_gen.generate_multi_metric_comparison(metrics=multi_metrics)
                multi_path = os.path.join(args.output_dir, f"uq_multi_plot.{plot_format}")
                plot_gen.save_figure(fig, multi_path)
                generated_files.append(multi_path)
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", "") == "matplotlib":
                message = "matplotlib is required for --plot_type != none. Install project dependencies and retry."
                if generated_files:
                    print(f"Warning: plot generation skipped after creating: {', '.join(generated_files)}")
                raise SystemExit(message) from exc
            raise

    if not generated_files:
        print("No outputs were generated. Choose a table_type or plot_type other than 'none'.")
        return

    print("\nGenerated files:")
    for path in generated_files:
        print(f"  - {path}")
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
