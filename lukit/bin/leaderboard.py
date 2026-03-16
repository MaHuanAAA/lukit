#!/usr/bin/env python3
"""Lukit leaderboard CLI - Display and compare method results."""

import argparse
import json
import os
import sys

from tabulate import tabulate


def main() -> None:
    """Main entry point for the leaderboard CLI."""
    parser = argparse.ArgumentParser(
        description="lukit leaderboard CLI - Display uncertainty method results"
    )

    parser.add_argument(
        "metrics",
        type=str,
        help="Path to metrics JSON file (output from lukit-eval)",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="auroc",
        choices=["auroc", "auprc", "accuracy"],
        help="Column to sort by",
    )

    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        print(f"Error: Metrics file not found: {args.metrics}")
        sys.exit(1)

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    metrics = metrics_data.get("metrics", {})
    dataset_name = metrics_data.get("dataset_name", "Unknown")
    num_samples = metrics_data.get("num_samples_eval", 0)

    print(f"\n{'='*60}")
    print(f"LUKIT UNCERTAINTY METHOD LEADERBOARD")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {num_samples}")
    print(f"{'='*60}\n")

    table_data = []
    for method_name, method_metrics in metrics.items():
        auroc = method_metrics.get("auroc", None)
        auprc = method_metrics.get("auprc", None)
        n_valid = method_metrics.get("n_valid", 0)
        n_correct = method_metrics.get("n_correct", 0)
        accuracy = n_correct / n_valid if n_valid > 0 else 0.0

        table_data.append({
            "Method": method_name,
            "AUROC": f"{auroc:.4f}" if auroc is not None else "N/A",
            "AUPRC": f"{auprc:.4f}" if auprc is not None else "N/A",
            "Accuracy": f"{accuracy:.4f}",
            "Valid": n_valid,
            "Correct": n_correct,
        })

    reverse = True
    if args.sort_by == "auroc":
        table_data.sort(key=lambda x: float(x["AUROC"]) if x["AUROC"] != "N/A" else -1, reverse=reverse)
    elif args.sort_by == "auprc":
        table_data.sort(key=lambda x: float(x["AUPRC"]) if x["AUPRC"] != "N/A" else -1, reverse=reverse)
    elif args.sort_by == "accuracy":
        table_data.sort(key=lambda x: float(x["Accuracy"]), reverse=reverse)

    headers = ["Rank", "Method", "AUROC", "AUPRC", "Accuracy", "Valid", "Correct", "Errors"]
    table_rows = []
    for rank, row in enumerate(table_data, start=1):
        method_str = f"* {row['Method']}" if rank == 1 else row['Method']
        errors = row["Valid"] - row["Correct"]
        table_rows.append([
            f"#{rank}",
            method_str,
            row["AUROC"],
            row["AUPRC"],
            row["Accuracy"],
            row["Valid"],
            row["Correct"],
            errors,
        ])

    print(tabulate(table_rows, headers=headers, tablefmt="github"))

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
