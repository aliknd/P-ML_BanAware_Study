#!/usr/bin/env python3
"""
aggregate_comparison.py
=======================

Recursively finds all `comparison_summary.csv` files under a root directory,
extracts ID, Substance, Task, and per-pipeline metrics, then outputs a
combined table with a MultiIndex (ID, Substance, Task, Pipeline) and columns:

  Threshold | Sensitivity | Specificity | Accuracy | ROC AUC

Usage:
  python aggregate_comparison.py --root /path/to/results --out combined_results.csv
"""
import os
import glob
import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate comparison_summary.csv files with MultiIndex output")
    p.add_argument("--root", required=True,
                   help="Root directory containing comparison_summary.csv files")
    p.add_argument("--out", default="combined_results.csv",
                   help="Filename for aggregated output CSV")
    return p.parse_args()


def main(root_dir, out_file):
    # normalize path
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    print(f"Searching for comparison_summary.csv under '{root_dir}'...")
    pattern = os.path.join(root_dir, "**", "comparison_summary.csv")
    matches = glob.glob(pattern, recursive=True)
    print(f"Found {len(matches)} comparison_summary.csv files")

    records = []
    for csv_path in matches:
        parts = csv_path.split(os.sep)
        if len(parts) < 3:
            continue
        id_ = parts[-3]
        fruit_task = parts[-2]
        if "_" in fruit_task:
            substance, task = fruit_task.split("_", 1)
        else:
            substance, task = fruit_task, ""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping unreadable {csv_path}: {e}")
            continue
        for _, row in df.iterrows():
            records.append({
                "ID": id_,
                "Substance": substance,
                "Task": task,
                "Pipeline": row.get("Pipeline", ""),
                "Threshold": row.get("Best_Threshold", ""),
                "Sensitivity": f"{row.get('Sensitivity_Mean',0):.3f} ± {row.get('Sensitivity_STD',0):.3f}",
                "Specificity": f"{row.get('Specificity_Mean',0):.3f} ± {row.get('Specificity_STD',0):.3f}",
                "Accuracy": f"{row.get('Accuracy_Mean',0):.3f} ± {row.get('Accuracy_STD',0):.3f}",
                "ROC AUC": f"{row.get('AUC_Mean',0):.3f} ± {row.get('AUC_STD',0):.3f}",
            })

    if not records:
        print(f"No comparison_summary.csv files found under '{root_dir}'")
        return

    combined = pd.DataFrame.from_records(records)
    combined = combined[["ID", "Substance", "Task", "Pipeline",
                         "Threshold", "Sensitivity", "Specificity",
                         "Accuracy", "ROC AUC"]]
    combined.set_index(["ID", "Substance", "Task", "Pipeline"], inplace=True)
    combined.sort_index(level=["ID", "Pipeline"], inplace=True)

    combined.to_csv(out_file)
    print(f"Aggregated {len(combined)} rows to {out_file}")

    print("\nCombined results (MultiIndex):")
    print(combined.to_markdown())


if __name__ == "__main__":
    args = parse_args()
    main(args.root, args.out)
