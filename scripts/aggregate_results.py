"""Aggregate runs/results.csv into a Table-1-style comparison.

Reads per-fold results, computes mean ± std per (task, model), and prints a
side-by-side comparison with the paper's published Table 1 numbers.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# (task, model) -> (paper_mean, paper_std)  [from Caduceus paper Table 1]
PAPER = {
    ("human_or_worm", "ph"): (0.973, 0.001),
    ("human_or_worm", "ps"): (0.968, 0.002),
    ("human_enhancers_cohn", "ph"): (0.747, 0.004),
    ("human_enhancers_cohn", "ps"): (0.745, 0.007),
}

TASK_ORDER = ["human_or_worm", "human_enhancers_cohn"]
TASK_LABEL = {"human_or_worm": "HUMAN VS WORM", "human_enhancers_cohn": "HUMAN ENHANCERS COHN"}
MODEL_LABEL = {"ph": "Caduceus-Ph", "ps": "Caduceus-PS"}


def main():
    results_csv = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/results.csv")
    rows = list(csv.DictReader(open(results_csv)))
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["task"], r["model"])].append(float(r["best_val_acc"]))

    print(f"\nResults from {results_csv} ({len(rows)} fold-runs)\n")
    print(f"{'Task':<24}  {'Model':<14}  {'Ours (linear probe)':>22}  {'Paper (full FT)':>18}  {'Delta':>8}")
    print("-" * 95)
    for task in TASK_ORDER:
        for model in ["ph", "ps"]:
            key = (task, model)
            if key not in by_key:
                print(f"{TASK_LABEL[task]:<24}  {MODEL_LABEL[model]:<14}  {'(no data)':>22}")
                continue
            accs = np.array(by_key[key])
            ours = f"{accs.mean()*100:.2f} ± {accs.std()*100:.2f} %"
            paper_mean, paper_std = PAPER[key]
            paper_str = f"{paper_mean*100:.1f} ± {paper_std*100:.1f} %"
            delta = (accs.mean() - paper_mean) * 100
            print(f"{TASK_LABEL[task]:<24}  {MODEL_LABEL[model]:<14}  {ours:>22}  {paper_str:>18}  {delta:>+7.2f}pp")
    print()


if __name__ == "__main__":
    main()
