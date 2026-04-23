#!/usr/bin/env python3
"""Plot non-zero exit counts by task."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "results" / "summary" / "benchmark_table.csv"
    out_path = root / "results" / "summary" / "failures.png"

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    failures = Counter()
    for row in rows:
        if int(row["exit_code"]) != 0:
            failures[row["task"]] += 1

    tasks = sorted({row["task"] for row in rows})
    values = [failures.get(task, 0) for task in tasks]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tasks, values, color="#b55d60")
    ax.set_ylabel("non-zero exit count")
    ax.set_title("Benchmark failures by task")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
