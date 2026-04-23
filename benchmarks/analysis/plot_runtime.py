#!/usr/bin/env python3
"""Plot runtime by benchmark run."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "results" / "summary" / "benchmark_table.csv"
    out_path = root / "results" / "summary" / "runtime.png"

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    labels = [f"{row['task']}:{row['case']}:{row['variant']}:{row['method']}" for row in rows]
    values = [float(row["runtime_seconds"]) for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.5), 4))
    ax.bar(range(len(rows)), values, color="#476c9b")
    ax.set_ylabel("runtime / s")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_title("ChemGP benchmark runtime")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
