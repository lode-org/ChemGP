#!/usr/bin/env python3
"""Plot oracle calls by method for the collected benchmark runs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "results" / "summary" / "benchmark_table.csv"
    out_path = root / "results" / "summary" / "convergence.png"

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    labels = [f"{row['case']}:{row['method']}" for row in rows]
    values = [float(row["oracle_calls"]) for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.45), 4))
    ax.bar(range(len(rows)), values, color="#4a8f67")
    ax.set_ylabel("oracle calls")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_title("ChemGP benchmark oracle-call summary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
