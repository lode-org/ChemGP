#!/usr/bin/env python3
"""Aggregate benchmark summary JSON files into a compact table."""

from __future__ import annotations

import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task_file in sorted(results_dir.glob("*/summary.json")):
        if task_file.parent.name == "summary":
            continue
        payload = load_summary(task_file)
        rows.append(
            {
                "task": payload["task"],
                "n_synthetic": len(payload.get("synthetic", [])),
                "n_literature": len(payload.get("literature", [])),
                "methods": ",".join(payload.get("methods", [])),
                "runtime_target_minutes": payload.get("runtime_target_minutes", 45),
            }
        )

    csv_path = summary_dir / "benchmark_table.csv"
    md_path = summary_dir / "benchmark_table.md"

    header = ["task", "n_synthetic", "n_literature", "methods", "runtime_target_minutes"]
    with csv_path.open("w", encoding="ascii") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in header) + "\n")

    with md_path.open("w", encoding="ascii") as handle:
        handle.write("| task | n_synthetic | n_literature | methods | runtime_target_minutes |\n")
        handle.write("|---|---:|---:|---|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['task']} | {row['n_synthetic']} | {row['n_literature']} | "
                f"{row['methods']} | {row['runtime_target_minutes']} |\n"
            )


if __name__ == "__main__":
    main()
