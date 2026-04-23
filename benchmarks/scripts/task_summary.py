#!/usr/bin/env python3
"""Aggregate per-run metadata into a task-level summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = []
    for path_str in args.inputs:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as handle:
            runs.append(json.load(handle))

    payload = {
        "task": args.task,
        "n_runs": len(runs),
        "cases": sorted({run["case"] for run in runs}),
        "variants": sorted({run["variant"] for run in runs}),
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
