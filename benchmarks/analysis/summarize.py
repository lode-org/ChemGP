#!/usr/bin/env python3
"""Aggregate benchmark run metadata and JSONL traces into compact tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize_run(meta: dict) -> list[dict]:
    jsonl_path = Path(meta["output_jsonl"])
    records = load_jsonl(jsonl_path)
    by_method: dict[str, list[dict]] = {}
    summary_record = None

    for record in records:
        if record.get("summary"):
            summary_record = record
            continue
        method = record.get("method")
        if method is None:
            continue
        by_method.setdefault(method, []).append(record)

    conv_tol = summary_record.get("conv_tol") if summary_record else None

    def method_summary(method: str) -> tuple[str, str]:
        if not summary_record:
            return ("", "")
        if meta["task"] == "minimize":
            if method == "classical":
                max_f = summary_record.get("direct_max_fatom")
                if max_f is not None and conv_tol is not None:
                    return (str(bool(max_f <= conv_tol)), str(max_f))
            elif method == summary_record.get("gp_method"):
                return (
                    str(summary_record.get("gp_converged", "")),
                    str(summary_record.get("gp_energy", "")),
                )
        if meta["task"] == "dimer":
            key = {"classical": "standard", "otgpd": "otgpd", "chemgp": "gpdimer", "physical_prior": "gpdimer", "adaptive_prior": "gpdimer", "recycled_local_pes": "gpdimer"}.get(method)
            if key:
                return (
                    str(summary_record.get(f"{key}_converged", "")),
                    str(summary_record.get(f"{key}_force", "")),
                )
        if meta["task"] == "neb":
            key = {"neb": "neb", "aie": "aie", "oie": "oie"}.get(method)
            if key:
                force_value = summary_record.get(f"{key}_ci_force")
                if force_value is None:
                    force_value = summary_record.get(f"{key}_max_force", "")
                return (
                    str(summary_record.get(f"{key}_converged", "")),
                    str(force_value),
                )
        return ("", "")

    rows = []
    for method, method_records in sorted(by_method.items()):
        method_records.sort(key=lambda rec: rec.get("step", -1))
        last = method_records[-1]
        oracle_calls = max((rec.get("oracle_calls", 0) for rec in method_records), default=0)
        convergence_basis = ""
        primary_force = ""
        if meta["task"] == "neb":
            if last.get("ci_force", "") != "":
                convergence_basis = "ci_force"
                primary_force = last.get("ci_force", "")
            elif last.get("max_force", "") != "":
                convergence_basis = "max_force"
                primary_force = last.get("max_force", "")
        elif last.get("force", "") != "":
            convergence_basis = "force"
            primary_force = last.get("force", "")
        elif last.get("max_fatom", "") != "":
            convergence_basis = "max_fatom"
            primary_force = last.get("max_fatom", "")
        converged, summary_value = method_summary(method)
        rows.append(
            {
                "task": meta["task"],
                "case": meta["case"],
                "group": meta["group"],
                "variant": meta["variant"],
                "method": method,
                "runtime_seconds": f"{meta['runtime_seconds']:.3f}",
                "exit_code": meta["exit_code"],
                "oracle_calls": oracle_calls,
                "energy": last.get("energy", ""),
                "force": last.get("force", ""),
                "max_force": last.get("max_force", ""),
                "max_fatom": last.get("max_fatom", ""),
                "ci_force": last.get("ci_force", ""),
                "convergence_basis": convergence_basis,
                "primary_force": primary_force,
                "converged": converged,
                "summary_value": summary_value,
                "summary_keys": ",".join(sorted(summary_record.keys())) if summary_record else "",
            }
        )
    if not rows:
        rows.append(
            {
                "task": meta["task"],
                "case": meta["case"],
                "group": meta["group"],
                "variant": meta["variant"],
                "method": "none",
                "runtime_seconds": f"{meta['runtime_seconds']:.3f}",
                "exit_code": meta["exit_code"],
                "oracle_calls": 0,
                "energy": "",
                "force": "",
                "max_force": "",
                "max_fatom": "",
                "ci_force": "",
                "convergence_basis": "",
                "primary_force": "",
                "converged": "",
                "summary_value": "",
                "summary_keys": ",".join(sorted(summary_record.keys())) if summary_record else "",
            }
        )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task_summary in sorted(results_dir.glob("*/summary.json")):
        if task_summary.parent.name == "summary":
            continue
        payload = load_json(task_summary)
        for run_meta in payload.get("runs", []):
            rows.extend(summarize_run(run_meta))

    header = [
        "task",
        "case",
        "group",
        "variant",
        "method",
        "runtime_seconds",
        "exit_code",
        "oracle_calls",
        "energy",
        "force",
        "max_force",
        "max_fatom",
        "ci_force",
        "convergence_basis",
        "primary_force",
        "converged",
        "summary_value",
        "summary_keys",
    ]

    csv_path = summary_dir / "benchmark_table.csv"
    md_path = summary_dir / "benchmark_table.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(header) + " |\n")
        handle.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row[key]) for key in header) + " |\n")


if __name__ == "__main__":
    main()
