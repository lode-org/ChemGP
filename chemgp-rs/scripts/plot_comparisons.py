#!/usr/bin/env python3
"""Plot comparison figures from JSONL output of chemgp-core examples.

Usage:
    python plot_comparisons.py

Reads: leps_minimize_comparison.jsonl, leps_neb_comparison.jsonl, leps_dimer_comparison.jsonl
Writes: fig_minimize.pdf, fig_neb.pdf, fig_dimer.pdf
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

PALETTE = {
    "gp_minimize": "#004D40",
    "direct_minimize": "#FF655D",
    "neb": "#FF655D",
    "gp_neb_aie": "#F1DB4B",
    "gp_neb_oie": "#004D40",
    "gp_dimer": "#FF655D",
    "otgpd": "#004D40",
}

LABELS = {
    "gp_minimize": "GP Minimize",
    "direct_minimize": "Direct GD",
    "neb": "Standard NEB",
    "gp_neb_aie": "GP-NEB AIE",
    "gp_neb_oie": "GP-NEB OIE",
    "gp_dimer": "GP-Dimer",
    "otgpd": "OTGPD",
}


def load_jsonl(path):
    """Load JSONL, grouping records by method."""
    groups = defaultdict(list)
    summary = None
    if not Path(path).exists():
        return groups, summary
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("summary"):
                summary = rec
            else:
                groups[rec["method"]].append(rec)
    return groups, summary


def plot_minimize():
    groups, summary = load_jsonl("leps_minimize_comparison.jsonl")
    if not groups:
        print("No minimize data found", file=sys.stderr)
        return

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    for method, records in groups.items():
        calls = [r["oracle_calls"] for r in records]
        energies = [r["energy"] for r in records]
        ax.plot(calls, energies, label=LABELS.get(method, method),
                color=PALETTE.get(method, "#333"), linewidth=1.5)

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("LEPS Minimization")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_minimize.pdf", bbox_inches="tight")
    print("Wrote fig_minimize.pdf")


def plot_neb():
    groups, summary = load_jsonl("leps_neb_comparison.jsonl")
    if not groups:
        print("No NEB data found", file=sys.stderr)
        return

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    for method, records in groups.items():
        calls = [r["oracle_calls"] for r in records]
        forces = [r["max_force"] for r in records]
        ax.plot(calls, forces, label=LABELS.get(method, method),
                color=PALETTE.get(method, "#333"), linewidth=1.5)

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("Max |F| (eV/A)")
    ax.set_yscale("log")
    ax.set_title("LEPS NEB Convergence")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_neb.pdf", bbox_inches="tight")
    print("Wrote fig_neb.pdf")


def plot_dimer():
    groups, summary = load_jsonl("leps_dimer_comparison.jsonl")
    if not groups:
        print("No dimer data found", file=sys.stderr)
        return

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    for method, records in groups.items():
        calls = [r["oracle_calls"] for r in records]
        forces = [r["force"] for r in records]
        ax.plot(calls, forces, label=LABELS.get(method, method),
                color=PALETTE.get(method, "#333"), linewidth=1.5, marker="o", markersize=3)

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("|F_trans| (eV/A)")
    ax.set_title("LEPS Saddle Point Search")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_dimer.pdf", bbox_inches="tight")
    print("Wrote fig_dimer.pdf")


if __name__ == "__main__":
    plot_minimize()
    plot_neb()
    plot_dimer()
