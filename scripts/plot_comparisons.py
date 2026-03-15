#!/usr/bin/env python3
"""Plot comparison figures from JSONL output of chemgp-core examples.

Usage:
    python scripts/plot_comparisons.py

Reads: leps_minimize_comparison.jsonl, leps_neb_comparison.jsonl, leps_dimer_comparison.jsonl
Writes: leps_minimize_convergence.pdf, leps_neb_convergence.pdf, leps_dimer_convergence.pdf
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Shared RUHI theme
sys.path.insert(0, str(Path(__file__).parent))
from _theme import CORAL, TEAL, YELLOW, HAS_PARSERS, plt

if HAS_PARSERS:
    from chemparseplot.parse.chemgp_jsonl import parse_comparison_jsonl

OUTDIR = Path("scripts/figures/tutorial/output")

PALETTE = {
    "gp_minimize": TEAL,
    "direct_minimize": CORAL,
    "neb": CORAL,
    "gp_neb_aie": YELLOW,
    "gp_neb_oie": TEAL,
    "standard_dimer": YELLOW,
    "gp_dimer": CORAL,
    "otgpd": TEAL,
}

LABELS = {
    "gp_minimize": "GP Minimize",
    "direct_minimize": "Direct GD",
    "neb": "Standard NEB",
    "gp_neb_aie": "GP-NEB AIE",
    "gp_neb_oie": "GP-NEB OIE",
    "standard_dimer": "Standard Dimer",
    "gp_dimer": "GP-Dimer",
    "otgpd": "OTGPD",
}

ORDER = {
    "direct_minimize": 0,
    "gp_minimize": 1,
    "neb": 0,
    "gp_neb_aie": 1,
    "gp_neb_oie": 2,
    "standard_dimer": 0,
    "gp_dimer": 1,
    "otgpd": 2,
}


def load_jsonl(path):
    """Load JSONL, grouping records by method (fallback without rgpycrumbs)."""
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


def _plot_from_groups(groups, ax, x_key, y_key):
    """Plot traces from grouped records."""
    for method in sorted(groups, key=lambda m: ORDER.get(m, 0)):
        records = groups[method]
        xs = [r[x_key] for r in records]
        ys = [r[y_key] for r in records]
        ax.plot(
            xs, ys,
            label=LABELS.get(method, method),
            color=PALETTE.get(method, "#333"),
            marker="o" if "gp" in method else None,
            markersize=4 if "gp" in method else 0,
            zorder=ORDER.get(method, 0) + 2,
        )


def _plot_from_data(data, ax, y_attr):
    """Plot traces from parsed ComparisonData."""
    for method in sorted(data.traces, key=lambda m: ORDER.get(m, 0)):
        trace = data.traces[method]
        ys = trace.energies if y_attr == "energies" else trace.forces
        if ys is None:
            continue
        ax.plot(
            trace.oracle_calls, ys,
            label=LABELS.get(method, method),
            color=PALETTE.get(method, "#333"),
            marker="o" if "gp" in method else None,
            markersize=4 if "gp" in method else 0,
            zorder=ORDER.get(method, 0) + 2,
        )


def plot_minimize():
    path = "leps_minimize_comparison.jsonl"
    if not Path(path).exists():
        print("No minimize data found", file=sys.stderr)
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))
    if HAS_PARSERS:
        data = parse_comparison_jsonl(path)
        _plot_from_data(data, ax, "energies")
    else:
        groups, _ = load_jsonl(path)
        _plot_from_groups(groups, ax, "oracle_calls", "energy")

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("LEPS Minimization")
    ax.legend(frameon=False)
    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "leps_minimize_convergence.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'leps_minimize_convergence.pdf'}")


def plot_neb():
    path = "leps_neb_comparison.jsonl"
    if not Path(path).exists():
        print("No NEB data found", file=sys.stderr)
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))
    if HAS_PARSERS:
        data = parse_comparison_jsonl(path)
        _plot_from_data(data, ax, "forces")
    else:
        groups, _ = load_jsonl(path)
        _plot_from_groups(groups, ax, "oracle_calls", "max_force")

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("max |F| (eV/A)")
    ax.set_yscale("log")
    ax.set_title("LEPS NEB Convergence")
    ax.legend(frameon=False)
    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "leps_neb_convergence.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'leps_neb_convergence.pdf'}")


def plot_dimer():
    path = "leps_dimer_comparison.jsonl"
    if not Path(path).exists():
        print("No dimer data found", file=sys.stderr)
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))
    if HAS_PARSERS:
        data = parse_comparison_jsonl(path)
        _plot_from_data(data, ax, "forces")
    else:
        groups, _ = load_jsonl(path)
        _plot_from_groups(groups, ax, "oracle_calls", "force")

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("|F_trans| (eV/A)")
    ax.set_yscale("log")
    ax.set_title("LEPS Saddle Point Search")
    ax.legend(frameon=False)
    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "leps_dimer_convergence.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'leps_dimer_convergence.pdf'}")


if __name__ == "__main__":
    plot_minimize()
    plot_neb()
    plot_dimer()
