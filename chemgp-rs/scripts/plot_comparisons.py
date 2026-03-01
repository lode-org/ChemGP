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
    from matplotlib import font_manager
except ImportError:
    print("matplotlib required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

# Try to use Jost font, fall back to sans-serif
_FONT_FAMILY = "sans-serif"
for font in font_manager.findSystemFonts():
    if "Jost" in font:
        _FONT_FAMILY = "Jost"
        break

plt.rcParams.update(
    {
        "font.family": _FONT_FAMILY,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

TEAL = "#004D40"
CORAL = "#FF655D"
YELLOW = "#F1DB4B"

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

# Plot order so GP methods draw on top
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

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))
    for method in sorted(groups, key=lambda m: ORDER.get(m, 0)):
        records = groups[method]
        calls = [r["oracle_calls"] for r in records]
        energies = [r["energy"] for r in records]
        ax.plot(
            calls,
            energies,
            label=LABELS.get(method, method),
            color=PALETTE.get(method, "#333"),
            marker="o" if "gp" in method else None,
            markersize=4 if "gp" in method else 0,
            zorder=ORDER.get(method, 0) + 2,
        )

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("LEPS Minimization")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("fig_minimize.pdf")
    plt.close(fig)
    print("Wrote fig_minimize.pdf")


def plot_neb():
    groups, summary = load_jsonl("leps_neb_comparison.jsonl")
    if not groups:
        print("No NEB data found", file=sys.stderr)
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))
    for method in sorted(groups, key=lambda m: ORDER.get(m, 0)):
        records = groups[method]
        calls = [r["oracle_calls"] for r in records]
        forces = [r["max_force"] for r in records]
        lw = 1.5 if method == "neb" else 1.8
        ax.plot(
            calls,
            forces,
            label=LABELS.get(method, method),
            color=PALETTE.get(method, "#333"),
            linewidth=lw,
            zorder=ORDER.get(method, 0) + 2,
        )

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("max |F| (eV/A)")
    ax.set_yscale("log")
    ax.set_title("LEPS NEB Convergence")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("fig_neb.pdf")
    plt.close(fig)
    print("Wrote fig_neb.pdf")


def plot_dimer():
    groups, summary = load_jsonl("leps_dimer_comparison.jsonl")
    if not groups:
        print("No dimer data found", file=sys.stderr)
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))
    for method in sorted(groups, key=lambda m: ORDER.get(m, 0)):
        records = groups[method]
        calls = [r["oracle_calls"] for r in records]
        forces = [r["force"] for r in records]
        ax.plot(
            calls,
            forces,
            label=LABELS.get(method, method),
            color=PALETTE.get(method, "#333"),
            marker="o",
            markersize=4,
            zorder=ORDER.get(method, 0) + 2,
        )

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("|F_trans| (eV/A)")
    ax.set_yscale("log")
    ax.set_title("LEPS Saddle Point Search")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("fig_dimer.pdf")
    plt.close(fig)
    print("Wrote fig_dimer.pdf")


if __name__ == "__main__":
    plot_minimize()
    plot_neb()
    plot_dimer()
