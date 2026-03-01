#!/usr/bin/env python3
"""Plot RFF approximation quality from leps_rff_quality.jsonl.

Generates:
    leps_rff_quality.pdf - MAE vs D_rff (energy + gradient, vs true + vs GP)

Usage:
    cargo run --release --example leps_rff_quality
    python scripts/plot_rff_quality.py
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except ImportError:
    print("matplotlib required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

# -- Theme ------------------------------------------------------------------

_FONT_FAMILY = "sans-serif"
for font in font_manager.findSystemFonts():
    if "Jost" in font:
        _FONT_FAMILY = "Jost"
        break

plt.rcParams.update(
    {
        "font.family": _FONT_FAMILY,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
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
SKY = "#1E88E5"
MAGENTA = "#D81B60"

OUTDIR = Path("scripts/figures/tutorial/output")


def load_data(path="leps_rff_quality.jsonl"):
    """Load JSONL into exact GP and RFF records."""
    exact = None
    rff_records = []

    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec["type"] == "exact_gp":
                exact = rec
            elif rec["type"] == "rff":
                rff_records.append(rec)

    return exact, rff_records


def plot_rff_quality(exact, rff_records):
    """Two-panel: energy MAE and gradient MAE vs D_rff."""
    d_vals = [r["d_rff"] for r in rff_records]
    e_vs_true = [r["energy_mae_vs_true"] for r in rff_records]
    g_vs_true = [r["gradient_mae_vs_true"] for r in rff_records]
    e_vs_gp = [r["energy_mae_vs_gp"] for r in rff_records]
    g_vs_gp = [r["gradient_mae_vs_gp"] for r in rff_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), sharey=False)

    # Energy panel
    ax1.semilogy(d_vals, e_vs_true, "o-", color=TEAL, label="RFF vs true surface")
    ax1.semilogy(d_vals, e_vs_gp, "s--", color=SKY, label="RFF vs exact GP")
    ax1.axhline(exact["energy_mae"], color=CORAL, linestyle=":", linewidth=1.2,
                label=f"Exact GP vs true ({exact['energy_mae']:.1e})")
    ax1.set_xlabel(r"$D_\mathrm{RFF}$")
    ax1.set_ylabel("Energy MAE")
    ax1.set_title("Energy")
    ax1.legend(fontsize=8, frameon=False)

    # Gradient panel
    ax2.semilogy(d_vals, g_vs_true, "o-", color=TEAL, label="RFF vs true surface")
    ax2.semilogy(d_vals, g_vs_gp, "s--", color=SKY, label="RFF vs exact GP")
    ax2.axhline(exact["gradient_mae"], color=CORAL, linestyle=":", linewidth=1.2,
                label=f"Exact GP vs true ({exact['gradient_mae']:.1e})")
    ax2.set_xlabel(r"$D_\mathrm{RFF}$")
    ax2.set_ylabel("Gradient MAE")
    ax2.set_title("Gradient")
    ax2.legend(fontsize=8, frameon=False)

    fig.suptitle("RFF Approximation Quality on LEPS Surface", fontsize=13)
    fig.tight_layout()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "leps_rff_quality.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'leps_rff_quality.pdf'}")


if __name__ == "__main__":
    path = "leps_rff_quality.jsonl"
    if not Path(path).exists():
        print(f"Run: cargo run --release --example leps_rff_quality", file=sys.stderr)
        sys.exit(1)

    exact, rff_records = load_data(path)
    plot_rff_quality(exact, rff_records)
