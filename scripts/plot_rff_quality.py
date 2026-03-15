#!/usr/bin/env python3
"""Plot RFF approximation quality from leps_rff_quality.jsonl.

Generates:
    leps_rff_quality.pdf - MAE vs D_rff (energy + gradient, vs true + vs GP)

Usage:
    cargo run --release --example leps_rff_quality
    python scripts/plot_rff_quality.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _theme import CORAL, SKY, TEAL, HAS_PARSERS, plt

if HAS_PARSERS:
    from chemparseplot.parse.chemgp_jsonl import parse_rff_quality_jsonl
else:
    import json

OUTDIR = Path("scripts/figures/tutorial/output")


def load_data_fallback(path):
    """Fallback loader without rgpycrumbs."""
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


def plot_rff_quality(path):
    """Two-panel: energy MAE and gradient MAE vs D_rff."""
    if HAS_PARSERS:
        data = parse_rff_quality_jsonl(path)
        d_vals = data.d_rff_values
        e_vs_true = data.energy_mae_vs_true
        g_vs_true = data.gradient_mae_vs_true
        e_vs_gp = data.energy_mae_vs_gp
        g_vs_gp = data.gradient_mae_vs_gp
        exact_e = data.exact_energy_mae
        exact_g = data.exact_gradient_mae
    else:
        exact, rff_records = load_data_fallback(path)
        d_vals = [r["d_rff"] for r in rff_records]
        e_vs_true = [r["energy_mae_vs_true"] for r in rff_records]
        g_vs_true = [r["gradient_mae_vs_true"] for r in rff_records]
        e_vs_gp = [r["energy_mae_vs_gp"] for r in rff_records]
        g_vs_gp = [r["gradient_mae_vs_gp"] for r in rff_records]
        exact_e = exact["energy_mae"]
        exact_g = exact["gradient_mae"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), sharey=False)

    ax1.semilogy(d_vals, e_vs_true, "o-", color=TEAL, label="RFF vs true surface")
    ax1.semilogy(d_vals, e_vs_gp, "s--", color=SKY, label="RFF vs exact GP")
    ax1.axhline(exact_e, color=CORAL, linestyle=":", linewidth=1.2,
                label=f"Exact GP vs true ({exact_e:.1e})")
    ax1.set_xlabel(r"$D_\mathrm{RFF}$")
    ax1.set_ylabel("Energy MAE")
    ax1.set_title("Energy")
    ax1.legend(fontsize=8, frameon=False)

    ax2.semilogy(d_vals, g_vs_true, "o-", color=TEAL, label="RFF vs true surface")
    ax2.semilogy(d_vals, g_vs_gp, "s--", color=SKY, label="RFF vs exact GP")
    ax2.axhline(exact_g, color=CORAL, linestyle=":", linewidth=1.2,
                label=f"Exact GP vs true ({exact_g:.1e})")
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
        print("Run: cargo run --release --example leps_rff_quality", file=sys.stderr)
        sys.exit(1)

    plot_rff_quality(path)
