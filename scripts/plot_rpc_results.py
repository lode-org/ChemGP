#!/usr/bin/env python3
"""Plot RPC-based optimization results (PET-MAD minimize, HCN NEB).

Generates:
    petmad_minimize_convergence.pdf  GP vs direct GD convergence
    hcn_neb_convergence.pdf          NEB vs GP-NEB convergence

Usage:
    cargo run --release --features rgpot --example petmad_minimize
    cargo run --release --features rgpot --example hcn_neb
    python scripts/plot_rpc_results.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _theme import CORAL, SKY, TEAL, HAS_PARSERS, plt

OUTDIR = Path("scripts/figures/tutorial/output")


def plot_petmad_minimize(path="petmad_minimize_comparison.jsonl"):
    """GP vs direct minimize convergence on PET-MAD system100."""
    if not Path(path).exists():
        print(f"No {path}, skipping petmad minimize plot", file=sys.stderr)
        return

    groups = defaultdict(list)
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if not rec.get("summary"):
                groups[rec["method"]].append(rec)

    palette = {"gp_minimize": TEAL, "direct_lbfgs": CORAL, "direct_minimize": CORAL}
    labels = {"gp_minimize": "GP Minimize", "direct_lbfgs": "Direct L-BFGS", "direct_minimize": "Direct GD"}
    order = {"direct_lbfgs": 0, "direct_minimize": 0, "gp_minimize": 1}

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))
    for method in sorted(groups, key=lambda m: order.get(m, 0)):
        records = groups[method]
        # Truncate at convergence (first point below threshold)
        conv_tol = 0.01
        truncated = []
        for r in records:
            truncated.append(r)
            # Use max_fatom if available, otherwise max_force
            force = r.get("max_fatom", r.get("max_force", float('inf')))
            if force < conv_tol:
                break
        if not truncated:
            continue
        calls = [r["oracle_calls"] for r in truncated]
        # Use max_fatom if available, otherwise compute from max_force
        if "max_fatom" in truncated[0]:
            max_fatom = [r["max_fatom"] for r in truncated]
        elif "max_force" in truncated[0]:
            max_fatom = [r["max_force"] for r in truncated]
        else:
            continue
        ax.semilogy(
            calls, max_fatom,
            label=labels.get(method, method),
            color=palette.get(method, "#333"),
            marker="o" if "gp" in method else None,
            markersize=4 if "gp" in method else 0,
            zorder=order.get(method, 0) + 2,
        )

    # Add convergence threshold line
    ax.axhline(0.01, color="k", linewidth=0.5, linestyle="--", alpha=0.5, label="conv_tol")

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("Max force per atom (eV/Å)")
    ax.set_title("PET-MAD Minimization (system100)")
    ax.legend(frameon=False)
    fig.tight_layout()


def plot_hcn_neb(path="hcn_neb_comparison.jsonl"):
    """NEB convergence and energy profile on HCN -> HNC."""
    if not Path(path).exists():
        print(f"No {path}, skipping HCN NEB plot", file=sys.stderr)
        return

    convergence = []
    path_energies = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("summary"):
                continue
            if rec.get("type") == "path_energy":
                path_energies.append(rec)
            elif "method" in rec:
                convergence.append(rec)

    palette = {"neb": CORAL, "gp_neb_aie": TEAL, "gp_neb_oie": SKY, "gp_neb_oie_naive": "#999"}
    labels = {"neb": "Standard NEB", "gp_neb_aie": "GP-NEB AIE", "gp_neb_oie": "GP-NEB OIE", "gp_neb_oie_naive": "GP-NEB OIE (naive)"}

    has_path = len(path_energies) > 0
    ncols = 2 if has_path else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 3.5))
    if ncols == 1:
        axes = [axes]

    # Panel 1: convergence
    ax = axes[0]
    groups = defaultdict(list)
    for rec in convergence:
        groups[rec["method"]].append(rec)

    for method in sorted(groups, key=lambda m: list(palette.keys()).index(m) if m in palette else 99):
        records = groups[method]
        # Truncate at convergence (first point below threshold)
        conv_tol = 0.1
        truncated = []
        for r in records:
            truncated.append(r)
            force = r.get("ci_force", r.get("max_force", float('inf')))
            if force < conv_tol:
                break
        if not truncated:
            continue
        calls = [r["oracle_calls"] for r in truncated]
        # Use ci_force when available (for climbing image), otherwise max_force
        forces = [r.get("ci_force", r.get("max_force", float('nan'))) for r in truncated]
        ax.semilogy(
            calls, forces,
            label=labels.get(method, method),
            color=palette.get(method, "#333"),
            marker="o", markersize=3,
        )

    # Add convergence threshold line at 0.1 (tutorial standard)
    ax.axhline(0.1, color="k", linewidth=0.5, linestyle="--", alpha=0.5, label="conv_tol")

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("CI force (eV/Å)")
    ax.set_title("GP-NEB Convergence")
    ax.legend(frameon=False, fontsize=8)

    # Panel 2: energy profile
    if has_path:
        ax2 = axes[1]
        # Group path energies by method
        path_groups = defaultdict(list)
        for rec in path_energies:
            path_groups[rec["method"]].append(rec)
        for method in sorted(path_groups, key=lambda m: list(palette.keys()).index(m) if m in palette else 99):
            records = path_groups[method]
            images = [r["image"] for r in records]
            energies = [r["energy"] for r in records]
            # Shift to relative energy
            e_ref = energies[0]
            rel_e = [e - e_ref for e in energies]
            ax2.plot(images, rel_e, "o-", color=palette.get(method, TEAL), markersize=5, label=labels.get(method, method))
        ax2.set_xlabel("Image index")
        ax2.set_ylabel("Relative energy (eV)")
        ax2.set_title("MEP Energy Profile")
        ax2.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.3)
        ax2.legend(frameon=False, fontsize=8)

    fig.suptitle("HCN -> HNC Isomerization (PET-MAD)", fontsize=13)
    fig.tight_layout()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "hcn_neb_convergence.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'hcn_neb_convergence.pdf'}")


if __name__ == "__main__":
    plot_petmad_minimize()
    plot_hcn_neb()
