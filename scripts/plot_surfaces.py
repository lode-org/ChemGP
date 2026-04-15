#!/usr/bin/env python3
"""Plot potential energy surfaces for tutorial documentation.

Generates:
    mb_pes.pdf              Muller-Brown PES with stationary points
    leps_contour.pdf        LEPS 2D contour (r_AB vs r_BC)
    mb_minimize_convergence.pdf  MB minimization convergence (GP vs GD)

Usage:
    python scripts/plot_surfaces.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _theme import CORAL, TEAL, YELLOW, plt

OUTDIR = Path("scripts/figures/tutorial/output")

# -- Muller-Brown potential -------------------------------------------------

_MB_A = [-200, -100, -170, 15]
_MB_a = [-1, -1, -6.5, 0.7]
_MB_b = [0, 0, 11, 0.6]
_MB_c = [-10, -10, -6.5, 0.7]
_MB_x0 = [1, 0, -0.5, -1]
_MB_y0 = [0, 0.5, 1.5, 1]

MB_MINIMA = [(-0.558, 1.442), (0.623, 0.028), (-0.050, 0.467)]
MB_SADDLES = [(-0.822, 0.624), (0.212, 0.293)]


def muller_brown(x, y):
    v = 0.0
    for i in range(4):
        dx = x - _MB_x0[i]
        dy = y - _MB_y0[i]
        v += _MB_A[i] * np.exp(
            _MB_a[i] * dx * dx + _MB_b[i] * dx * dy + _MB_c[i] * dy * dy
        )
    return v


# -- LEPS potential (2D projection) ----------------------------------------

_LEPS_DE = [4.746, 4.746, 3.445]
_LEPS_BETA = [1.942, 1.942, 1.942]
_LEPS_RE = [0.7414, 0.7414, 0.7414]
_LEPS_K = [0.05, 0.30, 0.05]


def _leps_q(r, i):
    d, b, re = _LEPS_DE[i], _LEPS_BETA[i], _LEPS_RE[i]
    x = np.exp(-b * (r - re))
    return d * (1.5 * x * x - x) / (1 + _LEPS_K[i])


def _leps_j(r, i):
    d, b, re = _LEPS_DE[i], _LEPS_BETA[i], _LEPS_RE[i]
    x = np.exp(-b * (r - re))
    return d * (x * x - 6.0 * x) / (4.0 * (1 + _LEPS_K[i]))


def leps_energy_2d(r_ab, r_bc):
    r_ac = r_ab + r_bc
    q_ab, q_bc, q_ac = _leps_q(r_ab, 0), _leps_q(r_bc, 1), _leps_q(r_ac, 2)
    j_ab, j_bc, j_ac = _leps_j(r_ab, 0), _leps_j(r_bc, 1), _leps_j(r_ac, 2)
    coulomb = q_ab + q_bc + q_ac
    exchange = np.sqrt(
        (j_ab - j_bc) ** 2 + (j_bc - j_ac) ** 2 + (j_ab - j_ac) ** 2
    )
    return coulomb - exchange


# -- Plot functions ---------------------------------------------------------


def plot_mb_pes():
    nx, ny = 200, 200
    x = np.linspace(-1.5, 1.2, nx)
    y = np.linspace(-0.3, 2.0, ny)
    X, Y = np.meshgrid(x, y)
    Z = muller_brown(X, Y)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))
    levels = np.linspace(-200, 50, 30)
    cs = ax.contourf(X, Y, Z, levels=levels, cmap="RdYlGn_r", extend="both")
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
    fig.colorbar(cs, ax=ax, label="Energy", shrink=0.85)

    for i, (mx, my) in enumerate(MB_MINIMA):
        label = f"Min {chr(65 + i)}"
        ax.plot(mx, my, "*", color=YELLOW, markersize=12, markeredgecolor="k",
                markeredgewidth=0.5, zorder=5)
        ax.annotate(label, (mx, my), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, color=TEAL, fontweight="bold")

    for i, (sx, sy) in enumerate(MB_SADDLES):
        label = f"S{i + 1}"
        ax.plot(sx, sy, "D", color=CORAL, markersize=7, markeredgecolor="k",
                markeredgewidth=0.5, zorder=5)
        ax.annotate(label, (sx, sy), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, color=CORAL, fontweight="bold")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Muller-Brown Potential Energy Surface")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "mb_pes.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'mb_pes.pdf'}")


def plot_leps_contour():
    nr = 200
    r_ab = np.linspace(0.5, 4.0, nr)
    r_bc = np.linspace(0.5, 4.0, nr)
    R_AB, R_BC = np.meshgrid(r_ab, r_bc)
    Z = leps_energy_2d(R_AB, R_BC)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    levels = np.linspace(-5.0, -2.0, 30)
    cs = ax.contourf(R_AB, R_BC, Z, levels=levels, cmap="RdYlGn_r",
                     extend="both")
    ax.contour(R_AB, R_BC, Z, levels=levels, colors="k", linewidths=0.3,
               alpha=0.4)
    fig.colorbar(cs, ax=ax, label="Energy (eV)", shrink=0.85)

    ax.annotate("H + H$_2$\n(reactant)", (3.0, 0.8), fontsize=10,
                color="white", fontweight="bold", ha="center")
    ax.annotate("H$_2$ + H\n(product)", (0.8, 3.0), fontsize=10,
                color="white", fontweight="bold", ha="center")
    ax.annotate("TS", (1.1, 1.1), fontsize=10,
                color=YELLOW, fontweight="bold", ha="center")

    ax.set_xlabel(r"$r_{AB}$ ($\AA$)")
    ax.set_ylabel(r"$r_{BC}$ ($\AA$)")
    ax.set_title("LEPS Surface (Collinear H + H$_2$)")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "leps_contour.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'leps_contour.pdf'}")


def plot_mb_minimize():
    path = "mb_minimize_comparison.jsonl"
    if not Path(path).exists():
        print(f"No {path} found, skipping MB minimize plot", file=sys.stderr)
        return

    groups = defaultdict(list)
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if not rec.get("summary"):
                groups[rec["method"]].append(rec)

    palette = {"gp_minimize": TEAL, "direct_minimize": CORAL}
    labels = {"gp_minimize": "GP Minimize", "direct_minimize": "L-BFGS"}
    order = {"direct_minimize": 0, "gp_minimize": 1}

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))
    for method in sorted(groups, key=lambda m: order.get(m, 0)):
        records = groups[method]
        calls = [r["oracle_calls"] for r in records]
        energies = [r["energy"] for r in records]
        ax.plot(
            calls, energies,
            label=labels.get(method, method),
            color=palette.get(method, "#333"),
            marker="o" if "gp" in method else None,
            markersize=4 if "gp" in method else 0,
            zorder=order.get(method, 0) + 2,
        )

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("Energy")
    ax.set_title("Muller-Brown Minimization")
    ax.legend(frameon=False)
    fig.tight_layout()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "mb_minimize_convergence.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'mb_minimize_convergence.pdf'}")


if __name__ == "__main__":
    plot_mb_pes()
    plot_leps_contour()
    plot_mb_minimize()
