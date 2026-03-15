#!/usr/bin/env python3
"""Plot GP quality figures from mb_gp_quality.jsonl.

Generates:
    mb_gp_progression.pdf  2x2 panel: GP mean at N=5,15,21,30
    mb_variance.pdf        GP variance overlay with crosshatching
    mb_gp_error.pdf        GP prediction error (mean - true)

Usage:
    cargo run --release --example mb_gp_quality
    python scripts/plot_gp_quality.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _theme import CORAL, MAGENTA, SKY, TEAL, YELLOW, HAS_PARSERS, plt
from matplotlib.patches import Patch

if HAS_PARSERS:
    from chemparseplot.parse.chemgp_jsonl import parse_gp_quality_jsonl

OUTDIR = Path("scripts/figures/tutorial/output")


def load_data(path="mb_gp_quality.jsonl"):
    """Load the JSONL data into structured arrays."""
    meta = None
    minima = []
    saddles = []
    train_points = defaultdict(list)
    grids = defaultdict(list)

    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            t = rec["type"]
            if t == "grid_meta":
                meta = rec
            elif t == "minimum":
                minima.append(rec)
            elif t == "saddle":
                saddles.append(rec)
            elif t == "train_point":
                train_points[rec["n_train"]].append(rec)
            elif t == "grid":
                grids[rec["n_train"]].append(rec)

    return meta, minima, saddles, train_points, grids


def grid_to_arrays(grid_records, meta):
    """Convert grid records to numpy arrays."""
    nx, ny = meta["nx"], meta["ny"]
    true_e = np.zeros((ny, nx))
    gp_e = np.zeros((ny, nx))
    gp_var = np.zeros((ny, nx))
    X = np.zeros((ny, nx))
    Y = np.zeros((ny, nx))

    for rec in grid_records:
        ix, iy = rec["ix"], rec["iy"]
        true_e[iy, ix] = rec["true_e"]
        gp_e[iy, ix] = rec["gp_e"]
        gp_var[iy, ix] = rec["gp_var"]
        X[iy, ix] = rec["x"]
        Y[iy, ix] = rec["y"]

    return X, Y, true_e, gp_e, gp_var


def plot_gp_progression(meta, minima, saddles, train_points, grids):
    """2x2 panel: GP mean at N=5, 15, 21, 30 with training points."""
    n_trains = [3, 8, 15, 30]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5), sharex=True, sharey=True)

    levels = np.linspace(-200, 50, 30)

    for ax, n in zip(axes.flat, n_trains):
        if n not in grids:
            continue
        X, Y, true_e, gp_e, gp_var = grid_to_arrays(grids[n], meta)

        cs = ax.contourf(X, Y, gp_e, levels=levels, cmap="RdYlGn_r",
                         extend="both")
        ax.contour(X, Y, gp_e, levels=levels, colors="k", linewidths=0.2,
                   alpha=0.3)

        pts = train_points[n]
        tx = [p["x"] for p in pts]
        ty = [p["y"] for p in pts]
        ax.scatter(tx, ty, c="white", edgecolors=TEAL, s=25, linewidths=0.8,
                   zorder=5)

        ax.set_title(f"N = {n} training points", fontsize=11)

    for ax in axes[1, :]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")

    fig.colorbar(cs, ax=axes, label="GP predicted energy", shrink=0.85,
                 pad=0.02)
    fig.suptitle("GP Surrogate Progression on Muller-Brown", fontsize=13)
    fig.subplots_adjust(right=0.88, hspace=0.25, wspace=0.15)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "mb_gp_progression.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'mb_gp_progression.pdf'}")


def plot_variance(meta, minima, saddles, train_points, grids):
    """GP variance overlay with diagonal crosshatching on MB surface."""
    n = 15
    if n not in grids:
        print(f"No grid data for N={n}", file=sys.stderr)
        return

    X, Y, true_e, gp_e, gp_var = grid_to_arrays(grids[n], meta)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    levels = np.linspace(-200, 50, 30)
    ax.contourf(X, Y, true_e, levels=levels, cmap="RdYlGn_r", extend="both",
                alpha=0.7)
    ax.contour(X, Y, true_e, levels=levels, colors="k", linewidths=0.2,
               alpha=0.3)

    var_clip = np.clip(gp_var, 0, np.percentile(gp_var, 95))

    med_thresh = np.percentile(gp_var, 50)
    high_thresh = np.percentile(gp_var, 75)

    ax.contourf(X, Y, gp_var, levels=[med_thresh, high_thresh],
                colors="none", hatches=["//"], alpha=0.0)
    ax.contour(X, Y, gp_var, levels=[med_thresh], colors=[SKY],
               linewidths=1.0, linestyles="dashed")

    ax.contourf(X, Y, gp_var, levels=[high_thresh, gp_var.max() * 1.1],
                colors="none", hatches=["xxxx"], alpha=0.0)
    ax.contour(X, Y, gp_var, levels=[high_thresh], colors=[MAGENTA],
               linewidths=1.2, linestyles="solid")

    pts = train_points[n]
    tx = [p["x"] for p in pts]
    ty = [p["y"] for p in pts]
    ax.scatter(tx, ty, c="white", edgecolors=TEAL, s=35, linewidths=1.0,
               zorder=5, label="Training points")

    for m in minima:
        ax.plot(m["x"], m["y"], "*", color=YELLOW, markersize=12,
                markeredgecolor="k", markeredgewidth=0.5, zorder=6)
    for s in saddles:
        ax.plot(s["x"], s["y"], "D", color=CORAL, markersize=7,
                markeredgecolor="k", markeredgewidth=0.5, zorder=6)

    max_idx = np.unravel_index(np.argmax(gp_var), gp_var.shape)
    ax.plot(X[max_idx], Y[max_idx], "v", color=MAGENTA, markersize=10,
            markeredgecolor="k", markeredgewidth=0.5, zorder=6)

    legend_elements = [
        Patch(facecolor="none", edgecolor=SKY, linestyle="dashed",
              label=f"Medium variance (>{med_thresh:.0f})"),
        Patch(facecolor="none", edgecolor=MAGENTA,
              label=f"High variance (>{high_thresh:.0f})"),
        plt.Line2D([0], [0], marker="o", color="w", markeredgecolor=TEAL,
                   markersize=6, label="Training points"),
        plt.Line2D([0], [0], marker="*", color=YELLOW, markersize=10,
                   markeredgecolor="k", label="Minima"),
        plt.Line2D([0], [0], marker="D", color=CORAL, markersize=6,
                   markeredgecolor="k", label="Saddles"),
        plt.Line2D([0], [0], marker="v", color=MAGENTA, markersize=8,
                   markeredgecolor="k", label="Max variance"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"GP Variance on Muller-Brown (N={n} training points)")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "mb_variance.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'mb_variance.pdf'}")


def plot_gp_error(meta, minima, saddles, train_points, grids):
    """GP prediction error (mean - true) for N=21."""
    n = 15
    if n not in grids:
        print(f"No grid data for N={n}", file=sys.stderr)
        return

    X, Y, true_e, gp_e, gp_var = grid_to_arrays(grids[n], meta)
    error = gp_e - true_e

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    vmax = np.percentile(np.abs(error), 95)
    cs = ax.contourf(X, Y, error, levels=np.linspace(-vmax, vmax, 25),
                     cmap="RdBu_r", extend="both")
    ax.contour(X, Y, error, levels=[0], colors="k", linewidths=1.0)
    fig.colorbar(cs, ax=ax, label="GP error (predicted - true)", shrink=0.85)

    pts = train_points[n]
    tx = [p["x"] for p in pts]
    ty = [p["y"] for p in pts]
    ax.scatter(tx, ty, c="white", edgecolors=TEAL, s=35, linewidths=1.0,
               zorder=5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"GP Prediction Error (N={n})")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "mb_gp_error.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'mb_gp_error.pdf'}")


if __name__ == "__main__":
    if not Path("mb_gp_quality.jsonl").exists():
        print("Run: cargo run --release --example mb_gp_quality", file=sys.stderr)
        sys.exit(1)

    meta, minima, saddles, train_points, grids = load_data()
    plot_gp_progression(meta, minima, saddles, train_points, grids)
    plot_variance(meta, minima, saddles, train_points, grids)
    plot_gp_error(meta, minima, saddles, train_points, grids)
