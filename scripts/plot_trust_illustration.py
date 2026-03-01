#!/usr/bin/env python3
"""Trust region illustration on Muller-Brown 1D slice.

Shows GP prediction along a 1D slice with training data clustered in one
region. Highlights the trust boundary beyond which GP predictions diverge.

Generates:
    mb_trust_illustration.pdf

Usage:
    python scripts/plot_trust_illustration.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _theme import CORAL, TEAL, plt

OUTDIR = Path("scripts/figures/tutorial/output")

# -- Muller-Brown potential (1D slice at y=0.5) --------------------------------

_MB_A = [-200, -100, -170, 15]
_MB_a = [-1, -1, -6.5, 0.7]
_MB_b = [0, 0, 11, 0.6]
_MB_c = [-10, -10, -6.5, 0.7]
_MB_x0 = [1, 0, -0.5, -1]
_MB_y0 = [0, 0.5, 1.5, 1]


def muller_brown_1d(x, y=0.5):
    v = 0.0
    for i in range(4):
        dx = x - _MB_x0[i]
        dy = y - _MB_y0[i]
        v += _MB_A[i] * np.exp(
            _MB_a[i] * dx * dx + _MB_b[i] * dx * dy + _MB_c[i] * dy * dy
        )
    return v


def rbf_kernel(x1, x2, ls=0.15, sigma2=5000.0):
    d = x1[:, None] - x2[None, :]
    return sigma2 * np.exp(-0.5 * d**2 / ls**2)


def gp_predict(x_train, y_train, x_test, ls=0.15, sigma2=5000.0, noise=1e-2):
    K = rbf_kernel(x_train, x_train, ls, sigma2) + noise * np.eye(len(x_train))
    Ks = rbf_kernel(x_train, x_test, ls, sigma2)
    Kss = rbf_kernel(x_test, x_test, ls, sigma2)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    var = np.diag(Kss) - np.sum(v**2, axis=0)
    var = np.maximum(var, 0)

    return mu, np.sqrt(var)


def plot_trust_illustration():
    np.random.seed(42)
    x_train = np.array([-0.25, -0.1, 0.0, 0.05, 0.15, 0.25, 0.35, -0.15, 0.1, 0.2])
    y_train = np.array([muller_brown_1d(x) for x in x_train])

    x_test = np.linspace(-1.5, 1.2, 400)
    y_true = np.array([muller_brown_1d(x) for x in x_test])

    mu, sigma = gp_predict(x_train, y_train, x_test)

    trust_radius = 0.4

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    ax.plot(x_test, y_true, color="k", linewidth=1.0, alpha=0.5, label="True surface")
    ax.plot(x_test, mu, color=TEAL, linewidth=1.5, label="GP prediction")
    ax.fill_between(
        x_test, mu - 2 * sigma, mu + 2 * sigma,
        alpha=0.15, color=TEAL, label="GP 2-sigma"
    )

    trust_left = x_train.min() - trust_radius
    trust_right = x_train.max() + trust_radius

    ax.axvspan(-1.5, trust_left, alpha=0.08, color=CORAL, zorder=0)
    ax.axvspan(trust_right, 1.2, alpha=0.08, color=CORAL, zorder=0)
    ax.axvline(trust_left, color=CORAL, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(trust_right, color=CORAL, linestyle="--", linewidth=1.0, alpha=0.7)

    ax.scatter(
        x_train, y_train, c="white", edgecolors=TEAL, s=40, linewidths=1.0,
        zorder=5, label="Training points"
    )

    ax.annotate(
        "outside trust\n(unreliable)",
        xy=(-1.1, mu[np.argmin(np.abs(x_test + 1.1))]),
        fontsize=9, color=CORAL, ha="center", fontweight="bold",
    )
    ax.annotate(
        "outside trust\n(unreliable)",
        xy=(1.0, mu[np.argmin(np.abs(x_test - 1.0))]),
        fontsize=9, color=CORAL, ha="center", fontweight="bold",
    )
    ax.annotate(
        "inside trust",
        xy=(0.0, -145),
        fontsize=9, color=TEAL, ha="center", fontweight="bold",
    )

    ax.set_xlabel("x (slice at y = 0.5)")
    ax.set_ylabel("Energy")
    ax.set_ylim(-220, 200)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Trust Region Illustration (Muller-Brown 1D Slice)")

    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "mb_trust_illustration.pdf")
    plt.close(fig)
    print(f"Wrote {OUTDIR / 'mb_trust_illustration.pdf'}")


if __name__ == "__main__":
    plot_trust_illustration()
