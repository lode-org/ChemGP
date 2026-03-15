"""Shared plotting theme for ChemGP figure scripts.

Uses the RUHI color scheme from chemparseplot when available,
with a built-in fallback for environments without it installed.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try chemparseplot RUHI theme; fall back to inline definition
try:
    from chemparseplot.plot.theme import (
        RUHI_COLORS,
        setup_publication_theme,
        get_theme,
    )
    setup_publication_theme(get_theme("ruhi"))
    TEAL = RUHI_COLORS["teal"]
    CORAL = RUHI_COLORS["coral"]
    YELLOW = RUHI_COLORS["sunshine"]
    SKY = RUHI_COLORS["sky"]
    MAGENTA = RUHI_COLORS["magenta"]
except ImportError:
    print("chemparseplot not found, using fallback theme", file=sys.stderr)
    TEAL = "#004D40"
    CORAL = "#FF655D"
    YELLOW = "#F1DB4B"
    SKY = "#1E88E5"
    MAGENTA = "#D81B60"

    from matplotlib import font_manager
    _FONT_FAMILY = "sans-serif"
    for font in font_manager.findSystemFonts():
        if "Atkinson" in font or "Hyperlegible" in font:
            _FONT_FAMILY = "Atkinson Hyperlegible"
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

# Try rgpycrumbs parsers; scripts that need them import directly
try:
    from chemparseplot.parse.chemgp_jsonl import (  # noqa: F401
        parse_comparison_jsonl,
        parse_rff_quality_jsonl,
        parse_gp_quality_jsonl,
    )
    HAS_PARSERS = True
except ImportError:
    HAS_PARSERS = False
