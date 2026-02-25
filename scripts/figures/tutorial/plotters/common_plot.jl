# Common utilities for figure plotters.
#
# Provides the Ruhi theme/palette, HDF5 read helpers, and save_figure.
# No ChemGP dependency -- plotters read from HDF5 only.
#
# Environment variables (set by figure_runner.py):
#   CHEMGP_FIG_OUTPUT -- output directory path
#   CHEMGP_FIG_STEM   -- output file stem
#   CHEMGP_FIG_H5     -- explicit HDF5 path (overrides stem-based default)

using CairoMakie
using AlgebraOfGraphics
using Colors
using LaTeXStrings
using HDF5

const OUTPUT_DIR = get(ENV, "CHEMGP_FIG_OUTPUT", joinpath(@__DIR__, "..", "output"))
mkpath(OUTPUT_DIR)

# --- Ruhi color palette ---
const RUHI = (
    coral=colorant"#FF655D",
    sunshine=colorant"#F1DB4B",
    teal=colorant"#004D40",
    sky=colorant"#1E88E5",
    magenta=colorant"#D81B60",
)

const RUHI_CYCLE = [RUHI.teal, RUHI.sky, RUHI.magenta, RUHI.coral, RUHI.sunshine]

const RUHI_DIVERGING = cgrad([RUHI.teal, RUHI.sky, RUHI.magenta, RUHI.coral, RUHI.sunshine])
const RUHI_FULL = cgrad([RUHI.coral, RUHI.sunshine, RUHI.teal, RUHI.sky, RUHI.magenta])

const ENERGY_COLORMAP = RUHI_DIVERGING
const VARIANCE_COLORMAP = cgrad([colorant"white", RUHI.sky, RUHI.teal])

# --- Publication theme (Jost font, Ruhi colors) ---
const PUBLICATION_THEME = Theme(;
    fontsize=12,
    fonts=(regular="Jost", bold="Jost Bold", italic="Jost Italic"),
    Axis=(
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=10,
        yticklabelsize=10,
        titlesize=13,
        backgroundcolor=:white,
        xgridcolor=colorant"floralwhite",
        ygridcolor=colorant"floralwhite",
        spinecolor=:black,
        xtickcolor=:black,
        ytickcolor=:black,
    ),
    Colorbar=(labelsize=12, ticklabelsize=10),
    Lines=(cycle=Cycle([:color]; covary=true),),
    Scatter=(cycle=Cycle([:color]; covary=true),),
    palette=(color=RUHI_CYCLE,),
    figure_padding=5,
    size=(504, 378),
)

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name * ".pdf")
    save(path, fig; pt_per_unit=1)
    println("Saved: $path")
    return path
end

# --- HDF5 path resolution ---

"""Resolve the HDF5 data file path.

Checks CHEMGP_FIG_H5, then falls back to CHEMGP_FIG_STEM-based path,
then to the provided fallback_stem.
"""
function get_h5_path(; fallback_stem="unnamed")
    p = get(ENV, "CHEMGP_FIG_H5", "")
    isempty(p) || return p
    stem = get(ENV, "CHEMGP_FIG_STEM", fallback_stem)
    return joinpath(OUTPUT_DIR, stem * ".h5")
end

# --- HDF5 read helpers ---

"""Read a table (group of same-length vectors) from HDF5. Returns a Dict{String,Vector}."""
function h5_read_table(path, name="table")
    h5open(path, "r") do f
        g = f[name]
        Dict{String,Any}(k => read(g[k]) for k in keys(g))
    end
end

"""Read a 2D grid from HDF5. Returns (data, x_range, y_range).

x_range and y_range are reconstructed from stored attributes when available.
"""
function h5_read_grid(path, name)
    h5open(path, "r") do f
        ds = f["grids/$name"]
        data = read(ds)
        a = attrs(ds)
        xr = if haskey(a, "x_range") && haskey(a, "x_length")
            lo, hi = a["x_range"]
            n = a["x_length"]
            range(lo, hi; length=n)
        else
            nothing
        end
        yr = if haskey(a, "y_range") && haskey(a, "y_length")
            lo, hi = a["y_range"]
            n = a["y_length"]
            range(lo, hi; length=n)
        else
            nothing
        end
        (data, xr, yr)
    end
end

"""Read a path from HDF5. Returns Dict{String,Vector}."""
function h5_read_path(path, name)
    h5open(path, "r") do f
        g = f["paths/$name"]
        Dict{String,Any}(k => read(g[k]) for k in keys(g))
    end
end

"""Read a point set from HDF5. Returns Dict{String,Vector}."""
function h5_read_points(path, name)
    h5open(path, "r") do f
        g = f["points/$name"]
        Dict{String,Any}(k => read(g[k]) for k in keys(g))
    end
end

"""Read root-level metadata attributes from HDF5. Returns Dict{String,Any}."""
function h5_read_metadata(path)
    h5open(path, "r") do f
        a = attrs(f)
        Dict{String,Any}(k => a[k] for k in keys(a))
    end
end
