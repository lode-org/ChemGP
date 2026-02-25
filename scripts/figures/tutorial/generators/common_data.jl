# Common utilities for figure data generators.
#
# Provides HDF5 write helpers, TCP socket metrics emission, and grid
# evaluation functions. All generators should `include("common_data.jl")`.
#
# Environment variables (set by figure_runner.py):
#   CHEMGP_FIG_PORT   -- TCP port for metrics socket (empty = disabled)
#   CHEMGP_FIG_OUTPUT -- output directory path
#   CHEMGP_FIG_STEM   -- output file stem (used for HDF5 naming)

using Sockets
using Printf
using HDF5

# --- Environment ---
const FIG_PORT = let p = get(ENV, "CHEMGP_FIG_PORT", "")
    isempty(p) || p == "0" ? nothing : parse(Int, p)
end
const FIG_OUTPUT = get(ENV, "CHEMGP_FIG_OUTPUT", joinpath(@__DIR__, "..", "output"))
const FIG_STEM = get(ENV, "CHEMGP_FIG_STEM", "unnamed")
mkpath(FIG_OUTPUT)

h5_path() = joinpath(FIG_OUTPUT, FIG_STEM * ".h5")

# --- TCP socket connection (metrics only) ---

"""Connect to the metrics socket. Returns `nothing` if port is unset or connection fails."""
function connect_metrics_socket()
    FIG_PORT === nothing && return nothing
    try
        return Sockets.connect("localhost", FIG_PORT)
    catch e
        @warn "Could not connect to metrics socket on port $FIG_PORT" exception = e
        return nothing
    end
end

"""Emit a metrics line as compact JSON (no JSON dependency -- manual formatting)."""
function emit_metric(io; kwargs...)
    io === nothing && return
    parts = String[]
    for (k, v) in kwargs
        key = "\"$(k)\""
        if v isa AbstractString
            push!(parts, "$key:\"$v\"")
        elseif v isa Integer
            push!(parts, "$key:$v")
        elseif v isa AbstractFloat
            if isnan(v) || isinf(v)
                push!(parts, "$key:null")
            else
                push!(parts, "$key:$(@sprintf("%.6g", v))")
            end
        elseif v isa AbstractVector
            elems = join([@sprintf("%.4g", x) for x in v], ",")
            push!(parts, "$key:[$elems]")
        else
            push!(parts, "$key:$(v)")
        end
    end
    println(io, "{" * join(parts, ",") * "}")
    flush(io)
    return nothing
end

# --- HDF5 write helpers ---

"""Write a table (named columns) to an HDF5 group.

Each entry in `cols` becomes a dataset under `/<name>/<key>`.
Creates the file if it does not exist; appends groups if it does.
"""
function h5_write_table(path, name, cols::Dict)
    h5open(path, isfile(path) ? "r+" : "w") do f
        g = create_group(f, name)
        for (k, v) in cols
            write(g, string(k), v)
        end
    end
end

"""Write a 2D grid to HDF5 under `/grids/<name>`.

Stores the matrix as a dataset and optionally x_range/y_range as attributes.
"""
function h5_write_grid(path, name, data::AbstractMatrix;
    x_range::Union{Nothing,AbstractVector}=nothing,
    y_range::Union{Nothing,AbstractVector}=nothing)
    h5open(path, isfile(path) ? "r+" : "w") do f
        g = haskey(f, "grids") ? f["grids"] : create_group(f, "grids")
        write(g, name, collect(data))
        ds = g[name]
        if x_range !== nothing
            attrs(ds)["x_range"] = collect(Float64, x_range[[begin, end]])
            attrs(ds)["x_length"] = length(x_range)
        end
        if y_range !== nothing
            attrs(ds)["y_range"] = collect(Float64, y_range[[begin, end]])
            attrs(ds)["y_length"] = length(y_range)
        end
    end
end

"""Write a path (ordered sequence of points) to HDF5 under `/paths/<name>`.

Keyword arguments become datasets, e.g. `h5_write_path(p, "neb"; x=xs, y=ys)`.
"""
function h5_write_path(path, name; kwargs...)
    h5open(path, isfile(path) ? "r+" : "w") do f
        pg = haskey(f, "paths") ? f["paths"] : create_group(f, "paths")
        g = create_group(pg, name)
        for (k, v) in kwargs
            write(g, string(k), collect(v))
        end
    end
end

"""Write point sets to HDF5 under `/points/<name>`.

Keyword arguments become datasets, e.g. `h5_write_points(p, "minima"; x=xs, y=ys, labels=ls)`.
"""
function h5_write_points(path, name; kwargs...)
    h5open(path, isfile(path) ? "r+" : "w") do f
        pg = haskey(f, "points") ? f["points"] : create_group(f, "points")
        g = create_group(pg, name)
        for (k, v) in kwargs
            write(g, string(k), collect(v))
        end
    end
end

"""Write root-level metadata attributes to the HDF5 file."""
function h5_write_metadata(path; kwargs...)
    h5open(path, isfile(path) ? "r+" : "w") do f
        for (k, v) in kwargs
            attrs(f)[string(k)] = v
        end
    end
end

# --- Grid evaluation (shared with current common.jl) ---

"""Evaluate oracle on a 2D grid. Returns matrix E of shape (nx, ny)."""
function eval_grid(oracle, x_range, y_range)
    nx, ny = length(x_range), length(y_range)
    E = zeros(nx, ny)
    for (i, x) in enumerate(x_range), (j, y) in enumerate(y_range)
        e, _ = oracle([x, y])
        E[i, j] = e
    end
    return E
end

"""GP predict on a 2D grid, returns (E_mean, E_var) matrices."""
function gp_predict_grid(model, x_range, y_range, y_mean, y_std)
    nx, ny = length(x_range), length(y_range)
    E_mean = zeros(nx, ny)
    E_var = zeros(nx, ny)
    for (i, x) in enumerate(x_range)
        for (j, y) in enumerate(y_range)
            X_test = reshape([x, y], 2, 1)
            mu, var = predict_with_variance(model, X_test)
            E_mean[i, j] = mu[1] * y_std + y_mean
            E_var[i, j] = var[1] * y_std^2
        end
    end
    return E_mean, E_var
end
