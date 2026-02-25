using CairoMakie
using AlgebraOfGraphics
using Colors
using LaTeXStrings

const OUTPUT_DIR = get(ENV, "CHEMGP_FIG_OUTPUT", joinpath(@__DIR__, "output"))
mkpath(OUTPUT_DIR)

# --- Ruhi color palette ---
const RUHI = (
    coral=colorant"#FF655D",
    sunshine=colorant"#F1DB4B",
    teal=colorant"#004D40",
    sky=colorant"#1E88E5",
    magenta=colorant"#D81B60",
)

# Ordered color cycle for line/scatter plots
const RUHI_CYCLE = [RUHI.teal, RUHI.sky, RUHI.magenta, RUHI.coral, RUHI.sunshine]

# Custom colormaps
const RUHI_DIVERGING = cgrad([RUHI.teal, RUHI.sky, RUHI.magenta, RUHI.coral, RUHI.sunshine])
const RUHI_FULL = cgrad([RUHI.coral, RUHI.sunshine, RUHI.teal, RUHI.sky, RUHI.magenta])

# Colormaps for specific uses
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
    size=(504, 378),  # ~7in x 5.25in at 72dpi
)

function save_figure(fig, name)
    path = joinpath(OUTPUT_DIR, name * ".pdf")
    save(path, fig; pt_per_unit=1)
    println("Saved: $path")
    return path
end

# Helper: evaluate oracle on a 2D grid.
# Returns matrix E of shape (nx, ny) in Makie convention: E[i, j] = f(x[i], y[j]).
function eval_grid(oracle, x_range, y_range)
    nx, ny = length(x_range), length(y_range)
    E = zeros(nx, ny)
    for (i, x) in enumerate(x_range), (j, y) in enumerate(y_range)
        e, _ = oracle([x, y])
        E[i, j] = e
    end
    return E
end

const JSONL_WRITER = joinpath(@__DIR__, "..", "..", "jsonl_writer.py")

"""Launch the JSONL writer subprocess. Returns (process, port)."""
function start_jsonl_writer(output_path; port::Int=9876)
    cmd = `python3 $JSONL_WRITER --port $port --output $output_path`
    proc = open(cmd, "r")
    sleep(0.5)  # wait for server to bind
    return proc, port
end

"""Stop the JSONL writer subprocess."""
function stop_jsonl_writer(proc)
    try
        kill(proc)
    catch
    end
end

"""Parse JSONL from gp_minimize machine_output. Returns NamedTuple of vectors."""
function parse_minimize_jsonl(path)
    ocs = Int[]
    forces = Float64[]
    energies = Float64[]
    gates = String[]
    for line in readlines(path)
        startswith(line, "{\"status\"") && continue
        m_oc = match(r"\"oc\":(\d+)", line)
        m_f = match(r"\"F\":([\d.eE+-]+)", line)
        m_e = match(r"\"E\":([\d.eE+-]+)", line)
        m_gate = match(r"\"gate\":\"(\w+)\"", line)
        m_oc === nothing && continue
        m_f === nothing && continue
        push!(ocs, parse(Int, m_oc[1]))
        push!(forces, parse(Float64, m_f[1]))
        push!(energies, m_e !== nothing ? parse(Float64, m_e[1]) : NaN)
        push!(gates, m_gate !== nothing ? String(m_gate[1]) : "ok")
    end
    return (; oracle_calls=ocs, max_fatom=forces, energy=energies, gate=gates)
end

# Helper: GP predict on a 2D grid, returns (E_mean, E_var) matrices.
# Matrices have shape (nx, ny) in Makie convention.
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
