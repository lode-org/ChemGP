# ==============================================================================
# NEB I/O: extxyz trajectories and eOn-compatible .dat files
# ==============================================================================

"""
    ELEMENT_SYMBOLS

Mapping from atomic number to element symbol for common elements.
"""
const ELEMENT_SYMBOLS = Dict{Int,String}(
    1 => "H",
    2 => "He",
    3 => "Li",
    4 => "Be",
    5 => "B",
    6 => "C",
    7 => "N",
    8 => "O",
    9 => "F",
    10 => "Ne",
    11 => "Na",
    12 => "Mg",
    13 => "Al",
    14 => "Si",
    15 => "P",
    16 => "S",
    17 => "Cl",
    18 => "Ar",
    19 => "K",
    20 => "Ca",
    21 => "Sc",
    22 => "Ti",
    23 => "V",
    24 => "Cr",
    25 => "Mn",
    26 => "Fe",
    27 => "Co",
    28 => "Ni",
    29 => "Cu",
    30 => "Zn",
    31 => "Ga",
    32 => "Ge",
    33 => "As",
    34 => "Se",
    35 => "Br",
    36 => "Kr",
    37 => "Rb",
    38 => "Sr",
    39 => "Y",
    40 => "Zr",
    41 => "Nb",
    42 => "Mo",
    44 => "Ru",
    45 => "Rh",
    46 => "Pd",
    47 => "Ag",
    48 => "Cd",
    49 => "In",
    50 => "Sn",
    53 => "I",
    54 => "Xe",
    55 => "Cs",
    56 => "Ba",
    72 => "Hf",
    73 => "Ta",
    74 => "W",
    75 => "Re",
    76 => "Os",
    77 => "Ir",
    78 => "Pt",
    79 => "Au",
    80 => "Hg",
    82 => "Pb",
    83 => "Bi",
)

# Reverse lookup: element symbol -> atomic number
const SYMBOL_TO_ATOMIC_NUMBER = Dict{String,Int}(v => k for (k, v) in ELEMENT_SYMBOLS)

# --------------------------------------------------------------------------
# ExtXYZ reader (minimal, no external dependencies)
# --------------------------------------------------------------------------

"""
    read_extxyz(filename) -> (positions, atomic_numbers, box)

Read a single frame from an extended XYZ file.

Returns:
- `positions::Vector{Float64}`: flat coordinate vector [x1,y1,z1, x2,y2,z2, ...]
- `atomic_numbers::Vector{Int}`: atomic number for each atom
- `box::Vector{Float64}`: 9-element lattice vector (row-major 3x3)
"""
function read_extxyz(filename::AbstractString)
    lines = readlines(filename)
    n_atoms = parse(Int, strip(lines[1]))

    # Parse Lattice from info line
    info = lines[2]
    box = Float64[20, 0, 0, 0, 20, 0, 0, 0, 20]  # default
    m = match(r"Lattice=\"([^\"]+)\"", info)
    if m !== nothing
        box = parse.(Float64, split(strip(m.captures[1])))
    end

    positions = Vector{Float64}(undef, 3 * n_atoms)
    atomic_numbers = Vector{Int}(undef, n_atoms)

    for i in 1:n_atoms
        parts = split(strip(lines[2 + i]))
        sym = String(parts[1])
        z = get(SYMBOL_TO_ATOMIC_NUMBER, sym, 0)
        if z == 0
            error("Unknown element symbol: $sym")
        end
        atomic_numbers[i] = z
        positions[3*(i-1)+1] = parse(Float64, parts[2])
        positions[3*(i-1)+2] = parse(Float64, parts[3])
        positions[3*(i-1)+3] = parse(Float64, parts[4])
    end

    return (positions=positions, atomic_numbers=atomic_numbers, box=box)
end

# --------------------------------------------------------------------------
# Internal: compute f_parallel via improved H&J tangent
# --------------------------------------------------------------------------
function _compute_f_para(path::NEBPath)
    n = length(path.images)
    f_para = zeros(n)
    for i in 1:n
        forces_i = -path.gradients[i]

        if i == 1 || i == n
            tau = if (i == 1)
                (path.images[2] .- path.images[1])
            else
                (path.images[n] .- path.images[n - 1])
            end
            tn = norm(tau)
            tn > 0 && (f_para[i] = dot(forces_i, tau / tn))
            continue
        end

        tau_plus = path.images[i + 1] .- path.images[i]
        tau_minus = path.images[i] .- path.images[i - 1]
        e_prev, e_curr, e_next = path.energies[i - 1],
        path.energies[i],
        path.energies[i + 1]
        de_plus = e_next - e_curr
        de_minus = e_curr - e_prev

        if e_next > e_curr > e_prev
            tau = tau_plus
        elseif e_next < e_curr < e_prev
            tau = tau_minus
        else
            de_max = max(abs(de_plus), abs(de_minus))
            de_min = min(abs(de_plus), abs(de_minus))
            tau = if (e_next > e_prev)
                tau_plus * de_max + tau_minus * de_min
            else
                tau_plus * de_min + tau_minus * de_max
            end
        end

        tn = norm(tau)
        tn > 0 && (f_para[i] = dot(forces_i, tau / tn))
    end
    return f_para
end

# --------------------------------------------------------------------------
# Extended XYZ trajectory writer
# --------------------------------------------------------------------------

"""
    write_neb_trajectory(path::NEBPath, filename, atomic_numbers, cell; pbc=false)
    write_neb_trajectory(result::NEBResult, filename, atomic_numbers, cell; pbc=false)

Write NEB images as an extended XYZ trajectory file.

Each image becomes one frame with energy in the info line and per-atom
species, positions, and forces (forces = -gradients).
"""
function write_neb_trajectory(
    path::NEBPath,
    filename::AbstractString,
    atomic_numbers::AbstractVector{<:Integer},
    cell::AbstractVector{<:Real};
    pbc::Bool=false,
)
    n_atoms = length(atomic_numbers)
    @assert length(cell) == 9 "cell must have 9 elements (3x3 row-major)"

    pbc_str = pbc ? "T T T" : "F F F"

    open(filename, "w") do io
        for (i, image) in enumerate(path.images)
            energy = path.energies[i]
            grad = path.gradients[i]

            println(io, n_atoms)
            @printf(
                io,
                "Lattice=\"%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\" ",
                cell[1],
                cell[2],
                cell[3],
                cell[4],
                cell[5],
                cell[6],
                cell[7],
                cell[8],
                cell[9]
            )
            @printf(io, "Properties=species:S:1:pos:R:3:forces:R:3 ")
            @printf(io, "energy=%.15e ", energy)
            @printf(io, "pbc=\"%s\" ", pbc_str)
            @printf(io, "image=%d\n", i - 1)

            for a in 1:n_atoms
                sym = get(ELEMENT_SYMBOLS, Int(atomic_numbers[a]), "X")
                idx = 3*(a-1)
                @printf(
                    io,
                    "%-2s %20.15f %20.15f %20.15f %20.15f %20.15f %20.15f\n",
                    sym,
                    image[idx + 1],
                    image[idx + 2],
                    image[idx + 3],
                    -grad[idx + 1],
                    -grad[idx + 2],
                    -grad[idx + 3]
                )
            end
        end
    end
    return nothing
end

function write_neb_trajectory(result::NEBResult, args...; kwargs...)
    write_neb_trajectory(result.path, args...; kwargs...)
end

# --------------------------------------------------------------------------
# eOn-compatible .dat writer
# --------------------------------------------------------------------------

"""
    write_neb_dat(path::NEBPath, filename)
    write_neb_dat(result::NEBResult, filename)

Write NEB path data in eOn-compatible `.dat` format.

Columns: `img  rxn_coord  energy  f_para`

Matches the output of eOn's `NudgedElasticBand::printImageData`.
"""
function write_neb_dat(path::NEBPath, filename::AbstractString)
    n = length(path.images)

    # Cumulative distance along path
    dists = zeros(n)
    for i in 2:n
        dists[i] = dists[i - 1] + norm(path.images[i] .- path.images[i - 1])
    end

    E_ref = path.energies[1]
    f_para = _compute_f_para(path)

    open(filename, "w") do io
        @printf(io, "%3s %12s %12s %12s\n", "img", "rxn_coord", "energy", "f_para")
        for i in 1:n
            @printf(
                io,
                "%3d %12.6f %12.6f %12.6f\n",
                i - 1,
                dists[i],
                path.energies[i] - E_ref,
                f_para[i]
            )
        end
    end
    return nothing
end

function write_neb_dat(result::NEBResult, filename::AbstractString)
    write_neb_dat(result.path, filename)
end

# --------------------------------------------------------------------------
# Convergence CSV
# --------------------------------------------------------------------------

"""
    write_convergence_csv(result::NEBResult, filename)

Write NEB convergence history to a CSV file.

Columns: iteration, max_force, ci_force, oracle_calls, max_energy
"""
function write_convergence_csv(result::NEBResult, filename::AbstractString)
    h = result.history
    n = length(h["max_force"])

    open(filename, "w") do io
        println(io, "iteration,max_force,ci_force,oracle_calls,max_energy")
        for i in 1:n
            max_f = h["max_force"][i]
            ci_f =
                haskey(h, "ci_force") && i <= length(h["ci_force"]) ? h["ci_force"][i] : 0.0
            oc = if haskey(h, "oracle_calls") && i <= length(h["oracle_calls"])
                h["oracle_calls"][i]
            else
                0
            end
            me = if haskey(h, "max_energy") && i <= length(h["max_energy"])
                h["max_energy"][i]
            else
                0.0
            end
            @printf(io, "%d,%.15e,%.15e,%d,%.15e\n", i, max_f, ci_f, oc, me)
        end
    end
    return nothing
end

# --------------------------------------------------------------------------
# Step callback helper for eOn-style per-step output
# --------------------------------------------------------------------------

"""
    make_neb_writer(output_dir, atomic_numbers, cell; pbc=false)

Create a step callback for NEB optimization that writes per-step `.dat` and
`.xyz` files, matching eOn's `write_movies` output format.

Returns a function `(path::NEBPath, iteration::Int) -> nothing` suitable for
passing as the `on_step` keyword argument to `neb_optimize`, `gp_neb_aie`,
or `gp_neb_oie`.

Output files per step:
- `neb_NNN.dat`       -- tabular energy/force data (eOn format)
- `neb_path_NNN.xyz`  -- atomic positions as extxyz

Also writes `neb.dat` (overwritten each step with latest data).
"""
function make_neb_writer(
    output_dir::AbstractString,
    atomic_numbers::AbstractVector{<:Integer},
    cell::AbstractVector{<:Real};
    pbc::Bool=false,
)
    mkpath(output_dir)
    return function (path::NEBPath, iteration::Int)
        idx = @sprintf("%03d", iteration)
        write_neb_dat(path, joinpath(output_dir, "neb_$idx.dat"))
        write_neb_trajectory(
            path, joinpath(output_dir, "neb_path_$idx.xyz"), atomic_numbers, cell; pbc
        )
        # Also write current state as neb.dat (final overwrite)
        write_neb_dat(path, joinpath(output_dir, "neb.dat"))
    end
end
