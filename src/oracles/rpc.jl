# ==============================================================================
# RPC Oracle: Connect to a remote potential server via rgpot
# ==============================================================================
#
# This integrates ChemGP with the rgpot library (OmniPotentRPC), allowing
# GP-guided optimization to call potentials served over Cap'n Proto RPC.
# Potentials can be served by rgpot's potserv, or by eOn's eonclient --serve
# mode, which wraps any eOn potential (including Metatomic ML models) as an
# rgpot-compatible Cap'n Proto server.
#
# Reference:
#   Bigi, F. et al. (2026). metatensor and metatomic: foundational libraries
#   for interoperable atomistic machine learning. J. Chem. Phys., 164(6),
#   064113. doi:10.1063/5.0304911.
#
#   Goswami, R. (2025). Efficient exploration of chemical kinetics. PhD thesis,
#   University of Iceland. arXiv:2510.21368.
#
# In a typical workflow:
#   1. A server runs a potential (e.g., metatensor, EAM, ML potential):
#        ./potserv 12345 CuH2
#      Or via eOn (for Metatomic / ML potentials):
#        eonclient --serve --potential metatomic --port 12345
#   2. ChemGP connects as a client and uses the potential as its oracle:
#        pot = RpcPotential("localhost", 12345, atmnrs, box)
#        result = gp_minimize(make_rpc_oracle(pot), x_init, kernel)
#
# Two C interfaces are supported:
#   - pot_bridge.h (C++ bridge, simpler): pot_client_init / pot_calculate
#   - rgpot.h (Rust core, richer error handling): rgpot_rpc_client_new / rgpot_rpc_calculate
#
# Library discovery:
#   The shared library is found automatically via find_rgpot_lib(), which
#   searches RGPOT_LIB_PATH, RGPOT_BUILD_DIR, CONDA_PREFIX, and common
#   relative paths. Set RGPOT_BUILD_DIR to the meson builddir of rgpot.
#
# Data layout (matching rgpot convention):
#   positions: flat [x1,y1,z1, x2,y2,z2, ...] (same as ChemGP)
#   atmnrs:    [Z1, Z2, ...] (atomic numbers)
#   box:       flat 3x3 row-major [a_x,a_y,a_z, b_x,b_y,b_z, c_x,c_y,c_z]
#   forces:    flat [Fx1,Fy1,Fz1, ...] (note: forces = -gradient)

# ==============================================================================
# Library and executable discovery
# ==============================================================================

const _RGPOT_BRIDGE_NAMES = Sys.isapple() ?
    ["libpot_client_bridge.dylib"] :
    ["libpot_client_bridge.so"]

const _RGPOT_CORE_NAMES = Sys.isapple() ?
    ["librgpot_core.dylib"] :
    ["librgpot_core.so"]

# Subdirectories within a meson/cmake build tree where libraries may appear
const _LIB_SEARCH_SUBDIRS = ["CppCore/rgpot/rpc", "lib", "lib64", ""]

"""
    find_rgpot_lib(; variant=:bridge)

Search for the rgpot shared library in common locations.

Checks (in order):
1. `RGPOT_LIB_PATH` environment variable (direct path to the .so/.dylib)
2. `RGPOT_BUILD_DIR` environment variable (meson/cmake build directory)
3. Pixi/conda prefix (`CONDA_PREFIX`)
4. Relative paths from the ChemGP project root

Returns the absolute path if found, `nothing` otherwise.
"""
function find_rgpot_lib(; variant::Symbol = :bridge)
    names = variant == :bridge ? _RGPOT_BRIDGE_NAMES : _RGPOT_CORE_NAMES

    # 1. Direct path from environment
    env_path = get(ENV, "RGPOT_LIB_PATH", nothing)
    if env_path !== nothing && isfile(env_path)
        return abspath(env_path)
    end

    # 2. Build directory
    build_dir = get(ENV, "RGPOT_BUILD_DIR", nothing)
    if build_dir !== nothing
        for name in names, sub in _LIB_SEARCH_SUBDIRS
            p = joinpath(build_dir, sub, name)
            isfile(p) && return abspath(p)
        end
    end

    # 3. Conda / pixi prefix
    prefix = get(ENV, "CONDA_PREFIX", nothing)
    if prefix !== nothing
        for name in names
            p = joinpath(prefix, "lib", name)
            isfile(p) && return abspath(p)
        end
    end

    # 4. Relative to ChemGP project root
    project_root = dirname(dirname(@__DIR__))
    for rel in ["rgpot/builddir", "build/rgpot", "../rgpot/builddir"]
        dir = joinpath(project_root, rel)
        for name in names, sub in _LIB_SEARCH_SUBDIRS
            p = joinpath(dir, sub, name)
            isfile(p) && return abspath(p)
        end
    end

    return nothing
end

"""
    find_potserv(; build_dir=nothing)

Search for the `potserv` executable in common locations.

Checks `RGPOT_BUILD_DIR`, `CONDA_PREFIX`, and relative paths.
Returns the absolute path if found, `nothing` otherwise.
"""
function find_potserv(; build_dir::Union{AbstractString,Nothing} = nothing)
    dirs = String[]
    if build_dir !== nothing
        push!(dirs, build_dir)
    end
    env_dir = get(ENV, "RGPOT_BUILD_DIR", nothing)
    if env_dir !== nothing
        push!(dirs, env_dir)
    end
    prefix = get(ENV, "CONDA_PREFIX", nothing)
    if prefix !== nothing
        push!(dirs, joinpath(prefix, "bin"))
    end
    project_root = dirname(dirname(@__DIR__))
    for rel in ["rgpot/builddir", "build/rgpot", "../rgpot/builddir"]
        push!(dirs, joinpath(project_root, rel))
    end

    for dir in dirs
        for sub in ["CppCore/rgpot/rpc", "bin", ""]
            p = joinpath(dir, sub, "potserv")
            isfile(p) && return abspath(p)
        end
    end
    return nothing
end

"""
    with_potserv(f, port, potential; build_dir=nothing, startup_time=2.0)

Start a `potserv` process serving `potential` on `port`, execute `f()`,
then kill the server. Useful for integration tests.

# Example
```julia
with_potserv(12345, "LJ") do
    pot = RpcPotential("localhost", 12345, Int32[0,0,0], zeros(9))
    oracle = make_rpc_oracle(pot)
    E, G = oracle(x)
end
```
"""
function with_potserv(
    f::Function,
    port::Integer,
    potential::AbstractString;
    build_dir::Union{AbstractString,Nothing} = nothing,
    startup_time::Real = 2.0,
)
    exe = find_potserv(; build_dir)
    exe === nothing && error(
        "Could not find potserv executable. " *
        "Set RGPOT_BUILD_DIR to the rgpot meson builddir.",
    )

    proc = run(
        pipeline(`$exe $port $potential`; stdout = devnull, stderr = devnull);
        wait = false,
    )
    sleep(startup_time)

    try
        f()
    finally
        kill(proc)
        wait(proc)
    end
end

"""
    rgpot_available(; variant=:bridge)

Return `true` if the rgpot shared library can be found.
"""
rgpot_available(; variant::Symbol = :bridge) = find_rgpot_lib(; variant) !== nothing

"""
    potserv_available(; build_dir=nothing)

Return `true` if the potserv executable can be found.
"""
potserv_available(; build_dir::Union{AbstractString,Nothing} = nothing) =
    find_potserv(; build_dir) !== nothing

# ==============================================================================
# RpcPotential: Wraps a connection to an rgpot Cap'n Proto server
# ==============================================================================

"""
    RpcPotential

A connection to a remote potential server via rgpot's Cap'n Proto RPC.

Holds the client handle, atomic numbers, and simulation cell. The client
handle is automatically freed when the object is garbage collected.

# Fields
- `client_ptr`: Opaque pointer to the C client handle
- `lib_handle`: dlopen handle (kept alive to prevent unloading)
- `atmnrs`: Atomic numbers for the system (fixed)
- `box`: Simulation cell vectors, flat 3x3 row-major (fixed or mutable)
- `n_atoms`: Number of atoms
"""
mutable struct RpcPotential
    client_ptr::Ptr{Cvoid}
    lib_handle::Ptr{Cvoid}
    atmnrs::Vector{Int32}
    box::Vector{Float64}
    n_atoms::Int

    # Store function pointers so we don't dlsym on every call
    _calculate::Ptr{Cvoid}
    _free::Ptr{Cvoid}
    _last_error::Ptr{Cvoid}
end

# ==============================================================================
# Constructor: connect via pot_bridge.h (C++ bridge)
# ==============================================================================

"""
    RpcPotential(host, port, lib_path, atmnrs, box)

Connect to an rgpot potential server using the pot_bridge C interface.

# Arguments
- `host`: Server hostname (e.g., "localhost")
- `port`: Server port (e.g., 12345)
- `lib_path`: Path to `libpot_client_bridge.so` (or `librgpot.so` if it
  includes the bridge)
- `atmnrs`: Vector of atomic numbers `[Z1, Z2, ...]`
- `box`: Simulation cell as a 9-element vector (flat 3x3 row-major),
  or a 3x3 matrix

# Example
```julia
pot = RpcPotential("localhost", 12345,
                   "/path/to/libpot_client_bridge.so",
                   Int32[29, 29, 1, 1],  # Cu2H2
                   Float64[10,0,0, 0,10,0, 0,0,10])
```

See also the auto-discovery constructor:
```julia
pot = RpcPotential("localhost", 12345,
                   Int32[29, 29, 1, 1],
                   Float64[10,0,0, 0,10,0, 0,0,10])
```
"""
function RpcPotential(
    host::AbstractString,
    port::Integer,
    lib_path::AbstractString,
    atmnrs::Vector{<:Integer},
    box::Union{Vector{Float64},Matrix{Float64}},
)
    # Flatten box if matrix
    box_flat = box isa Matrix ? vec(box') : copy(box)
    length(box_flat) == 9 || error("box must have 9 elements (flat 3x3)")
    atmnrs32 = Int32.(atmnrs)
    n_atoms = length(atmnrs32)

    # Load the shared library
    lib = Libc.Libdl.dlopen(lib_path)
    lib == C_NULL && error("Failed to load library: $lib_path")

    # Resolve function pointers
    fn_init = Libc.Libdl.dlsym(lib, :pot_client_init)
    fn_calc = Libc.Libdl.dlsym(lib, :pot_calculate)
    fn_free = Libc.Libdl.dlsym(lib, :pot_client_free)
    fn_err = Libc.Libdl.dlsym(lib, :pot_get_last_error)

    if fn_init == C_NULL || fn_calc == C_NULL || fn_free == C_NULL
        Libc.Libdl.dlclose(lib)
        error("Library $lib_path does not export pot_bridge functions. " *
              "Ensure it was built with RPC support.")
    end

    # Connect to server
    client = ccall(fn_init, Ptr{Cvoid}, (Cstring, Int32), host, Int32(port))
    if client == C_NULL
        err_msg = _get_bridge_error(fn_err, C_NULL)
        Libc.Libdl.dlclose(lib)
        error("Failed to connect to $host:$port: $err_msg")
    end

    pot = RpcPotential(client, lib, atmnrs32, box_flat, n_atoms,
                       fn_calc, fn_free, fn_err)

    # Register finalizer to clean up on GC
    finalizer(pot) do p
        if p.client_ptr != C_NULL
            ccall(p._free, Cvoid, (Ptr{Cvoid},), p.client_ptr)
            p.client_ptr = C_NULL
        end
        if p.lib_handle != C_NULL
            Libc.Libdl.dlclose(p.lib_handle)
            p.lib_handle = C_NULL
        end
    end

    return pot
end

"""
    RpcPotential(host, port, atmnrs, box)

Convenience constructor that auto-discovers the rgpot shared library.

The library is searched via [`find_rgpot_lib`](@ref). Set the environment
variable `RGPOT_LIB_PATH` (direct path) or `RGPOT_BUILD_DIR` (meson
builddir) to guide discovery.

# Example
```julia
# With RGPOT_BUILD_DIR set:
pot = RpcPotential("localhost", 12345,
                   Int32[29, 29, 1, 1],
                   Float64[10,0,0, 0,10,0, 0,0,10])
oracle = make_rpc_oracle(pot)
result = gp_minimize(oracle, x_init, kernel)
```
"""
function RpcPotential(
    host::AbstractString,
    port::Integer,
    atmnrs::Vector{<:Integer},
    box::Union{Vector{Float64},Matrix{Float64}},
)
    lib_path = find_rgpot_lib(; variant = :bridge)
    lib_path === nothing && error(
        "Could not find rgpot shared library. " *
        "Set RGPOT_LIB_PATH or RGPOT_BUILD_DIR environment variable. " *
        "See: https://github.com/OmniPotentRPC/rgpot",
    )
    return RpcPotential(host, port, lib_path, atmnrs, box)
end

function _get_bridge_error(fn_err::Ptr{Cvoid}, client::Ptr{Cvoid})
    if fn_err == C_NULL
        return "unknown error"
    end
    msg_ptr = ccall(fn_err, Cstring, (Ptr{Cvoid},), client)
    return msg_ptr == C_NULL ? "unknown error" : unsafe_string(msg_ptr)
end

# ==============================================================================
# Calculate: call the remote potential
# ==============================================================================

"""
    calculate(pot::RpcPotential, positions::AbstractVector{Float64})

Call the remote potential server for a given set of atomic positions.
Returns `(energy::Float64, forces::Vector{Float64})`.

Note: returns **forces** (F = -dE/dx), not gradients. Use `make_rpc_oracle`
to get a ChemGP-compatible oracle that returns gradients.
"""
function calculate(pot::RpcPotential, positions::AbstractVector{Float64})
    pot.client_ptr == C_NULL && error("RpcPotential has been closed")
    length(positions) == 3 * pot.n_atoms ||
        error("Expected $(3 * pot.n_atoms) coordinates, got $(length(positions))")

    forces = zeros(Float64, 3 * pot.n_atoms)
    energy = Ref{Float64}(0.0)

    status = ccall(
        pot._calculate, Int32,
        (Ptr{Cvoid}, Int32, Ptr{Float64}, Ptr{Int32}, Ptr{Float64},
         Ptr{Float64}, Ptr{Float64}),
        pot.client_ptr,
        Int32(pot.n_atoms),
        positions,
        pot.atmnrs,
        pot.box,
        energy,
        forces,
    )

    if status != 0
        err_msg = _get_bridge_error(pot._last_error, pot.client_ptr)
        error("RPC calculation failed (status=$status): $err_msg")
    end

    return energy[], forces
end

"""
    close!(pot::RpcPotential)

Explicitly close the RPC connection and free resources.
The potential cannot be used after this call.
"""
function close!(pot::RpcPotential)
    if pot.client_ptr != C_NULL
        ccall(pot._free, Cvoid, (Ptr{Cvoid},), pot.client_ptr)
        pot.client_ptr = C_NULL
    end
    if pot.lib_handle != C_NULL
        Libc.Libdl.dlclose(pot.lib_handle)
        pot.lib_handle = C_NULL
    end
end

# ==============================================================================
# Oracle adapter: wraps RpcPotential as a ChemGP oracle
# ==============================================================================

"""
    make_rpc_oracle(pot::RpcPotential)

Create a ChemGP-compatible oracle function from an `RpcPotential`.

Returns a function `x -> (E, G)` where:
- `x` is a flat coordinate vector `[x1,y1,z1, x2,y2,z2, ...]`
- `E` is the energy
- `G` is the gradient (dE/dx), which equals `-forces`

This can be passed directly to `gp_minimize` or `gp_dimer`:

```julia
pot = RpcPotential("localhost", 12345, libpath, atmnrs, box)
oracle = make_rpc_oracle(pot)
result = gp_minimize(oracle, x_init, kernel)
```
"""
function make_rpc_oracle(pot::RpcPotential)
    return function (x::AbstractVector{<:Real})
        E, F = calculate(pot, Float64.(x))
        G = -F  # gradient = -forces
        return E, G
    end
end

# ==============================================================================
# Alternative: connect via rgpot.h (Rust core C API)
# ==============================================================================
#
# The Rust core API uses typed structs and has richer error handling.
# This is used when the library exports rgpot_rpc_* symbols instead of
# pot_* symbols (i.e., when using librgpot_core directly).

"""
    RpcPotentialCore(host, port, lib_path, atmnrs, box)

Connect using the Rust core C API (`rgpot.h`). This is an alternative to
the C++ bridge when using `librgpot_core.so` directly.

Same interface as `RpcPotential` but uses `rgpot_rpc_client_new` /
`rgpot_rpc_calculate` / `rgpot_rpc_client_free`.
"""
mutable struct RpcPotentialCore
    client_ptr::Ptr{Cvoid}
    lib_handle::Ptr{Cvoid}
    atmnrs::Vector{Int32}
    box::Vector{Float64}
    n_atoms::Int

    # Function pointers
    _calculate::Ptr{Cvoid}
    _free::Ptr{Cvoid}
    _last_error::Ptr{Cvoid}
    _input_create::Ptr{Cvoid}
    _output_create::Ptr{Cvoid}
end

# Mirror the C structs for ccall
struct CForceInput
    n_atoms::Csize_t
    pos::Ptr{Float64}
    atmnrs::Ptr{Cint}
    box_::Ptr{Float64}
end

mutable struct CForceOutput
    forces::Ptr{Float64}
    energy::Float64
    variance::Float64
end

function RpcPotentialCore(
    host::AbstractString,
    port::Integer,
    lib_path::AbstractString,
    atmnrs::Vector{<:Integer},
    box::Union{Vector{Float64},Matrix{Float64}},
)
    box_flat = box isa Matrix ? vec(box') : copy(box)
    length(box_flat) == 9 || error("box must have 9 elements (flat 3x3)")
    atmnrs32 = Int32.(atmnrs)
    n_atoms = length(atmnrs32)

    lib = Libc.Libdl.dlopen(lib_path)
    lib == C_NULL && error("Failed to load library: $lib_path")

    fn_new = Libc.Libdl.dlsym(lib, :rgpot_rpc_client_new)
    fn_calc = Libc.Libdl.dlsym(lib, :rgpot_rpc_calculate)
    fn_free = Libc.Libdl.dlsym(lib, :rgpot_rpc_client_free)
    fn_err = Libc.Libdl.dlsym(lib, :rgpot_last_error)
    fn_input = Libc.Libdl.dlsym(lib, :rgpot_force_input_create)
    fn_output = Libc.Libdl.dlsym(lib, :rgpot_force_out_create)

    if fn_new == C_NULL || fn_calc == C_NULL || fn_free == C_NULL
        Libc.Libdl.dlclose(lib)
        error("Library $lib_path does not export rgpot_rpc_* functions. " *
              "Ensure it was built with the 'rpc' feature.")
    end

    client = ccall(fn_new, Ptr{Cvoid}, (Cstring, UInt16), host, UInt16(port))
    if client == C_NULL
        msg_ptr = ccall(fn_err, Cstring, ())
        err = msg_ptr == C_NULL ? "unknown" : unsafe_string(msg_ptr)
        Libc.Libdl.dlclose(lib)
        error("Failed to connect to $host:$port: $err")
    end

    pot = RpcPotentialCore(client, lib, atmnrs32, box_flat, n_atoms,
                           fn_calc, fn_free, fn_err, fn_input, fn_output)

    finalizer(pot) do p
        if p.client_ptr != C_NULL
            ccall(p._free, Cvoid, (Ptr{Cvoid},), p.client_ptr)
            p.client_ptr = C_NULL
        end
        if p.lib_handle != C_NULL
            Libc.Libdl.dlclose(p.lib_handle)
            p.lib_handle = C_NULL
        end
    end

    return pot
end

function calculate(pot::RpcPotentialCore, positions::AbstractVector{Float64})
    pot.client_ptr == C_NULL && error("RpcPotentialCore has been closed")
    length(positions) == 3 * pot.n_atoms ||
        error("Expected $(3 * pot.n_atoms) coordinates, got $(length(positions))")

    forces = zeros(Float64, 3 * pot.n_atoms)

    # Build C structs
    input = CForceInput(
        Csize_t(pot.n_atoms),
        pointer(positions),
        pointer(pot.atmnrs),
        pointer(pot.box),
    )
    output = CForceOutput(pointer(forces), 0.0, 0.0)

    status = ccall(
        pot._calculate, Cint,
        (Ptr{Cvoid}, Ref{CForceInput}, Ref{CForceOutput}),
        pot.client_ptr, input, output,
    )

    if status != 0  # RGPOT_SUCCESS = 0
        msg_ptr = ccall(pot._last_error, Cstring, ())
        err = msg_ptr == C_NULL ? "unknown" : unsafe_string(msg_ptr)
        error("RPC calculation failed (status=$status): $err")
    end

    return output.energy, forces
end

function close!(pot::RpcPotentialCore)
    if pot.client_ptr != C_NULL
        ccall(pot._free, Cvoid, (Ptr{Cvoid},), pot.client_ptr)
        pot.client_ptr = C_NULL
    end
    if pot.lib_handle != C_NULL
        Libc.Libdl.dlclose(pot.lib_handle)
        pot.lib_handle = C_NULL
    end
end

# Oracle adapter works the same way
function make_rpc_oracle(pot::RpcPotentialCore)
    return function (x::AbstractVector{<:Real})
        E, F = calculate(pot, Float64.(x))
        G = -F  # gradient = -forces
        return E, G
    end
end
