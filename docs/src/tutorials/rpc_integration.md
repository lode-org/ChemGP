# RPC Integration

This tutorial explains how to connect ChemGP to external potential energy
calculators served over RPC using the [rgpot](https://github.com/OmniPotentRPC/rgpot)
library. The RPC integration is designed to work with interoperable atomistic
machine learning frameworks such as
[metatensor](https://doi.org/10.1063/5.0304911) (Bigi et al. 2026).

## Overview

In production use, the oracle is not a simple Lennard-Jones potential but an
expensive quantum chemistry or machine-learning potential. The rgpot library
provides a Cap'n Proto RPC interface that allows ChemGP to call these potentials
over the network.

The workflow is:
1. A server runs a potential (rgpot's `potserv`, or eOn's `eonclient --serve`)
2. ChemGP connects as a client and uses the potential as its oracle

```
eonclient --serve "metatomic:12345" --config model.ini
    |
    |  Cap'n Proto RPC (binary, efficient)
    |
    v
ChemGP: oracle = make_rpc_oracle(pot)
    |
    v
GP-guided optimization (gp_minimize, gp_dimer, gp_neb, otgpd)
```

## Setup

### Prerequisites

- [pixi](https://pixi.sh) for environment management
- Cap'n Proto (installed via pixi)
- For ML potentials: PyTorch, metatensor, metatomic (installed via pixi/pip)

### Building the bridge library (rgpot)

ChemGP communicates with potential servers via `libpot_client_bridge.so`
from the [rgpot](https://github.com/OmniPotentRPC/rgpot) project.

The easiest way is via the pixi tasks shipped with ChemGP:

```bash
pixi run rgpot-build
```

This clones rgpot, configures meson with RPC support, and compiles. The
resulting `potserv` binary and `libpot_client_bridge.so` end up in
`rgpot/builddir/CppCore/rgpot/rpc/`.

To build manually:

```bash
git clone https://github.com/OmniPotentRPC/rgpot.git
cd rgpot
pixi run -e default meson setup builddir -Dwith_rpc=true
meson compile -C builddir
```

### Building eOn with serve mode

To serve eOn potentials (LJ, EAM, Metatomic, etc.) over RPC:

```bash
cd /path/to/eOn

# Serve mode only (LJ, EAM, etc.)
pixi run -e serve bash -c 'meson setup bbdir --prefix=$CONDA_PREFIX --libdir=lib -Dwith_serve=true && meson compile -C bbdir'

# Serve + Metatomic ML potentials
pixi run -e dev-serve-mta bash -c 'meson setup bbdir --prefix=$CONDA_PREFIX --libdir=lib -Dwith_serve=true -Dwith_metatomic=true -Dpip_metatomic=true -Dtorch_version=2.9 && meson compile -C bbdir'
```

The `dev-serve-mta` pixi environment pins `fmt` to v11 to match torch's
bundled fmt, preventing ABI mismatches with spdlog.

### Environment variables

Set these so ChemGP can find the bridge library automatically:

```bash
# Option 1: point to the rgpot build directory
export RGPOT_BUILD_DIR=/path/to/rgpot/builddir

# Option 2: point directly to the library
export RGPOT_LIB_PATH=/path/to/libpot_client_bridge.so

# Option 3: install rgpot into the conda/pixi prefix
# (the library will be found in $CONDA_PREFIX/lib)
```

## Starting a Potential Server

### rgpot potserv (built-in potentials)

```bash
# Lennard-Jones
rgpot/builddir/CppCore/rgpot/rpc/potserv 12345 LJ

# CuH2 EAM
rgpot/builddir/CppCore/rgpot/rpc/potserv 12345 CuH2
```

### eOn (Metatomic / ML potentials)

With eOn built with `--serve` support:

```bash
# Single potential (spec format: potential:port)
eonclient --serve "metatomic:12345" --config model.ini

# Multiple potentials concurrently
eonclient --serve "lj:12345,metatomic:12346" --config model.ini

# Gateway mode: single port backed by a pool of instances
eonclient -p metatomic --serve-port 12345 --replicas 4 --gateway --config model.ini
```

The config file provides potential-specific parameters (model paths, etc.):

```ini
[Metatomic]
model_path = /path/to/model.pt
device = cuda
```

This exposes any eOn potential (including Metatomic models backed by
PyTorch/metatensor) as an rgpot-compatible RPC server. The serve mode
uses a flat-array callback interface internally, avoiding any type
collision between eOn and rgpot data structures.

## Connecting from ChemGP

### Auto-discovery (recommended)

Set the `RGPOT_BUILD_DIR` environment variable:

```bash
export RGPOT_BUILD_DIR=/path/to/rgpot/builddir
```

Then connect with the convenience constructor:

```julia
using ChemGP

pot = RpcPotential("localhost", 12345,
                   Int32[29, 29, 1, 1],  # Atomic numbers (Cu2H2)
                   Float64[10,0,0, 0,10,0, 0,0,10])  # Cell (flat 3x3)
```

The library is found automatically via [`find_rgpot_lib`](@ref), which searches:
1. `RGPOT_LIB_PATH` (direct path to the .so/.dylib)
2. `RGPOT_BUILD_DIR` (meson build directory)
3. `CONDA_PREFIX` (pixi/conda environment)
4. Relative paths from the ChemGP project root

### Explicit path

```julia
pot = RpcPotential("localhost", 12345,
                   "/path/to/libpot_client_bridge.so",
                   Int32[29, 29, 1, 1],
                   Float64[10,0,0, 0,10,0, 0,0,10])
```

## Creating an Oracle

Wrap the RPC potential as a ChemGP-compatible oracle:

```julia
oracle = make_rpc_oracle(pot)

# Now use it like any other oracle
kernel = MolInvDistSE(1.0, [0.5], Float64[])
result = gp_minimize(oracle, x_init, kernel)
```

The [`make_rpc_oracle`](@ref) function handles the sign convention: rgpot returns
forces (``F = -\nabla E``), while ChemGP expects gradients (``\nabla E``).

## Integration Testing

ChemGP includes a helper for running tests against a temporary server:

```julia
using ChemGP

with_potserv(12345, "LJ") do
    pot = RpcPotential("localhost", 12345, atmnrs, box)
    oracle = make_rpc_oracle(pot)
    E, G = oracle(x)
end
# Server is automatically killed when the block exits
```

Run the full integration test suite:

```bash
pixi run integration-test
```

## Two C Interfaces

ChemGP supports two ways to connect:

| Type | Library | Symbols | Use Case |
|:-----|:--------|:--------|:---------|
| [`RpcPotential`](@ref) | `libpot_client_bridge.so` | `pot_client_init` | C++ bridge (simpler) |
| [`RpcPotentialCore`](@ref) | `librgpot_core.so` | `rgpot_rpc_client_new` | Rust core (richer errors) |

Both provide the same oracle interface via [`make_rpc_oracle`](@ref).

## Data Layout

ChemGP and rgpot use the same flat coordinate convention:

- **Positions**: `[x1,y1,z1, x2,y2,z2, ...]`
- **Atomic numbers**: `[Z1, Z2, ...]`
- **Cell**: flat 3x3 row-major `[a_x,a_y,a_z, b_x,b_y,b_z, c_x,c_y,c_z]`
- **Forces**: `[Fx1,Fy1,Fz1, ...]` (note: forces = -gradient)

## Architecture

```
                     +------------------+
                     |   ChemGP (Julia) |
                     |   GP Surrogate   |
                     +--------+---------+
                              |
                     oracle(x) -> (E, G)
                              |
                     +--------+---------+
                     |  rpc.jl client   |
                     |  (ccall to .so)  |
                     +--------+---------+
                              |
                     Cap'n Proto RPC (TCP)
                              |
              +---------------+---------------+
              |                               |
   +----------+----------+        +-----------+-----------+
   | rgpot potserv       |        | eonclient --serve     |
   | (LJ, CuH2, ...)    |        | (Metatomic, EAM, ...) |
   +---------------------+        +-----------+-----------+
                                              |
                                  +-----------+-----------+
                                  | MetatomicPotential    |
                                  | (PyTorch/metatensor)  |
                                  +-----------------------+
```

## Worked Example: PET-MAD with GP Optimization

End-to-end example using a universal ML potential (PET-MAD) served from
eOn with ChemGP performing GP-guided minimization.

### Terminal 1: Start the server

```bash
cd /path/to/eOn

# Create a config file for the model
cat > /tmp/petmad.ini << 'EOF'
[Metatomic]
model_path = /path/to/pet-mad-s-v1.1.0.pt
device = cpu
length_unit = angstrom
EOF

# Start serving on port 12345
pixi run -e dev-serve-mta bash -c \
    './bbdir/client/eonclient --serve "metatomic:12345" --config /tmp/petmad.ini'
```

### Terminal 2: Run ChemGP

```julia
using ChemGP

# Connect to the PET-MAD server
pot = RpcPotential("localhost", 12345,
                   Int32[29, 29],  # Cu2
                   Float64[20,0,0, 0,20,0, 0,0,20])

# Verify connectivity
E, F = calculate(pot, Float64[0,0,0, 2.2,0,0])
println("E = $E eV")  # Should be ~ -3.1 eV

# Create oracle and run GP minimization
oracle = make_rpc_oracle(pot)
kernel = MolInvDistSE(1.0, [0.5], Float64[])
x0 = Float64[0,0,0, 2.5,0,0]  # Initial guess
result = gp_minimize(oracle, x0, kernel; maxiter=10)

close(pot)
```

## Parallel NEB Evaluation

For NEB calculations, each iteration evaluates the oracle at N-2 intermediate
images. These evaluations are independent and can run in parallel when the
server supports concurrent connections.

### Server setup (gateway mode)

Start eOn serve with multiple replicas behind a single gateway port:

```bash
eonclient -p metatomic --serve-port 12345 --replicas 8 --gateway \
          --config petmad_hcn.ini
```

This creates 8 PET-MAD model instances behind port 12345, with round-robin
dispatch. Multiple client connections to the same port get load-balanced
across replicas.

### Client setup

Create an oracle pool and pass it to any NEB function:

```julia
using ChemGP

n_workers = min(Threads.nthreads(), neb_cfg.n_images - 2)
oracles = make_oracle_pool("localhost", 12345, atmnrs, box, n_workers)

# Standard NEB with parallel evaluation
result = neb_optimize(oracles, x_start, x_end; config = neb_cfg)

# GP-NEB AIE with parallel evaluation
result = gp_neb_aie(oracles, x_start, x_end, kernel; config = gp_cfg)
```

Launch Julia with multiple threads:

```bash
julia -t auto --project=. examples/petmad_hcn_neb.jl
# or specify explicitly:
julia -t 8 --project=. examples/petmad_hcn_neb.jl
```

A single `Function` oracle still works -- the pool is optional.

## Troubleshooting

**Could not find rgpot shared library**: Set `RGPOT_LIB_PATH` to the full
path of `libpot_client_bridge.so`, or set `RGPOT_BUILD_DIR` to the meson
builddir containing it.

**Connection refused**: Ensure the server is running and the port matches.

**Missing symbols**: Ensure rgpot was built with `-Dwith_rpc=true`.

**Energy explosion**: Check that atomic numbers and cell match the server's
expectation. Mismatched systems will produce nonsensical results.

**Timeout on startup**: Increase `startup_time` in `with_potserv` if the
server needs more time to initialize (e.g., loading large ML models).

**fmt ABI mismatch (metatomic builds)**: The eOn `dev-serve-mta` pixi
environment pins fmt to v11 to match torch's bundled fmt. If you see
linker errors about `fmt::v11` vs `fmt::v12` symbols, ensure you are
using the `dev-serve-mta` environment (not `serve` or `dev-mta` alone).
