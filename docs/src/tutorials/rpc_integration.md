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
1. A server runs a potential (e.g., metatensor, EAM, ML model)
2. ChemGP connects as a client and uses the potential as its oracle

## Prerequisites

Build the rgpot shared library from source:

```bash
git clone https://github.com/OmniPotentRPC/rgpot.git
cd rgpot
cargo build --release --features rpc
```

This produces `libpot_client_bridge.so` (or `.dylib` on macOS).

## Connecting to a Server

Start the potential server (see rgpot documentation):

```bash
./potserv 12345 CuH2
```

Then connect from Julia:

```julia
using ChemGP

pot = RpcPotential(
    "localhost", 12345,
    "/path/to/libpot_client_bridge.so",
    Int32[29, 29, 1, 1],  # Atomic numbers (Cu₂H₂)
    Float64[10,0,0, 0,10,0, 0,0,10],  # Simulation cell (flat 3x3)
)
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

## Troubleshooting

**Connection refused**: Ensure the server is running and the port matches.

**Library not found**: Provide the full absolute path to the shared library.

**Missing symbols**: Ensure rgpot was built with the `rpc` feature flag.

**Energy explosion**: Check that atomic numbers and cell match the server's
expectation. Mismatched systems will produce nonsensical results.
