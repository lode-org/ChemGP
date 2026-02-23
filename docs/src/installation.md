# Installation

## Using Julia's Package Manager

ChemGP is not yet registered in the General registry. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/lode-org/ChemGP.jl")
```

Or in Pkg mode (press `]` at the REPL):

```
pkg> add https://github.com/lode-org/ChemGP.jl
```

## Using pixi

If you are developing ChemGP or running the examples, the repository includes a
[pixi](https://pixi.sh) environment:

```bash
git clone https://github.com/lode-org/ChemGP.jl.git
cd ChemGP.jl
pixi install
pixi r julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running Tests

```julia
using Pkg
Pkg.test("ChemGP")
```

Or from the command line:

```bash
pixi r julia --project=. -e 'using Pkg; Pkg.test()'
```

## Dependencies

ChemGP depends on the following Julia packages:

| Package | Purpose |
|:--------|:--------|
| `ForwardDiff.jl` | Automatic differentiation for kernel derivative blocks |
| `KernelFunctions.jl` | Base `Kernel` type for the type hierarchy |
| `LinearAlgebra` | Cholesky factorization, matrix operations |
| `Optim.jl` | Nelder-Mead (hyperparameters) and L-BFGS (GP surface optimization) |
| `ParameterHandling.jl` | Positive-constrained parameter transformations |
| `Printf` | Formatted output during optimization |
| `Statistics` | Mean and standard deviation for normalization |

### Optional: AtomsBase Integration

ChemGP provides a package extension that activates when
[AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl) is loaded. This
allows converting between AtomsBase systems and ChemGP's flat-vector
convention. To use it:

```julia
using AtomsBase
using AtomsIO    # for reading structure files
using ChemGP

sys = load_system("structure.extxyz")
(; positions, atomic_numbers, box) = chemgp_coords(sys)
```

Install AtomsBase and AtomsIO in your environment:

```julia
using Pkg
Pkg.add(["AtomsBase", "AtomsIO"])
```

AtomsIO supports extended XYZ, POSCAR, CIF, and other common formats.
See [AtomsBase Integration](@ref) for the full API.

### Optional: RPC Potentials

To use remote potentials via rgpot, you need to build the
[rgpot](https://github.com/OmniPotentRPC/rgpot) shared library separately.
See the [RPC Integration](@ref) tutorial for details.
