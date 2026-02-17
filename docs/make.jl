using Documenter
using ChemGP

makedocs(;
    modules = [ChemGP],
    sitename = "ChemGP.jl",
    authors = "Rohit Goswami and contributors",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://lode-org.github.io/ChemGP.jl",
    ),
    repo = Remotes.GitHub("lode-org", "ChemGP.jl"),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Tutorials" => [
            "Quick Start" => "tutorials/quickstart.md",
            "GP Basics" => "tutorials/gp_basics.md",
            "Molecular Kernels" => "tutorials/molecular_kernels.md",
            "Kernel Comparison" => "tutorials/kernel_comparison.md",
            "Minimization" => "tutorials/minimization.md",
            "Dimer Method" => "tutorials/dimer_method.md",
            "NEB Method" => "tutorials/neb_method.md",
            "RPC Integration" => "tutorials/rpc_integration.md",
        ],
        "Guides" => [
            "Kernel Design" => "guides/kernel_design.md",
            "Trust Regions" => "guides/trust_regions.md",
        ],
        "API Reference" => [
            "GP Core" => "api/gp_core.md",
            "Kernels" => "api/kernels.md",
            "Optimizers" => "api/optimizers.md",
            "Oracles" => "api/oracles.md",
            "Utilities" => "api/utilities.md",
        ],
        "References" => "references.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(;
    repo = "github.com/lode-org/ChemGP.jl",
    devbranch = "main",
    push_preview = true,
)
