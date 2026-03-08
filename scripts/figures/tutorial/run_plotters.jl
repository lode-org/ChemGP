# Run all tutorial plotters in a single Julia process.
#
# Usage: julia --project=scripts/figures/tutorial -- scripts/figures/tutorial/run_plotters.jl
#
# Loads CairoMakie once, then includes each plot_*.jl sequentially.
# ~50x faster than spawning a new Julia process per plotter.

plotdir = joinpath(@__DIR__, "plotters")
scripts = filter(f -> startswith(f, "plot_") && endswith(f, ".jl"), readdir(plotdir))
sort!(scripts)

n_ok = 0
n_err = 0
for s in scripts
    path = joinpath(plotdir, s)
    print("  $s ... ")
    try
        include(path)
        n_ok += 1
    catch e
        n_err += 1
        println("ERROR")
        showerror(stderr, e, catch_backtrace())
        println(stderr)
    end
end

println("\n$n_ok plotters succeeded, $n_err failed ($(length(scripts)) total)")
n_err > 0 && exit(1)
