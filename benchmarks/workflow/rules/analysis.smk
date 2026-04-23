rule aggregate:
    input:
        "results/minimize/summary.json",
        "results/dimer/summary.json",
        "results/neb/summary.json",
    output:
        "results/summary/benchmark_table.csv",
        "results/summary/benchmark_table.md",
    shell:
        "python analysis/summarize.py"


rule plot_runtime:
    input:
        "results/summary/benchmark_table.csv",
    output:
        "results/summary/runtime.png",
    shell:
        "python analysis/plot_runtime.py"


rule plot_failures:
    input:
        "results/summary/benchmark_table.csv",
    output:
        "results/summary/failures.png",
    shell:
        "python analysis/plot_failures.py"


rule plot_convergence:
    input:
        "results/summary/benchmark_table.csv",
    output:
        "results/summary/convergence.png",
    shell:
        "python analysis/plot_convergence.py"
