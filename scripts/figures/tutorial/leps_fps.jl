# LEPS-4: Farthest point sampling subset selection
#
# Collects ~50 training points from a GP-NEB run on LEPS, applies FPS
# to select 20 points, projects to 2D via PCA on inverse distance features,
# and colors selected vs pruned points.

using ChemGP
using KernelFunctions
using LinearAlgebra
using Statistics
include(joinpath(@__DIR__, "common.jl"))

# --- Run GP-NEB AIE to collect training data ---
println("Running GP-NEB AIE on LEPS to collect training points...")
# 3 H atoms -> 1 unique pair type (H-H), 3 inv-distance features sharing 1 lengthscale
kernel = MolInvDistSE([1, 1, 1], Float64[])

config = NEBConfig(;
    images=7,
    spring_constant=5.0,
    climbing_image=true,
    conv_tol=0.5,  # Loose tolerance -- just need training points, not convergence
    gp_train_iter=50,
    max_outer_iter=10,
    trust_radius=0.1,
    verbose=true,
)

result = gp_neb_aie(
    leps_energy_gradient,
    Float64.(LEPS_REACTANT),
    Float64.(LEPS_PRODUCT),
    kernel;
    config=config,
)

# Collect all images from the final path as candidate training points
# (In a real run, we would accumulate from the history; here we sample
# from multiple NEB runs with perturbations)
candidates = hcat(result.path.images...)

# Add perturbed copies to get ~50 candidates
using Random
Random.seed!(42)
rng = MersenneTwister(42)
n_needed = max(0, 50 - size(candidates, 2))
perturbed_cols = [
    result.path.images[rand(rng, 1:length(result.path.images))] .+ 0.1 * randn(rng, 9) for
    _ in 1:n_needed
]
if !isempty(perturbed_cols)
    candidates = hcat(candidates, perturbed_cols...)
end

n_candidates = size(candidates, 2)
println("Candidate points: $n_candidates")

# --- Compute inverse distance features ---
frozen = Float64[]  # No frozen atoms
feature_matrix = zeros(3, n_candidates)  # 3 inverse distances for 3 atoms
for i in 1:n_candidates
    feature_matrix[:, i] = compute_inverse_distances(candidates[:, i], frozen)
end

# --- FPS: select 20 points ---
n_select = 20
# Start with the first point as seed
X_selected = reshape(feature_matrix[:, 1], 3, 1)
remaining = feature_matrix[:, 2:end]
remaining_mat = Matrix(remaining)

euclidean_dist(x, y) = LinearAlgebra.norm(x - y)
selected_indices = farthest_point_sampling(
    remaining_mat, X_selected, n_select - 1; distance_fn=euclidean_dist
)
# Adjust indices (since we started from index 2)
all_selected = vcat([1], selected_indices .+ 1)

println("Selected $(length(all_selected)) points via FPS")

# --- PCA for 2D projection ---
F = feature_matrix  # 3 x N
F_centered = F .- mean(F; dims=2)
C = F_centered * F_centered' / n_candidates
eigvals, eigvecs = eigen(C; sortby=x -> -x)
proj = eigvecs[:, 1:2]' * F_centered  # 2 x N

# Split into selected and pruned
is_selected = falses(n_candidates)
for idx in all_selected
    is_selected[idx] = true
end

proj_sel = proj[:, is_selected]
proj_pru = proj[:, .!is_selected]

# --- Plot ---
set_theme!(PUBLICATION_THEME)

fig = Figure(; size=(504, 400))
ax = Axis(fig[1, 1]; xlabel=L"$\mathrm{PC}_1$", ylabel=L"$\mathrm{PC}_2$")

# Pruned points (gray)
scatter!(
    ax,
    proj_pru[1, :],
    proj_pru[2, :];
    marker=:circle,
    markersize=8,
    color=(:gray70, 0.6),
    strokecolor=:gray50,
    strokewidth=0.5,
    label="Pruned",
)

# Selected points (colored)
scatter!(
    ax,
    proj_sel[1, :],
    proj_sel[2, :];
    marker=:diamond,
    markersize=10,
    color=RUHI.teal,
    strokecolor=:white,
    strokewidth=1.0,
    label="FPS selected",
)

Legend(fig[2, 1], ax; orientation=:horizontal, tellwidth=false, tellheight=true)

save_figure(fig, "leps_fps")
