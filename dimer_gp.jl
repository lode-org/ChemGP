# ==============================================================================
# Dimer Method Structures
# ==============================================================================

mutable struct DimerState
    R::Vector{Float64}           # Middle point coordinates
    orient::Vector{Float64}      # Dimer orientation (unit vector)
    dimer_sep::Float64           # Dimer separation (half-distance)
end

mutable struct DimerConfig
    # Convergence thresholds
    T_force_true::Float64        # True convergence threshold
    T_force_gp::Float64          # GP convergence threshold
    T_angle_rot::Float64         # Rotation angle threshold

    # Trust region
    trust_radius::Float64        # Max distance from training data
    ratio_at_limit::Float64      # Inter-atomic distance ratio limit

    # Iteration limits
    max_outer_iter::Int          # Max oracle calls
    max_inner_iter::Int          # Max GP steps per outer iteration
    max_rot_iter::Int            # Max rotations per translation

    # Step sizes
    alpha_trans::Float64         # Translation step size factor
end

# ==============================================================================
# Dimer Utilities
# ==============================================================================

function dimer_images(state::DimerState)
    """Get positions of dimer image points"""
    R0 = state.R
    R1 = state.R + state.dimer_sep * state.orient
    R2 = state.R - state.dimer_sep * state.orient
    return R0, R1, R2
end

function curvature(G0, G1, orient, dimer_sep)
    """Compute curvature along dimer direction"""
    return dot(G1 - G0, orient) / dimer_sep
end

function rotational_force(G0, G1, orient, dimer_sep)
    """Force perpendicular to dimer for rotation"""
    G_diff = (G1 - G0) / dimer_sep
    F_rot = G_diff - dot(G_diff, orient) * orient
    return F_rot
end

function translational_force(G0, orient)
    """Modified force for translation (inverted along dimer)"""
    F_parallel = dot(G0, orient) * orient
    F_perp = G0 - F_parallel
    F_trans = F_perp - F_parallel  # Invert force along dimer
    return F_trans
end

# ==============================================================================
# Distance-based stopping criteria
# ==============================================================================

function min_distance_to_data(x::Vector{Float64}, X_train::Matrix{Float64})
    """Minimum Euclidean distance from x to any training point"""
    N_train = size(X_train, 2)
    min_dist = Inf
    for i in 1:N_train
        d = norm(x - X_train[:, i])
        min_dist = min(min_dist, d)
    end
    return min_dist
end

function min_interatomic_distance(x::Vector{Float64})
    """Minimum distance between any two atoms in configuration"""
    N_atoms = div(length(x), 3)
    min_dist = Inf

    for i in 1:(N_atoms - 1)
        xi = x[(3i - 2):(3i)]
        for j in (i + 1):N_atoms
            xj = x[(3j - 2):(3j)]
            d = norm(xi - xj)
            min_dist = min(min_dist, d)
        end
    end

    return min_dist
end

function check_interatomic_ratio(
    x_new::Vector{Float64}, X_train::Matrix{Float64}, ratio_limit::Float64
)
    """Check if inter-atomic distances changed too much vs nearest training point"""
    N_atoms = div(length(x_new), 3)
    N_train = size(X_train, 2)

    # Compute inter-atomic distances for new point
    distances_new = Float64[]
    for i in 1:(N_atoms - 1)
        for j in (i + 1):N_atoms
            r_new = norm(x_new[(3i - 2):(3i)] - x_new[(3j - 2):(3j)])
            push!(distances_new, r_new)
        end
    end

    # Check against all training points
    for k in 1:N_train
        x_train = X_train[:, k]
        distances_train = Float64[]

        for i in 1:(N_atoms - 1)
            for j in (i + 1):N_atoms
                r_train = norm(x_train[(3i - 2):(3i)] - x_train[(3j - 2):(3j)])
                push!(distances_train, r_train)
            end
        end

        # Check if all ratios are within bounds
        ratios = distances_new ./ (distances_train .+ 1e-10)
        max_log_ratio = maximum(abs.(log.(ratios)))

        if max_log_ratio < abs(log(ratio_limit))
            return true  # Found a close enough training point
        end
    end

    return false  # All training points too different
end

# ==============================================================================
# Dimer Rotation
# ==============================================================================

function rotate_dimer!(
    state::DimerState, model::GPModel, config::DimerConfig; verbose=false
)
    """Rotate dimer to find lowest curvature mode"""

    for rot_iter in 1:config.max_rot_iter
        # Get current images
        R0, R1, R2 = dimer_images(state)

        # Predict at both endpoints
        pred1 = predict(model, reshape(R1, :, 1))
        pred2 = predict(model, reshape(R2, :, 1))

        G1 = pred1[2:end]
        G2 = pred2[2:end]
        G0 = -(G1 + G2) / 2  # Gradient at center by finite difference

        # Compute rotational force
        F_rot = rotational_force(G0, G1, state.orient, state.dimer_sep)
        F_rot_norm = norm(F_rot)

        if F_rot_norm < 1e-10
            verbose && println("  Rotation converged (F_rot ≈ 0)")
            break
        end

        # Estimate rotation angle
        C = curvature(G0, G1, state.orient, state.dimer_sep)
        dtheta = 0.5 * atan(F_rot_norm / (abs(C) + 1e-10))

        if dtheta < config.T_angle_rot
            verbose && println(
                "  Rotation converged (θ = $(round(dtheta, digits=5)) < $(config.T_angle_rot))",
            )
            break
        end

        # Rotate orientation
        b1 = F_rot / F_rot_norm  # Unit vector perpendicular to orient
        orient_new = cos(dtheta) * state.orient + sin(dtheta) * b1
        state.orient = orient_new / norm(orient_new)  # Renormalize

        verbose && println(
            "  Rotation $rot_iter: θ = $(round(dtheta, digits=5)), |F_rot| = $(round(F_rot_norm, digits=5))",
        )
    end
end

# ==============================================================================
# Main GP-Dimer Algorithm
# ==============================================================================

function gp_dimer_optimize(
    oracle_func,
    x_init::Vector{Float64},
    orient_init::Vector{Float64},
    X_train_init::Matrix{Float64},
    y_vals_init::Vector{Float64},
    y_grads_init::Vector{Float64},
    kernel_params;
    dimer_sep=0.01,
    T_force_true=1e-3,
    T_force_gp=1e-2,
    T_angle_rot=1e-3,
    trust_radius=0.1,
    ratio_at_limit=2/3,
    max_outer_iter=50,
    max_inner_iter=100,
    max_rot_iter=10,
    alpha_trans=0.01,
    verbose=true,
)

    # Initialize configuration
    config = DimerConfig(
        T_force_true,
        T_force_gp,
        T_angle_rot,
        trust_radius,
        ratio_at_limit,
        max_outer_iter,
        max_inner_iter,
        max_rot_iter,
        alpha_trans,
    )

    # Initialize dimer state
    orient_init = orient_init / norm(orient_init)  # Ensure unit vector
    state = DimerState(copy(x_init), orient_init, dimer_sep)

    # Initialize training data
    X_train = copy(X_train_init)
    y_vals = copy(y_vals_init)
    y_grads = copy(y_grads_init)

    D = length(x_init)
    oracle_calls = size(X_train, 2)

    verbose && println("="^70)
    verbose && println("GP-Dimer Saddle Point Search")
    verbose && println("="^70)
    verbose && println("Initial training points: $oracle_calls")
    verbose && println("Dimer separation: $dimer_sep")
    verbose && println("Trust radius: $trust_radius")
    verbose && println()

    # History
    history = Dict(
        "E_true" => Float64[],
        "F_true" => Float64[],
        "curv_true" => Float64[],
        "oracle_calls" => Int[],
    )

    for outer_iter in 1:max_outer_iter
        verbose && println("─"^70)
        verbose && println("OUTER ITERATION $outer_iter (Oracle calls: $oracle_calls)")
        verbose && println("─"^70)

        # Train GP on current data
        y_mean = mean(y_vals)
        y_std = max(std(y_vals), 1e-10)

        y_norm = (y_vals .- y_mean) ./ y_std
        g_norm = y_grads ./ y_std
        y_gp = vcat(y_norm, g_norm)

        # Create kernel and model
        k = MolInvDistSE(
            kernel_params.signal_var,
            kernel_params.inv_lengthscales,
            kernel_params.frozen_coords,
            kernel_params.feature_map,
        )

        model = GPModel(k, X_train, y_gp; noise_var=1e-2, grad_noise_var=1e-1, jitter=1e-3)

        verbose && println("\n📊 Training GP on $(size(X_train, 2)) points...")
        train_model!(model; iterations=300)

        # Inner loop: Optimize on GP surface
        gp_converged = false
        trust_violated = false
        ratio_violated = false

        for inner_iter in 1:max_inner_iter
            # Rotate dimer to find lowest curvature mode
            rotate_dimer!(state, model, config; verbose=false)

            # Predict at current position
            R0, R1, _ = dimer_images(state)

            pred0 = predict(model, reshape(R0, :, 1))
            pred1 = predict(model, reshape(R1, :, 1))

            E0_pred = pred0[1] * y_std + y_mean
            G0_pred = pred0[2:end] .* y_std
            G1_pred = pred1[2:end] .* y_std

            F_trans = translational_force(G0_pred, state.orient)
            F_norm = norm(F_trans)
            C = curvature(G0_pred, G1_pred, state.orient, state.dimer_sep)

            # Distance to nearest training point
            min_dist = min_distance_to_data(state.R, X_train)

            if inner_iter % 10 == 0 || inner_iter == 1
                verbose && @printf(
                    "  GP step %3d: E = %8.4f | |F| = %.5f | C = %+.3e | d_min = %.4f\n",
                    inner_iter,
                    E0_pred,
                    F_norm,
                    C,
                    min_dist
                )
            end

            # Check GP convergence
            if F_norm < config.T_force_gp
                verbose && println("  ✓ Converged on GP surface!")
                gp_converged = true
                break
            end

            # Propose translation step
            # For saddle search, use adaptive step based on curvature
            if abs(C) > 1e-6
                step_size = min(config.alpha_trans, 0.1 * F_norm / abs(C))
            else
                step_size = config.alpha_trans
            end

            R_new = state.R + step_size * F_trans

            # Check trust radius
            min_dist_new = min_distance_to_data(R_new, X_train)
            if min_dist_new > trust_radius
                verbose && @printf(
                    "  ⚠ Trust radius exceeded (%.4f > %.4f)\n",
                    min_dist_new,
                    trust_radius
                )
                # Take step to boundary
                scale = trust_radius / min_dist_new * 0.95
                state.R = state.R + scale * step_size * F_trans
                trust_violated = true
                break
            end

            # Check inter-atomic distance ratios
            if !check_interatomic_ratio(R_new, X_train, ratio_at_limit)
                verbose && println("  ⚠ Inter-atomic distance ratio violated")
                ratio_violated = true
                break
            end

            # Accept step
            state.R = R_new
        end

        # Call oracle at current position
        verbose && println("\n🔬 Calling Oracle...")
        E_true, G_true = oracle_func(state.R)
        oracle_calls += 1

        # Also evaluate R1 for curvature
        _, R1, _ = dimer_images(state)
        E1_true, G1_true = oracle_func(R1)
        oracle_calls += 1

        C_true = curvature(G_true, G1_true, state.orient, state.dimer_sep)
        F_trans_true = translational_force(G_true, state.orient)
        F_norm_true = norm(F_trans_true)

        verbose && @printf(
            "  True: E = %8.4f | |F| = %.5f | C = %+.3e\n", E_true, F_norm_true, C_true
        )

        # Store history
        push!(history["E_true"], E_true)
        push!(history["F_true"], F_norm_true)
        push!(history["curv_true"], C_true)
        push!(history["oracle_calls"], oracle_calls)

        # Add to training set
        X_train = hcat(X_train, state.R, R1)
        push!(y_vals, E_true, E1_true)
        append!(y_grads, G_true)
        append!(y_grads, G1_true)

        # Check true convergence
        if F_norm_true < config.T_force_true && abs(C_true) < 0.1
            verbose && println("\n" * "="^70)
            verbose && println("🎉 CONVERGED TO SADDLE POINT!")
            verbose && println("="^70)
            verbose && @printf("Final Energy:     %.6f\n", E_true)
            verbose && @printf("Final |F|:        %.6f\n", F_norm_true)
            verbose && @printf("Final Curvature:  %+.6f\n", C_true)
            verbose && println("Oracle calls:     $oracle_calls")
            break
        end

        if outer_iter == max_outer_iter
            verbose && println("\n⚠ Maximum outer iterations reached")
            break
        end

        verbose && println()
    end

    return state, X_train, y_vals, y_grads, history
end

# ==============================================================================
# Example Usage for LJ13 Saddle Point Search
# ==============================================================================

function run_gp_dimer_lj13()
    println("Starting GP-Dimer for LJ13 Saddle Point Search")

    # Use the oracle from before
    include("example_system.jl")  # Assumes lennard_jones_oracle is defined

    # Start from a perturbed minimum
    N_atoms = 13
    sys = make_random_cluster(N_atoms)
    x_init = extract_flat(sys)

    # Random initial orientation for dimer
    orient_init = randn(length(x_init))
    orient_init /= norm(orient_init)

    # Generate initial training data around starting point
    X_train = Matrix{Float64}(undef, length(x_init), 0)
    y_vals = Float64[]
    y_grads = Float64[]

    for k in 1:5
        perturb = (rand(length(x_init)) .- 0.5) .* 0.15
        x = x_init + perturb

        E, G = lennard_jones_oracle(x)
        if E < 1e6
            X_train = hcat(X_train, x)
            push!(y_vals, E)
            append!(y_grads, G)
        end
    end

    # Kernel parameters
    kernel_params = (
        signal_var=1.0, inv_lengthscales=[0.5], frozen_coords=Float64[], feature_map=Int[]
    )

    # Run GP-Dimer
    state, X_final, y_final, g_final, history = gp_dimer_optimize(
        x -> lennard_jones_oracle(x),
        x_init,
        orient_init,
        X_train,
        y_vals,
        y_grads,
        kernel_params;
        dimer_sep=0.01,
        T_force_true=5e-3,
        T_force_gp=2e-2,
        trust_radius=0.1,
        max_outer_iter=30,
        max_inner_iter=50,
    )

    println("\nFinal saddle point position saved.")
    println("Total oracle calls: $(history["oracle_calls"][end])")

    return state, history
end
