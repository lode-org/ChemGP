using Pkg
Pkg.activate(".")
using ChemGP
using AtomsBase
using AtomsIO
using Unitful
using LinearAlgebra
using Statistics
using Optim

# ==============================================================================
# 1. LENNARD-JONES ORACLE (Ground Truth)
# ==============================================================================

function lennard_jones_oracle(sys::AbstractSystem)
    pos_unitful = position(sys, :)
    pos = [ustrip.(p) for p in pos_unitful] 
    
    N = length(pos)
    E_total = 0.0
    Forces = zeros(Float64, 3, N)
    
    ϵ = 1.0
    σ = 1.0
    
    for i in 1:N
        for j in (i+1):N
            rij = pos[j] - pos[i] 
            r_sq = dot(rij, rij)
            r = sqrt(r_sq)
            
            sr6  = (σ / r)^6
            sr12 = sr6^2
            
            E_total += 4 * ϵ * (sr12 - sr6)
            
            f_scalar = (24 * ϵ / r_sq) * (2 * sr12 - sr6)
            f_vec = f_scalar .* rij
            
            Forces[:, i] -= f_vec
            Forces[:, j] += f_vec
        end
    end
    
    return E_total, vec(-Forces)
end

# ==============================================================================
# 2. SETUP LJ13 CLUSTER
# ==============================================================================

function make_random_cluster(N_atoms; min_dist=1.5, max_attempts=10000)
    coords = zeros(3, N_atoms)
    coords[:, 1] = zeros(3)
    
    for i in 2:N_atoms
        placed = false
        for attempt in 1:max_attempts
            r = 2.0 * N_atoms^(1/3) * rand()^(1/3)
            θ = 2π * rand()
            φ = acos(2rand() - 1)
            
            candidate = [r*sin(φ)*cos(θ), r*sin(φ)*sin(θ), r*cos(φ)]
            
            too_close = false
            for j in 1:(i-1)
                dist = norm(candidate - coords[:, j])
                if dist < min_dist
                    too_close = true
                    break
                end
            end
            
            if !too_close
                coords[:, i] = candidate
                placed = true
                break
            end
        end
        
        if !placed
            error("Could not place atom $i after $max_attempts attempts")
        end
    end
    
    atoms = [Atom(:Ar, coords[:, i] * 1.0u"Å") for i in 1:N_atoms]
    box_size = 3.0 * N_atoms^(1/3)
    return FlexibleSystem(atoms, 
        [[box_size,0.0,0.0], [0.0,box_size,0.0], [0.0,0.0,box_size]]u"Å", 
        (false, false, false))
end

function extract_flat(sys)
    p = position(sys, :)
    vcat([ustrip.(x) for x in p]...)
end

# ==============================================================================
# 3. GP OBJECTIVE FUNCTION FOR OPTIMIZATION
# ==============================================================================

"""
Wrapper for GP prediction that provides objective and gradient for Optim.jl
"""
mutable struct GPObjective
    model::GPModel
    y_mean::Float64
    y_std::Float64
    X_train::Matrix{Float64}
    trust_radius::Float64
    penalty_coeff::Float64  # Penalty for trust region violation
    
    function GPObjective(model, y_mean, y_std, X_train, trust_radius)
        new(model, y_mean, y_std, X_train, trust_radius, 1e3)
    end
end

function (obj::GPObjective)(x::Vector{Float64})
    """Evaluate GP energy (objective)"""
    preds = predict(obj.model, reshape(x, :, 1))
    E = preds[1] * obj.y_std + obj.y_mean
    
    # Add soft trust region penalty
    min_dist = minimum([norm(x - obj.X_train[:, i]) for i in 1:size(obj.X_train, 2)])
    if min_dist > obj.trust_radius
        penalty = obj.penalty_coeff * (min_dist - obj.trust_radius)^2
        E += penalty
    end
    
    return E
end

function gradient!(G::Vector{Float64}, obj::GPObjective, x::Vector{Float64})
    """Evaluate GP gradient (in-place)"""
    preds = predict(obj.model, reshape(x, :, 1))
    G_pred = preds[2:end] .* obj.y_std
    
    # Add trust region penalty gradient
    min_dist = Inf
    nearest_idx = 1
    for i in 1:size(obj.X_train, 2)
        d = norm(x - obj.X_train[:, i])
        if d < min_dist
            min_dist = d
            nearest_idx = i
        end
    end
    
    if min_dist > obj.trust_radius
        direction = (x - obj.X_train[:, nearest_idx]) / (min_dist + 1e-10)
        penalty_grad = 2 * obj.penalty_coeff * (min_dist - obj.trust_radius) * direction
        G .= G_pred + penalty_grad
    else
        G .= G_pred
    end
    
    return G
end

# ==============================================================================
# 4. GP MINIMIZATION LOOP WITH PROPER OPTIMIZATION
# ==============================================================================

function run_lj13()
    println("--- Starting LJ13 GP Minimization with Trust Region ---")
    
    N_atoms = 13
    sys_curr = make_random_cluster(N_atoms)
    
    mov_types = ones(Int, N_atoms)
    fro_types = Int[]
    pair_map = ones(Int, 1, 1) 
    frozen_coords = Float64[]
    
    X_train = Matrix{Float64}(undef, 3*N_atoms, 0)
    y_vals = Float64[]
    y_grads = Float64[]
    traj = [sys_curr]
    
    # Hyperparameters
    trust_radius = 0.1
    true_conv_tol = 5e-3
    max_outer_iter = 500
    gp_opt_tol = 1e-2  # Tolerance for GP optimization
    
    println("Generating initial seed data...")
    x_start = extract_flat(sys_curr)
    
    # Initial training set
    E_start, G_start = lennard_jones_oracle(sys_curr)
    println("Initial energy: $(round(E_start, digits=4))")
    
    X_train = hcat(X_train, x_start)
    push!(y_vals, E_start)
    append!(y_grads, G_start)
    
    # Add perturbed points
    for k in 1:4
        perturb = (rand(length(x_start)) .- 0.5) .* 0.1
        x_p = x_start + perturb
        
        at_p = [Atom(:Ar, x_p[(i-1)*3+1:i*3]u"Å") for i in 1:N_atoms]
        sys_p = FlexibleSystem(at_p, cell_vectors(sys_curr), periodicity(sys_curr))
        
        E, G = lennard_jones_oracle(sys_p)
        
        if E < 1e6
            X_train = hcat(X_train, x_p)
            push!(y_vals, E)
            append!(y_grads, G)
            println("  Point $k: E = $(round(E, digits=2))")
        end
    end
    
    x_curr = X_train[:, 1]
    oracle_calls = size(X_train, 2)
    
    println("\n=== Starting Optimization ===")
    println("Trust radius: $trust_radius Å")
    println("Initial training points: $oracle_calls\n")
    
    for outer_step in 1:max_outer_iter
        println("─"^60)
        println("OUTER ITERATION $outer_step (Oracle calls: $oracle_calls)")
        println("─"^60)
        
        # ===== Train GP on current data =====
        y_mean = mean(y_vals)
        y_std = max(std(y_vals), 1e-10)
        
        y_norm = (y_vals .- y_mean) ./ y_std
        g_norm = y_grads ./ y_std
        y_gp = vcat(y_norm, g_norm)
        
        k = MolInvDistSE(1.0, [0.5], frozen_coords, mov_types, fro_types, pair_map)
        model = GPModel(k, X_train, y_gp; 
            noise_var=1e-2, 
            grad_noise_var=1e-1, 
            jitter=1e-3)
        
        println("\n📊 Training GP on $(size(X_train, 2)) points...")
        train_model!(model, iterations=300)
        
        # ===== Optimize on GP surface using L-BFGS =====
        println("\n🔍 Optimizing on GP surface with L-BFGS...")
        
        # Create objective function
        gp_obj = GPObjective(model, y_mean, y_std, X_train, trust_radius)
        
        # Optimize using L-BFGS with automatic differentiation
        result = optimize(
            gp_obj,
            (G, x) -> gradient!(G, gp_obj, x),
            x_curr,
            LBFGS(),
            Optim.Options(
                g_tol = gp_opt_tol,
                iterations = 100,
                show_trace = true,
                store_trace = true
            )
        )
        
        x_curr = Optim.minimizer(result)
        
        # Check convergence on GP
        preds = predict(model, reshape(x_curr, :, 1))
        pred_E = preds[1] * y_std + y_mean
        pred_grad = preds[2:end] .* y_std
        gnorm = norm(pred_grad)
        
        min_dist = minimum([norm(x_curr - X_train[:, i]) for i in 1:size(X_train, 2)])
        
        println("  Optimization result:")
        println("    Iterations: $(Optim.iterations(result))")
        println("    Converged: $(Optim.converged(result))")
        println("    E_pred: $(round(pred_E, digits=4))")
        println("    |∇E|: $(round(gnorm, digits=5))")
        println("    Distance to data: $(round(min_dist, digits=4))")
        
        # ===== Call Oracle =====
        println("\n🔬 Calling Oracle...")
        
        at_new = [Atom(:Ar, x_curr[(i-1)*3+1:i*3]u"Å") for i in 1:N_atoms]
        sys_new = FlexibleSystem(at_new, cell_vectors(sys_curr), periodicity(sys_curr))
        
        E_true, G_true = lennard_jones_oracle(sys_new)
        G_true_norm = norm(G_true)
        oracle_calls += 1
        
        println("  True: E = $(round(E_true, digits=4)) | |∇E| = $(round(G_true_norm, digits=5))")
        
        # Prediction error
        E_error = abs(E_true - pred_E)
        println("  Prediction error: ΔE = $(round(E_error, digits=4))")
        
        # Sanity check
        if E_true > 1e6
            println("  ⚠ Energy exploded - resetting")
            x_curr = X_train[:, end]
            continue
        end
        
        # Add to training set
        X_train = hcat(X_train, x_curr)
        push!(y_vals, E_true)
        append!(y_grads, G_true)
        push!(traj, sys_new)
        
        # ===== Check TRUE convergence =====
        if G_true_norm < true_conv_tol
            println("\n" * "="^60)
            println("🎉 CONVERGED ON TRUE SURFACE!")
            println("="^60)
            println("Final Energy: $(round(E_true, digits=6))")
            println("Final |∇E|:   $(round(G_true_norm, digits=6))")
            println("Oracle calls: $oracle_calls")
            break
        end
        
        println()
    end
    
    AtomsIO.save_trajectory("lj13_traj.xyz", traj)
    println("\n✓ Saved trajectory to lj13_traj.xyz")
    println("✓ Total oracle calls: $oracle_calls")
end

run_lj13()
