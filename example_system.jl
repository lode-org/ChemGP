using Pkg
Pkg.activate(".")
using ChemGP
using AtomsBase
using AtomsIO
using Unitful
using LinearAlgebra
using Statistics

# ==============================================================================
# 1. LENNARD-JONES ORACLE (Ground Truth)
# ==============================================================================

function lennard_jones_oracle(sys::AbstractSystem)
    # position(sys, :) returns a Vector of SVector{3, Quantity}
    pos_unitful = position(sys, :)
    pos = [ustrip.(p) for p in pos_unitful] 
    
    N = length(pos)
    E_total = 0.0
    Forces = zeros(Float64, 3, N)
    
    # LJ Parameters (Reduced Units)
    ϵ = 1.0
    σ = 1.0
    
    for i in 1:N
        for j in (i+1):N
            # Distance vector (Now pure Float64)
            rij = pos[j] - pos[i] 
            r_sq = dot(rij, rij)
            r = sqrt(r_sq)
            
            # Dimensionless Math
            # V = 4ϵ * ((σ/r)^12 - (σ/r)^6)
            sr6  = (σ / r)^6
            sr12 = sr6^2
            
            E_total += 4 * ϵ * (sr12 - sr6)
            
            # Force Magnitude (F = -dV/dr)
            # F_scalar = (24ϵ/r^2) * (2*(σ/r)^12 - (σ/r)^6)
            f_scalar = (24 * ϵ / r_sq) * (2 * sr12 - sr6)
            f_vec = f_scalar .* rij
            
            Forces[:, i] -= f_vec
            Forces[:, j] += f_vec
        end
    end
    
    # Return Energy and Gradient (-Force)
    return E_total, vec(-Forces)
end

# ==============================================================================
# 2. SETUP LJ13 CLUSTER
# ==============================================================================

function make_random_cluster(N_atoms; min_dist=1.5, max_attempts=10000)
    """Generate cluster avoiding close contacts"""
    coords = zeros(3, N_atoms)
    
    # First atom at origin
    coords[:, 1] = zeros(3)
    
    # Place remaining atoms with minimum distance constraint
    for i in 2:N_atoms
        placed = false
        for attempt in 1:max_attempts
            # Try random position in sphere of radius ~2*N_atoms^(1/3)
            r = 2.0 * N_atoms^(1/3) * rand()^(1/3)
            θ = 2π * rand()
            φ = acos(2rand() - 1)
            
            candidate = [r*sin(φ)*cos(θ), r*sin(φ)*sin(θ), r*cos(φ)]
            
            # Check distances to all existing atoms
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
    # Helper to get flat vector for GP
    p = position(sys, :)
    vcat([ustrip.(x) for x in p]...)
end

# ==============================================================================
# 3. GP MINIMIZATION LOOP
# ==============================================================================

function run_lj13()
    println("--- Starting LJ13 GP Minimization ---")
    
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
    
    println("Generating initial seed data...")
    x_start = extract_flat(sys_curr)
    
    # Evaluate starting point
    E_start, G_start = lennard_jones_oracle(sys_curr)
    println("Initial energy: $E_start")
    
    X_train = hcat(X_train, x_start)
    push!(y_vals, E_start)
    append!(y_grads, G_start)
    
    # Generate diverse training points with SMALL perturbations
    for k in 1:9
        # Small perturbations to avoid creating bad contacts
        perturb = (rand(length(x_start)) .- 0.5) .* 0.1  # ±0.05 Å
        x_p = x_start + perturb
        
        at_p = [Atom(:Ar, x_p[(i-1)*3+1:i*3]u"Å") for i in 1:N_atoms]
        sys_p = FlexibleSystem(at_p, cell_vectors(sys_curr), periodicity(sys_curr))
        
        E, G = lennard_jones_oracle(sys_p)
        
        println("Point $k: E = $(round(E, digits=2))")
        
        # Skip if energy is absurdly high (bad contact)
        if E > 1e6
            println("  ⚠ Skipping - energy too high")
            continue
        end
        
        X_train = hcat(X_train, x_p)
        push!(y_vals, E)
        append!(y_grads, G)
    end
    
    if size(X_train, 2) < 3
        error("Not enough valid training points. Initial structure may be bad.")
    end
    
    println("\nStarting optimization with $(size(X_train, 2)) training points...")
    
    x_curr = X_train[:, 1]
    
    for step in 1:20
        # ===== NORMALIZE DATA =====
        y_mean = mean(y_vals)
        y_std = std(y_vals)
        
        if y_std < 1e-10
            println("⚠ Energy variance too small, using unit scaling")
            y_std = 1.0
        end
        
        # Normalize energies and gradients
        y_norm = (y_vals .- y_mean) ./ y_std
        g_norm = y_grads ./ y_std
        
        y_gp = vcat(y_norm, g_norm)
        # ==========================
        
        # Better kernel initialization
        k = MolInvDistSE(1.0, [0.5], frozen_coords, mov_types, fro_types, pair_map)
        
        # Higher noise for stability
        model = GPModel(k, X_train, y_gp; 
            noise_var=1e-2, 
            grad_noise_var=1e-1, 
            jitter=1e-3)
        
        println("\nStep $step: Training...")
        train_model!(model, iterations=300)
        
        # Predict
        preds = predict(model, reshape(x_curr, :, 1))
        
        # Denormalize predictions
        pred_E = preds[1] * y_std + y_mean
        pred_grad = preds[2:end] .* y_std
        
        gnorm = norm(pred_grad)
        println("  Predicted: E = $(round(pred_E, digits=4)) | |∇E| = $(round(gnorm, digits=4))")
        
        if gnorm < 1e-2
            println("✓ Converged!")
            break
        end
        
        # Take step (force = -gradient)
        alpha = min(0.01, 0.1 / (gnorm + 1e-8))  # Adaptive step size
        x_new = x_curr - alpha * pred_grad
        
        # Oracle evaluation
        at_new = [Atom(:Ar, x_new[(i-1)*3+1:i*3]u"Å") for i in 1:N_atoms]
        sys_new = FlexibleSystem(at_new, cell_vectors(sys_curr), periodicity(sys_curr))
        
        E_true, G_true = lennard_jones_oracle(sys_new)
        println("  True:      E = $(round(E_true, digits=4)) | |∇E| = $(round(norm(G_true), digits=4))")
        
        # Check for bad step
        if E_true > 1e6
            println("  ⚠ Energy exploded - reducing step size")
            alpha *= 0.1
            x_new = x_curr - alpha * pred_grad
            
            at_new = [Atom(:Ar, x_new[(i-1)*3+1:i*3]u"Å") for i in 1:N_atoms]
            sys_new = FlexibleSystem(at_new, cell_vectors(sys_curr), periodicity(sys_curr))
            E_true, G_true = lennard_jones_oracle(sys_new)
        end
        
        # Add to training set
        X_train = hcat(X_train, x_new)
        push!(y_vals, E_true)
        append!(y_grads, G_true)
        
        x_curr = x_new
        push!(traj, sys_new)
    end
    
    AtomsIO.save_trajectory("lj13_traj.xyz", traj)
    println("\n✓ Done! Saved trajectory to lj13_traj.xyz")
end

run_lj13()
