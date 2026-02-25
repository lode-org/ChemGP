@testset "SCG optimizer" begin
    # Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    # Minimum at (1,1) with f=0
    function rosenbrock_fg!(f_ref, g, w)
        x, y = w[1], w[2]
        f_ref[] = (1 - x)^2 + 100 * (y - x^2)^2
        g[1] = -2 * (1 - x) + 200 * (y - x^2) * (-2x)
        g[2] = 200 * (y - x^2)
    end

    w0 = [-1.0, 1.0]
    w_best, f_best, converged = scg_optimize(rosenbrock_fg!, w0; max_iter=500, tol_f=1e-12, tol_x=1e-10)

    @test converged
    @test isapprox(w_best[1], 1.0, atol=1e-3)
    @test isapprox(w_best[2], 1.0, atol=1e-3)
    @test f_best < 1e-4

    # Quadratic: f(x) = 0.5 * x' * A * x - b' * x
    # Minimum at A \ b
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    x_opt = A \ b

    function quad_fg!(f_ref, g, w)
        f_ref[] = 0.5 * dot(w, A * w) - dot(b, w)
        g .= A * w - b
    end

    w0_q = [0.0, 0.0]
    w_best_q, f_best_q, conv_q = scg_optimize(quad_fg!, w0_q; max_iter=100, tol_f=1e-12)
    @test conv_q
    @test isapprox(w_best_q, x_opt, atol=1e-6)
end

@testset "NLL gradient (finite difference check)" begin
    # Small 2-atom system with 2 training points
    D = 6
    N = 2
    X = randn(D, N)
    frozen = Float64[]
    feat_map = Int[]

    k_test = MolInvDistSE(1.0, [1.0], frozen)

    # Build y vector (energies + gradients)
    energies = randn(N)
    gradients = randn(D * N)
    y = vcat(energies, gradients)

    noise_e = 1e-4
    noise_g = 1e-2
    jitter = 1e-6

    # Log-space hyperparameters: [log(sigma2), log(inv_ls)]
    w = [log(1.5), log(0.8)]
    w_prior = [log(1.0), log(1.0)]
    prior_var = [1.0, 1.0]

    # Analytical gradient
    nll_val, grad_anal = nll_and_grad(w, X, y, frozen, feat_map, noise_e, noise_g, jitter, w_prior, prior_var)

    # Finite difference gradient
    eps_fd = 1e-5
    grad_fd = zeros(length(w))
    for j in 1:length(w)
        w_plus = copy(w)
        w_minus = copy(w)
        w_plus[j] += eps_fd
        w_minus[j] -= eps_fd
        f_plus, _ = nll_and_grad(w_plus, X, y, frozen, feat_map, noise_e, noise_g, jitter, w_prior, prior_var)
        f_minus, _ = nll_and_grad(w_minus, X, y, frozen, feat_map, noise_e, noise_g, jitter, w_prior, prior_var)
        grad_fd[j] = (f_plus - f_minus) / (2 * eps_fd)
    end

    # Check that analytical and FD gradients agree
    for j in 1:length(w)
        @test isapprox(grad_anal[j], grad_fd[j], rtol=1e-3, atol=1e-6)
    end
end

@testset "NLL gradient with pair types" begin
    # HCN-like: 3 atoms, 3 pair types
    D = 9
    N = 2
    X = zeros(D, N)
    # Point 1: linear HCN
    X[1:3, 1] = [0.0, 0.0, 0.0]   # C
    X[4:6, 1] = [1.2, 0.0, 0.0]   # N
    X[7:9, 1] = [-1.1, 0.0, 0.0]  # H
    # Point 2: slightly bent
    X[1:3, 2] = [0.0, 0.0, 0.0]
    X[4:6, 2] = [1.25, 0.05, 0.0]
    X[7:9, 2] = [-1.05, 0.03, 0.0]

    k_hcn = MolInvDistSE([6, 7, 1], Float64[])  # 3 pair types
    frozen = k_hcn.frozen_coords
    feat_map = k_hcn.feature_params_map

    energies = [-50.0, -49.8]
    gradients = randn(D * N) * 0.1
    y = vcat(energies, gradients)

    noise_e = 1e-4
    noise_g = 1e-2
    jitter = 1e-6

    # w = [log(sigma2), log(theta_CN), log(theta_CH), log(theta_NH)]
    n_ls = length(k_hcn.inv_lengthscales)
    w = vcat([log(2.0)], [log(0.5 + 0.1 * i) for i in 1:n_ls])
    w_prior = copy(w)
    prior_var = fill(1.0, length(w))

    nll_val, grad_anal = nll_and_grad(w, X, y, frozen, feat_map, noise_e, noise_g, jitter, w_prior, prior_var)

    # Finite difference
    eps_fd = 1e-5
    grad_fd = zeros(length(w))
    for j in 1:length(w)
        w_plus = copy(w)
        w_minus = copy(w)
        w_plus[j] += eps_fd
        w_minus[j] -= eps_fd
        f_plus, _ = nll_and_grad(w_plus, X, y, frozen, feat_map, noise_e, noise_g, jitter, w_prior, prior_var)
        f_minus, _ = nll_and_grad(w_minus, X, y, frozen, feat_map, noise_e, noise_g, jitter, w_prior, prior_var)
        grad_fd[j] = (f_plus - f_minus) / (2 * eps_fd)
    end

    for j in 1:length(w)
        @test isapprox(grad_anal[j], grad_fd[j], rtol=5e-3, atol=1e-5)
    end
end

@testset "SCG train_model! path" begin
    # Verify that train_model! with MolInvDistSE + fix_noise uses SCG
    r_vals = [1.0, 1.5, 2.0, 2.5]
    N = length(r_vals)
    X_train = zeros(6, N)
    energies = Float64[]
    grads = Float64[]

    for i in 1:N
        r = r_vals[i]
        X_train[4, i] = r
        De, a, re = 10.0, 1.0, 1.5
        val = 1 - exp(-a * (r - re))
        E = De * val^2
        F = -2 * De * val * (exp(-a * (r - re)) * a)
        push!(energies, E)
        g_vec = [F, 0.0, 0.0, -F, 0.0, 0.0]
        append!(grads, g_vec)
    end

    td = TrainingData(X_train, energies, grads)
    k_init = MolInvDistSE(1.0, [1.0], Float64[])
    k_init = init_mol_invdist_se(td, k_init)

    E_ref = energies[1]
    y = vcat(energies .- E_ref, grads)

    model = GPModel(k_init, X_train, y; noise_var=1e-6, grad_noise_var=1e-4, jitter=1e-6)
    train_model!(model; iterations=100, fix_noise=true, verbose=false)

    # Model should have updated kernel
    @test model.kernel.signal_variance > 0
    @test all(l -> l > 0, model.kernel.inv_lengthscales)

    # Predictions should be reasonable
    X_test = zeros(6, 1)
    X_test[4, 1] = 1.5  # equilibrium
    preds = predict(model, X_test)
    E_pred = preds[1] + E_ref
    De, a, re = 10.0, 1.0, 1.5
    @test isapprox(E_pred, 0.0, atol=0.5)  # near equilibrium energy
end
