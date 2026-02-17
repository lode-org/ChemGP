@testset "Predictive Variance" begin
    # Train a simple GP on Morse data
    r_vals = [1.0, 1.5, 2.0, 2.5]
    N = length(r_vals)
    X_train = zeros(6, N)
    y_vals = Float64[]
    y_grads = Float64[]

    for i in 1:N
        r = r_vals[i]
        X_train[4, i] = r
        De, a, re = 10.0, 1.0, 1.5
        val = 1 - exp(-a * (r - re))
        E = De * val^2
        F = -2 * De * val * exp(-a * (r - re)) * a
        push!(y_vals, E)
        append!(y_grads, [F, 0.0, 0.0, -F, 0.0, 0.0])
    end

    y_mean = mean(y_vals)
    y_std = max(std(y_vals), 1e-10)
    y_full = [(y_vals .- y_mean) ./ y_std; y_grads ./ y_std]

    k = MolInvDistSE(1.0, [1.0], Float64[])
    model = GPModel(k, X_train, y_full; noise_var = 1e-4, grad_noise_var = 1e-4)
    train_model!(model; iterations = 100)

    # Predict at a training point: variance should be small
    X_at_train = zeros(6, 1)
    X_at_train[4, 1] = 1.5
    mu_train, var_train = predict_with_variance(model, X_at_train)

    @test var_train[1] < 0.1  # Energy variance at training point should be small

    # Predict far from training data: variance should be larger
    X_far = zeros(6, 1)
    X_far[4, 1] = 5.0  # Far from any training point
    mu_far, var_far = predict_with_variance(model, X_far)

    @test var_far[1] > var_train[1]  # Far point should have higher variance

    # Variance should be non-negative everywhere
    @test all(var_train .>= 0.0)
    @test all(var_far .>= 0.0)

    # Mean should match predict()
    mu_check = predict(model, X_at_train)
    @test isapprox(mu_train, mu_check, atol = 1e-10)
end
