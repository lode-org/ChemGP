# ==============================================================================
# Scaled Conjugate Gradient (SCG) Optimizer
# ==============================================================================
#
# Port of Moller's SCG from C++ gpr_optim (gpr/ml/SCG.inl).
# Used for MAP NLL hyperparameter optimization with analytical gradients.
#
# Convention: r = gradient, p = search direction (initially -r).
# mu = p'r < 0 for a descent direction.
# alpha = -mu/delta > 0, update w <- w + alpha*p.
# comparison = 2*(f_new - f_old)/(alpha*mu) >= 0 means accept.

"""
    scg_optimize(fg!, w0; max_iter=200, tol_x=1e-6, tol_f=1e-8,
                 sigma0=1e-4, lambda_init=1.0, lambda_max=1e100)

Minimize `f(w)` using Scaled Conjugate Gradient (Moller 1993).

`fg!(f_ref, g_vec, w)` must set `f_ref[]` to the function value and fill
`g_vec` with the gradient at `w`.

Returns `(w_best, f_best, converged)`.
"""
function scg_optimize(
    fg!::Function,
    w0::Vector{Float64};
    max_iter::Int=200,
    tol_x::Float64=1e-6,
    tol_f::Float64=1e-8,
    sigma0::Float64=1e-4,
    lambda_init::Float64=1.0,
    lambda_max::Float64=1e100,
    lambda_min::Float64=1e-15,
    verbose::Bool=false,
)
    n = length(w0)
    w = copy(w0)

    # Allocate work vectors
    r = Vector{Float64}(undef, n)     # gradient
    p = Vector{Float64}(undef, n)     # conjugate direction
    g_new = Vector{Float64}(undef, n)
    g_plus = Vector{Float64}(undef, n)
    w_new = Vector{Float64}(undef, n)

    f_ref = Ref(0.0)
    f_new_ref = Ref(0.0)

    # Initial function + gradient: r = gradient(w)
    fg!(f_ref, r, w)
    f_old = f_ref[]
    if !isfinite(f_old)
        return (w, f_old, false)
    end

    # p = -gradient (steepest descent)
    p .= .-r

    lambda = lambda_init
    success = true
    nsuccess = 0
    f_best = f_old
    w_best = copy(w)
    gamma = 0.0
    kappa = 0.0
    mu = 0.0

    for iter in 1:max_iter
        # --- Compute second-order information ---
        if success
            mu = dot(p, r)
            if mu >= 0.0
                # p is not a descent direction: reset to steepest descent
                p .= .-r
                mu = dot(p, r)
            end

            kappa = dot(p, p)
            if kappa < eps(Float64)
                verbose && @printf("SCG: kappa < eps at iter %d\n", iter)
                return (w_best, f_best, true)
            end

            sigma = sigma0 / sqrt(kappa)

            # Finite-difference Hessian-vector product along p
            @. w_new = w + sigma * p
            fg!(f_new_ref, g_plus, w_new)
            if !isfinite(f_new_ref[])
                lambda *= 4.0
                success = false
                continue
            end

            # gamma = p' * (g_plus - r) / sigma
            # where g_plus = gradient(w + sigma*p), r = gradient(w)
            gamma = dot(p, g_plus .- r) / sigma
        end

        # --- Scale and adjust delta ---
        delta = gamma + lambda * kappa
        if delta <= 0.0
            delta = lambda * kappa
            lambda = lambda - gamma / kappa
        end

        # alpha = -mu/delta > 0 (since mu < 0, delta > 0)
        alpha = -mu / delta

        # --- Trial step ---
        @. w_new = w + alpha * p
        fg!(f_new_ref, g_new, w_new)
        f_new = f_new_ref[]

        # Handle non-finite values
        if !isfinite(f_new)
            lambda *= 4.0
            if lambda >= lambda_max
                verbose && @printf("SCG: lambda overflow at iter %d\n", iter)
                break
            end
            success = false
            continue
        end

        # --- Comparison ratio ---
        # Delta >= 0 means actual reduction matches or exceeds predicted reduction
        # alpha*mu < 0 (alpha > 0, mu < 0), so when f_new < f_old: Delta > 0
        comparison = 2.0 * (f_new - f_old) / (alpha * mu)

        if comparison >= 0.0
            # Accept step
            f_prev = f_old
            w .= w_new
            f_old = f_new

            if f_new < f_best
                f_best = f_new
                w_best .= w_new
            end

            # Convergence checks
            max_step = alpha * sqrt(kappa)
            if max_step < tol_x
                verbose && @printf("SCG converged (tol_x) at iter %d: f=%.6e\n", iter, f_new)
                return (w_best, f_best, true)
            end
            if abs(f_new - f_prev) < tol_f
                verbose && @printf("SCG converged (tol_f) at iter %d: f=%.6e\n", iter, f_new)
                return (w_best, f_best, true)
            end
            if maximum(abs, g_new) < eps(Float64)
                verbose && @printf("SCG converged (grad~0) at iter %d: f=%.6e\n", iter, f_new)
                return (w_best, f_best, true)
            end

            # Update CG direction (Polak-Ribiere)
            # r_old = r (current gradient), g_new = new gradient
            beta = (dot(g_new, g_new) - dot(g_new, r)) / (-mu)
            r .= g_new
            p .= .-r .+ beta .* p

            # Periodic restart to steepest descent
            nsuccess += 1
            if nsuccess >= n
                p .= .-r
                nsuccess = 0
            end

            success = true
        else
            success = false
        end

        # Adjust lambda based on comparison ratio
        if comparison < 0.25
            lambda = min(lambda * 4.0, lambda_max)
        end
        if comparison > 0.75
            lambda = max(lambda * 0.5, lambda_min)
        end

        if lambda >= lambda_max
            verbose && @printf("SCG: lambda_max reached at iter %d\n", iter)
            break
        end
    end

    verbose && @printf("SCG: max_iter reached, f_best=%.6e\n", f_best)
    return (w_best, f_best, false)
end
