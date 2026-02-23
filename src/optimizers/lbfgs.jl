# ==============================================================================
# L-BFGS 2-Loop Recursion
# ==============================================================================
#
# Standard L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) optimizer
# using the two-loop recursion algorithm. This is a critical building block for
# advanced dimer rotation/translation (Phase 2) and NEB image relaxation (Phase 3).
#
# The algorithm maintains a circular buffer of the m most recent {s, y} pairs
# (step differences and gradient differences) and uses them to approximate the
# inverse Hessian-gradient product without ever forming the Hessian.
#
# Reference:
#   Nocedal, J. (1980). Updating quasi-Newton matrices with limited storage.
#   Mathematics of Computation, 35(151), 773-782.
#
# MATLAB reference: gpr_dimer_matlab/GPR/dimer/rot_iter_lbfgs.m lines 45-77

"""
    LBFGSHistory

Circular buffer storing the `m` most recent step/gradient-difference pairs for
the L-BFGS two-loop recursion.

# Fields
- `m::Int`: Maximum memory depth (number of stored pairs)
- `s::Vector{Vector{Float64}}`: Step differences `s_k = x_{k+1} - x_k`
- `y::Vector{Vector{Float64}}`: Gradient differences `y_k = g_{k+1} - g_k`
- `count::Int`: Total number of pairs pushed (used to track buffer fill)

See also: [`push_pair!`](@ref), [`compute_direction`](@ref), [`reset!`](@ref)
"""
mutable struct LBFGSHistory
    m::Int
    s::Vector{Vector{Float64}}
    y::Vector{Vector{Float64}}
    count::Int
end

"""
    LBFGSHistory(m::Int)

Create an empty L-BFGS history with memory depth `m`.

Typical values: `m = 5` for dimer rotation, `m = 10-20` for translation/minimization.
"""
function LBFGSHistory(m::Int)
    return LBFGSHistory(m, Vector{Float64}[], Vector{Float64}[], 0)
end

"""
    push_pair!(hist::LBFGSHistory, s::Vector{Float64}, y::Vector{Float64})

Append a new `(s, y)` pair to the history buffer. If the buffer is full
(length ≥ m), the oldest pair is removed first (circular buffer semantics).

Arguments:
- `s`: Step difference `x_{k+1} - x_k`
- `y`: Gradient difference `g_{k+1} - g_k`

The pair is silently skipped if the curvature condition `y'*s > 0` is not
satisfied, since such pairs would produce a non-positive-definite Hessian
approximation.
"""
function push_pair!(hist::LBFGSHistory, s::Vector{Float64}, y::Vector{Float64})
    # Skip pairs that violate the curvature condition
    ys = dot(y, s)
    if ys <= 1e-18
        return nothing
    end

    # Circular buffer: remove oldest if full
    if length(hist.s) >= hist.m
        popfirst!(hist.s)
        popfirst!(hist.y)
    end

    push!(hist.s, copy(s))
    push!(hist.y, copy(y))
    hist.count += 1
end

"""
    compute_direction(hist::LBFGSHistory, gradient::Vector{Float64}) -> Vector{Float64}

Compute the L-BFGS search direction `d = -H_k * g` using the standard two-loop
recursion (Nocedal & Wright, Algorithm 7.4).

Returns the search direction (descent direction, pointing downhill). If the
history is empty, falls back to steepest descent: `d = -gradient`.

The initial Hessian approximation is `H_0 = γI` where `γ = (s'y)/(y'y)` from
the most recent pair (Rayleigh quotient scaling).
"""
function compute_direction(hist::LBFGSHistory, gradient::Vector{Float64})
    m = length(hist.s)

    # Fallback to steepest descent if no history
    if m == 0
        return -gradient
    end

    q = copy(gradient)
    alpha = zeros(m)
    rho = zeros(m)

    # Precompute rho
    for i in 1:m
        ys = dot(hist.y[i], hist.s[i])
        rho[i] = ys > 1e-18 ? 1.0 / ys : 0.0
    end

    # Loop 1: Backward pass (newest to oldest)
    for i in m:-1:1
        alpha[i] = rho[i] * dot(hist.s[i], q)
        q .-= alpha[i] .* hist.y[i]
    end

    # Initial Hessian scaling: γ = (s'y)/(y'y) from most recent pair
    s_last = hist.s[m]
    y_last = hist.y[m]
    yy = dot(y_last, y_last)
    gamma = yy > 1e-18 ? dot(s_last, y_last) / yy : 1.0

    r = gamma .* q

    # Loop 2: Forward pass (oldest to newest)
    for i in 1:m
        beta = rho[i] * dot(hist.y[i], r)
        r .+= (alpha[i] - beta) .* hist.s[i]
    end

    return -r
end

"""
    reset!(hist::LBFGSHistory)

Clear the history buffer, discarding all stored pairs. The memory depth `m`
is preserved. Use this when the optimization problem changes (e.g., after
a trust region violation or when switching between rotation and translation).
"""
function reset!(hist::LBFGSHistory)
    empty!(hist.s)
    empty!(hist.y)
    hist.count = 0
end
