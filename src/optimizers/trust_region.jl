# ==============================================================================
# Trust Region Utilities
# ==============================================================================
#
# In GP-guided optimization, the GP model is only reliable near its training
# data. The trust region ensures that optimization steps on the GP surface
# don't stray too far from where the model has been validated by oracle calls.
#
# Two complementary checks are used:
# 1. Distance to nearest training point (Euclidean or MAX_1D_LOG)
# 2. Interatomic distance ratio check (physical reasonability)

"""
    min_distance_to_data(x, X_train; distance_fn)

Minimum distance from configuration `x` to any training point in `X_train`.

The default distance function is Euclidean (norm). For rotationally invariant
checks, pass `distance_fn=max_1d_log_distance`.
"""
function min_distance_to_data(
    x::AbstractVector,
    X_train::Matrix{Float64};
    distance_fn::Function = (a, b) -> norm(a - b),
)
    N = size(X_train, 2)
    return minimum(distance_fn(x, X_train[:, i]) for i in 1:N)
end

"""
    check_interatomic_ratio(x_new, X_train, ratio_limit)

Check whether `x_new` is "geometrically reasonable" relative to the training
data. Returns `true` if at least one training point has all interatomic
distance ratios within bounds.

This prevents the optimizer from distorting the molecule beyond the GP's
knowledge. The check uses `max_1d_log_distance` (from gpr_optim) to compare
the maximum log-ratio of any interatomic distance pair.

The `ratio_limit` (e.g., 2/3) means no interatomic distance should change by
more than a factor of `ratio_limit` relative to the nearest training point.
"""
function check_interatomic_ratio(
    x_new::AbstractVector,
    X_train::Matrix{Float64},
    ratio_limit::Float64,
)
    threshold = abs(log(ratio_limit))
    for k in 1:size(X_train, 2)
        d = max_1d_log_distance(x_new, X_train[:, k])
        if d < threshold
            return true  # At least one training point is geometrically close
        end
    end
    return false
end
