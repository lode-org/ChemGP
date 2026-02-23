# ==============================================================================
# Shared Trust Region Distance Utilities
# ==============================================================================
#
# Factored out of OTGPD for reuse across optimizers (OTGPD, GP-NEB).
# Provides:
# - trust_distance_fn: metric factory (:emd, :max_1d_log, :euclidean)
# - trust_min_distance: minimum distance from a point to training data
# - adaptive_trust_threshold: sigmoidal decay with training set size

"""
    trust_distance_fn(metric, atom_types) -> Function

Return a pairwise distance function for the given metric symbol.
Supports `:emd`, `:max_1d_log`, `:euclidean`.

When `metric == :emd`, falls back to Euclidean for non-3D coordinate vectors.
"""
function trust_distance_fn(metric::Symbol, atom_types::Vector{Int}=Int[])
    if metric == :emd
        return (x1, x2) -> begin
            if length(x1) % 3 == 0
                emd_distance(x1, x2; atom_types)
            else
                norm(x1 - x2)
            end
        end
    elseif metric == :max_1d_log
        return max_1d_log_distance
    else  # :euclidean fallback
        return (x1, x2) -> norm(x1 - x2)
    end
end

"""
    trust_min_distance(x, X_train, metric; atom_types) -> Float64

Minimum distance from `x` to any column of `X_train` using the specified metric.
Returns `Inf` when `X_train` has no columns.
"""
function trust_min_distance(
    x::AbstractVector,
    X_train::AbstractMatrix,
    metric::Symbol;
    atom_types::Vector{Int}=Int[],
)
    N = size(X_train, 2)
    N == 0 && return Inf
    dist_fn = trust_distance_fn(metric, atom_types)
    min_d = Inf
    for i in 1:N
        d = dist_fn(x, view(X_train, :, i))
        min_d = min(min_d, d)
    end
    return min_d
end

"""
    adaptive_trust_threshold(trust_radius, n_data, n_atoms; kwargs...) -> Float64

Sigmoidal trust threshold that decays with training set size:

    T(n) = t_min + delta_t / (1 + A * exp(n_eff / n_half))

where `n_eff = n_data / max(n_atoms, 1)`.

Returns fixed `trust_radius` when `use_adaptive == false`.
A floor prevents the threshold from dropping below `floor`.
"""
function adaptive_trust_threshold(
    trust_radius::Float64,
    n_data::Int,
    n_atoms::Int;
    use_adaptive::Bool=false,
    t_min::Float64=0.15,
    delta_t::Float64=0.35,
    n_half::Int=50,
    A::Float64=1.3,
    floor::Float64=0.2,
)
    if !use_adaptive
        return trust_radius
    end
    n_eff = n_data / max(n_atoms, 1)
    t = t_min + delta_t / (1.0 + A * exp(n_eff / n_half))
    return max(t, floor)
end
