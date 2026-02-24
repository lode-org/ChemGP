module ChemGPMLflowExt

using ChemGP
using HTTP
using JSON3

# ==============================================================================
# MLflow REST API client
# ==============================================================================

"""
    MLflowTracker <: AbstractTracker

Lightweight MLflow REST API client for logging optimization metrics.

All HTTP calls are wrapped in try/catch so tracking failures never crash
the optimizer. Warnings are printed on connection failure.
"""
struct MLflowTracker <: ChemGP.AbstractTracker
    tracking_uri::String
    experiment_id::String
    run_id::String
    step_counter::Ref{Int}
end

function _mlflow_post(tracker::MLflowTracker, endpoint::String, body::Dict)
    url = tracker.tracking_uri * endpoint
    headers = ["Content-Type" => "application/json"]
    try
        resp = HTTP.post(url, headers, JSON3.write(body); connect_timeout=5, readtimeout=10)
        return JSON3.read(resp.body)
    catch e
        @warn "MLflow tracking failed" endpoint exception = e
        return nothing
    end
end

function _mlflow_get(tracker::MLflowTracker, endpoint::String; query::Dict=Dict())
    url = tracker.tracking_uri * endpoint
    try
        resp = HTTP.get(url; query, connect_timeout=5, readtimeout=10)
        return JSON3.read(resp.body)
    catch e
        @warn "MLflow tracking failed" endpoint exception = e
        return nothing
    end
end

"""
    MLflowTracker(; uri="http://localhost:5000", experiment="ChemGP")

Create a tracker by finding or creating the named experiment, then starting a new run.
"""
function MLflowTracker(; uri::String="http://localhost:5000", experiment::String="ChemGP")
    tracker_uri = rstrip(uri, '/')

    # Get or create experiment
    exp_id = _get_or_create_experiment(tracker_uri, experiment)
    if exp_id === nothing
        error("Failed to create/find MLflow experiment '$experiment' at $tracker_uri")
    end

    # Create run
    run_id = _create_run(tracker_uri, exp_id)
    if run_id === nothing
        error("Failed to create MLflow run in experiment '$experiment'")
    end

    return MLflowTracker(tracker_uri, exp_id, run_id, Ref(0))
end

function _get_or_create_experiment(uri::String, name::String)
    # Try to get existing experiment
    try
        resp = HTTP.get(
            uri * "/api/2.0/mlflow/experiments/get-by-name";
            query=Dict("experiment_name" => name),
            connect_timeout=5,
            readtimeout=10,
        )
        data = JSON3.read(resp.body)
        return string(data.experiment.experiment_id)
    catch
        # Experiment doesn't exist; create it
        try
            headers = ["Content-Type" => "application/json"]
            resp = HTTP.post(
                uri * "/api/2.0/mlflow/experiments/create",
                headers,
                JSON3.write(Dict("name" => name));
                connect_timeout=5,
                readtimeout=10,
            )
            data = JSON3.read(resp.body)
            return string(data.experiment_id)
        catch e
            @warn "Failed to create MLflow experiment" exception = e
            return nothing
        end
    end
end

function _create_run(uri::String, experiment_id::String)
    try
        headers = ["Content-Type" => "application/json"]
        body = Dict(
            "experiment_id" => experiment_id,
            "start_time" => round(Int, time() * 1000),
        )
        resp = HTTP.post(
            uri * "/api/2.0/mlflow/runs/create",
            headers,
            JSON3.write(body);
            connect_timeout=5,
            readtimeout=10,
        )
        data = JSON3.read(resp.body)
        return string(data.run.info.run_id)
    catch e
        @warn "Failed to create MLflow run" exception = e
        return nothing
    end
end

# ==============================================================================
# Logging functions
# ==============================================================================

"""
    log_metric!(tracker, key, value, step)

Log a single metric value at the given step.
"""
function ChemGP.log_metric!(tracker::MLflowTracker, key::String, value::Real, step::Int)
    _mlflow_post(tracker, "/api/2.0/mlflow/runs/log-metric", Dict(
        "run_id" => tracker.run_id,
        "key" => key,
        "value" => Float64(value),
        "step" => step,
        "timestamp" => round(Int, time() * 1000),
    ))
    return nothing
end

"""
    log_params!(tracker, params::Dict)

Log a dictionary of parameters (string key-value pairs).
"""
function ChemGP.log_params!(tracker::MLflowTracker, params::Dict)
    param_list = [Dict("key" => string(k), "value" => string(v)) for (k, v) in params]
    _mlflow_post(tracker, "/api/2.0/mlflow/runs/log-batch", Dict(
        "run_id" => tracker.run_id,
        "params" => param_list,
    ))
    return nothing
end

"""
    log_batch!(tracker, metrics::Vector{<:NamedTuple})

Batch-log multiple metrics. Each element should have fields `key`, `value`, `step`.
"""
function ChemGP.log_batch!(
    tracker::MLflowTracker, metrics::Vector{<:NamedTuple}
)
    ts = round(Int, time() * 1000)
    metric_list = [
        Dict(
            "key" => string(m.key),
            "value" => Float64(m.value),
            "step" => Int(m.step),
            "timestamp" => ts,
        ) for m in metrics
    ]
    _mlflow_post(tracker, "/api/2.0/mlflow/runs/log-batch", Dict(
        "run_id" => tracker.run_id,
        "metrics" => metric_list,
    ))
    return nothing
end

"""
    finish_run!(tracker; status="FINISHED")

Set the run status. Use "FINISHED", "FAILED", or "KILLED".
"""
function ChemGP.finish_run!(tracker::MLflowTracker; status::String="FINISHED")
    _mlflow_post(tracker, "/api/2.0/mlflow/runs/update", Dict(
        "run_id" => tracker.run_id,
        "status" => status,
        "end_time" => round(Int, time() * 1000),
    ))
    return nothing
end

# ==============================================================================
# Callback constructors for each optimizer type
# ==============================================================================

"""
    mlflow_callback(tracker, optimizer_type::Symbol)

Return an `on_step` callback appropriate for the given optimizer type.

Supported types: `:minimize`, `:dimer`, `:neb`, `:neb_aie`, `:neb_oie`.

The returned function logs all relevant metrics to MLflow at each step.
"""
function ChemGP.mlflow_callback(tracker::MLflowTracker, optimizer_type::Symbol)
    if optimizer_type == :minimize
        return _minimize_callback(tracker)
    elseif optimizer_type == :dimer
        return _dimer_callback(tracker)
    elseif optimizer_type in (:neb, :neb_aie, :neb_oie)
        return _neb_callback(tracker)
    else
        error("Unknown optimizer type for MLflow callback: $optimizer_type")
    end
end

function _minimize_callback(tracker::MLflowTracker)
    return function(info::Dict)
        step = info["step"]
        ts = round(Int, time() * 1000)
        metrics = [
            Dict("key" => "energy", "value" => Float64(info["energy"]),
                 "step" => step, "timestamp" => ts),
            Dict("key" => "max_force", "value" => Float64(info["max_force"]),
                 "step" => step, "timestamp" => ts),
            Dict("key" => "oracle_calls", "value" => Float64(info["oracle_calls"]),
                 "step" => step, "timestamp" => ts),
        ]
        _mlflow_post(tracker, "/api/2.0/mlflow/runs/log-batch", Dict(
            "run_id" => tracker.run_id,
            "metrics" => metrics,
        ))
        return nothing
    end
end

function _dimer_callback(tracker::MLflowTracker)
    return function(info::Dict)
        step = info["step"]
        ts = round(Int, time() * 1000)
        metrics = [
            Dict("key" => "energy", "value" => Float64(info["energy"]),
                 "step" => step, "timestamp" => ts),
            Dict("key" => "force_trans", "value" => Float64(info["force_trans"]),
                 "step" => step, "timestamp" => ts),
            Dict("key" => "curvature", "value" => Float64(info["curvature"]),
                 "step" => step, "timestamp" => ts),
            Dict("key" => "oracle_calls", "value" => Float64(info["oracle_calls"]),
                 "step" => step, "timestamp" => ts),
        ]
        _mlflow_post(tracker, "/api/2.0/mlflow/runs/log-batch", Dict(
            "run_id" => tracker.run_id,
            "metrics" => metrics,
        ))
        return nothing
    end
end

function _neb_callback(tracker::MLflowTracker)
    return function(path, iter)
        tracker.step_counter[] += 1
        step = tracker.step_counter[]
        ts = round(Int, time() * 1000)

        max_e = maximum(path.energies)
        metrics = [
            Dict("key" => "max_energy", "value" => Float64(max_e),
                 "step" => step, "timestamp" => ts),
            Dict("key" => "iteration", "value" => Float64(iter),
                 "step" => step, "timestamp" => ts),
        ]
        _mlflow_post(tracker, "/api/2.0/mlflow/runs/log-batch", Dict(
            "run_id" => tracker.run_id,
            "metrics" => metrics,
        ))
        return nothing
    end
end

# Re-export MLflowTracker
export MLflowTracker

end # module
