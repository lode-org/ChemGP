# ==============================================================================
# HDF5 output for NEB data
# ==============================================================================
#
# Stores NEB path data in a structured HDF5 file that can be read by
# Python (h5py), MATLAB, or any HDF5-capable tool. The layout mirrors
# eOn's per-step .dat + .con output in a single file.

using HDF5

"""
    write_neb_hdf5(result::NEBResult, filename; atomic_numbers=nothing, cell=nothing)

Write final NEB result to HDF5.

Layout:
```
/path/images        (n_images x ndof)    -- positions
/path/energies      (n_images,)          -- energies
/path/gradients     (n_images x ndof)    -- gradients
/path/f_para        (n_images,)          -- force parallel to path
/path/rxn_coord     (n_images,)          -- cumulative distance
/convergence/max_force     (n_iters,)
/convergence/ci_force      (n_iters,)
/convergence/oracle_calls  (n_iters,)
/convergence/max_energy    (n_iters,)
/metadata/converged        Bool
/metadata/oracle_calls     Int
/metadata/max_energy_image Int
/metadata/atomic_numbers   (n_atoms,)    -- if provided
/metadata/cell             (9,)          -- if provided
```
"""
function write_neb_hdf5(
    result::NEBResult,
    filename::AbstractString;
    atomic_numbers::Union{AbstractVector{<:Integer},Nothing} = nothing,
    cell::Union{AbstractVector{<:Real},Nothing} = nothing,
)
    path = result.path
    n = length(path.images)
    ndof = length(path.images[1])

    # Compute derived quantities
    f_para = _compute_f_para(path)
    dists = zeros(n)
    for i in 2:n
        dists[i] = dists[i-1] + norm(path.images[i] .- path.images[i-1])
    end

    h5open(filename, "w") do fid
        # Path data
        g = create_group(fid, "path")
        g["images"] = collect(reduce(hcat, path.images)')  # n_images x ndof
        g["energies"] = path.energies
        g["gradients"] = collect(reduce(hcat, path.gradients)')
        g["f_para"] = f_para
        g["rxn_coord"] = dists

        # Convergence history
        g = create_group(fid, "convergence")
        h = result.history
        g["max_force"] = h["max_force"]
        haskey(h, "ci_force") && (g["ci_force"] = h["ci_force"])
        haskey(h, "oracle_calls") && (g["oracle_calls"] = h["oracle_calls"])
        haskey(h, "max_energy") && (g["max_energy"] = h["max_energy"])
        haskey(h, "image_evaluated") && (g["image_evaluated"] = h["image_evaluated"])

        # Metadata
        g = create_group(fid, "metadata")
        g["converged"] = result.converged
        g["oracle_calls"] = result.oracle_calls
        g["max_energy_image"] = result.max_energy_image
        atomic_numbers !== nothing && (g["atomic_numbers"] = collect(Int32, atomic_numbers))
        cell !== nothing && (g["cell"] = collect(Float64, cell))
    end
    return nothing
end

"""
    make_neb_hdf5_writer(filename; atomic_numbers=nothing, cell=nothing)

Create a step callback that appends per-step NEB path data to an HDF5 file.

Each call creates a group `/steps/NNN` with images, energies, gradients,
f_para, and rxn_coord datasets. This gives the full optimization history
in a single file.

Returns `(path::NEBPath, iteration::Int) -> nothing`.
"""
function make_neb_hdf5_writer(
    filename::AbstractString;
    atomic_numbers::Union{AbstractVector{<:Integer},Nothing} = nothing,
    cell::Union{AbstractVector{<:Real},Nothing} = nothing,
)
    # Create the file with metadata on first call
    initialized = Ref(false)

    return function(path::NEBPath, iteration::Int)
        mode = initialized[] ? "r+" : "w"
        h5open(filename, mode) do fid
            if !initialized[]
                g = create_group(fid, "metadata")
                atomic_numbers !== nothing && (g["atomic_numbers"] = collect(Int32, atomic_numbers))
                cell !== nothing && (g["cell"] = collect(Float64, cell))
                create_group(fid, "steps")
                initialized[] = true
            end

            n = length(path.images)
            f_para = _compute_f_para(path)
            dists = zeros(n)
            for i in 2:n
                dists[i] = dists[i-1] + norm(path.images[i] .- path.images[i-1])
            end

            step_name = @sprintf("%03d", iteration)
            g = create_group(fid["steps"], step_name)
            g["images"] = collect(reduce(hcat, path.images)')
            g["energies"] = path.energies
            g["gradients"] = collect(reduce(hcat, path.gradients)')
            g["f_para"] = f_para
            g["rxn_coord"] = dists
        end
    end
end
