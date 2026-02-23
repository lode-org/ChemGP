module ChemGPAtomsBaseExt

using ChemGP
using AtomsBase
using Unitful: @u_str, ustrip
using StaticArrays: SA

function ChemGP.chemgp_coords(sys::AbstractSystem)
    n = length(sys)
    pos = Vector{Float64}(undef, 3n)
    for i in 1:n
        p = position(sys, i)
        pos[3(i-1)+1] = ustrip(u"Å", p[1])
        pos[3(i-1)+2] = ustrip(u"Å", p[2])
        pos[3(i-1)+3] = ustrip(u"Å", p[3])
    end

    atnrs = Int32[atomic_number(sys, i) for i in 1:n]

    pbc = periodicity(sys)
    if any(pbc)
        bb = bounding_box(sys)
        box = Float64[
            ustrip(u"Å", bb[1][1]), ustrip(u"Å", bb[1][2]),
            ustrip(u"Å", bb[1][3]), ustrip(u"Å", bb[2][1]),
            ustrip(u"Å", bb[2][2]), ustrip(u"Å", bb[2][3]),
            ustrip(u"Å", bb[3][1]), ustrip(u"Å", bb[3][2]),
            ustrip(u"Å", bb[3][3]),
        ]
    else
        box = Float64[20, 0, 0, 0, 20, 0, 0, 0, 20]
    end

    return (positions = pos, atomic_numbers = atnrs, box = box)
end

function ChemGP.atomsbase_system(
    positions::AbstractVector{<:Real},
    atomic_numbers::AbstractVector{<:Integer},
    box::AbstractVector{<:Real};
    pbc::Bool = false,
)
    n = length(atomic_numbers)
    atoms = [Atom(Int(atomic_numbers[j]),
                  SA[positions[3(j-1)+1],
                     positions[3(j-1)+2],
                     positions[3(j-1)+3]] * u"Å")
             for j in 1:n]

    if pbc
        cvecs = (SA[box[1], box[2], box[3]] * u"Å",
                 SA[box[4], box[5], box[6]] * u"Å",
                 SA[box[7], box[8], box[9]] * u"Å")
        return periodic_system(atoms, cvecs)
    else
        return isolated_system(atoms)
    end
end

function ChemGP.atomsbase_neb_trajectory(
    result,
    atomic_numbers::AbstractVector{<:Integer},
    box::AbstractVector{<:Real};
    pbc::Bool = false,
)
    path = hasproperty(result, :path) ? result.path : result
    return [ChemGP.atomsbase_system(img, atomic_numbers, box; pbc)
            for img in path.images]
end

end
