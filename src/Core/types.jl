"""
Abstract supertype for tensor networks (MPS, MPO, MPDO, etc.)
"""
abstract type TensorNetwork{T} end

# A "pure virtual" type: it cannot be instantiated itself,
# but MPS, MPO, and MPDO are subtypes.
# Functions can be written to accept any TensorNetwork,
# so you can dispatch on e.g. f(x::TensorNetwork) for generic behavior.

"""
An MPS is a sequence of rank-3 tensors of element-type `T`.
"""
struct MPS{T} <: TensorNetwork{T}
  tensors::Vector{Array{T,3}}
end

"""
An MPO is a sequence of rank-4 tensors of element-type `T` representing operators.
"""
struct MPO{T} <: TensorNetwork{T}
  tensors::Vector{Array{T,4}}
end

"""
An MPDO is a sequence of rank-4 tensors of element-type `T` representing operators.
"""
struct MPDO{T} <: TensorNetwork{T}
  tensors::Vector{Array{T,4}}
end

"""
Environment holds boundary tensors for efficient contraction.
env.tensors[i] represents the environment between sites i-1 and i.
- env.tensors[1] is the left boundary
- env.tensors[N+1] is the right boundary
"""
struct Environment{T}
    tensors::Vector{Union{Array{T,3}, Nothing}}
end

"""
Options for DMRG sweeps (e.g. krylov dim, ctf, chi_max).
"""
struct DMRGOptions
  chi_max::Int
  cutoff::Float64
  local_dim::Int
end

"""
Options for TDVP sweeps (e.g. krylov dim, ctf, chi_max).
"""
struct TDVPOptions
  dt::Float64
  chi_max::Int
  cutoff::Float64
  local_dim::Int
end

"""
    SweepSchedule

Per-sweep χ ramp and cutoff schedule for DMRG.

    schedule = SweepSchedule(chi_max, n_sweeps; chi_min=chi_max÷8, cutoff_final=1e-10)

Ramps χ linearly from `chi_min` to `chi_max` over the first half of sweeps,
then holds at `chi_max` for the rest.
"""
struct SweepSchedule
    maxdims::Vector{Int}
    cutoffs::Vector{Float64}
    n_sweeps::Int
end

function SweepSchedule(chi_max::Int, n_sweeps::Int;
                       chi_min::Int = max(chi_max ÷ 8, 2),
                       cutoff_final::Float64 = 1e-10)
    ramp_sweeps = n_sweeps ÷ 2
    hold_sweeps = n_sweeps - ramp_sweeps

    # Linear ramp from chi_min to chi_max
    if ramp_sweeps > 0
        ramp = [round(Int, chi_min + (chi_max - chi_min) * (i - 1) / max(ramp_sweeps - 1, 1))
                for i in 1:ramp_sweeps]
    else
        ramp = Int[]
    end
    maxdims = vcat(ramp, fill(chi_max, hold_sweeps))

    # Cutoff: looser early, tighten to final
    cutoffs = vcat(
        [cutoff_final * 10.0^max(0, ramp_sweeps - i) for i in 1:ramp_sweeps],
        fill(cutoff_final, hold_sweeps)
    )

    return SweepSchedule(maxdims, cutoffs, n_sweeps)
end

