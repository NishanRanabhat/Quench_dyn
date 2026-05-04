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

