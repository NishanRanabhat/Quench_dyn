# TensorOps/decomposition.jl

"""
    _svd_truncate(A::Matrix, chi_max::Int, cutoff::Float64)

Perform SVD with truncation based on maximum bond dimension and singular-
value cutoff. Norm-preserving: the returned `S` are the actual (un-rescaled)
singular values of `A` restricted to the kept bond, so the reconstructed
`U * Diagonal(S) * V` has the same Frobenius norm as the kept subspace of
`A`. The `cutoff` is applied to the normalized singular-value spectrum
`F.S / norm(F.S)`, i.e. it is a *relative* threshold.

Works for both MPS and MPDO since they're reshaped to matrices before
calling this.
"""
function _svd_truncate(A::Matrix{T}, chi_max::Int, cutoff::Float64) where T
    F = svd(A, alg=LinearAlgebra.QRIteration())

    nrm = norm(F.S)
    # Apply cutoff on the relative spectrum so `cutoff` is meaningful
    # regardless of the overall scale of the state.
    S_rel = nrm > 0 ? F.S ./ nrm : F.S
    chi_cut = findfirst(x -> x < cutoff, S_rel)
    chi_cut = isnothing(chi_cut) ? length(F.S) : chi_cut - 1
    chi = min(chi_cut, chi_max, length(F.S))

    U = F.U[:, 1:chi]
    S = F.S[1:chi]              # keep the true singular values
    V = F.Vt[1:chi, :]

    return U, S, V
end


# ===== Utility functions =====

"""
    entropy(S::Vector)

Calculate entanglement entropy from singular values.
"""
function entropy(S::Vector{T}) where T
    S_normalized = S ./ norm(S)
    S_squared = S_normalized .^ 2
    
    # Remove zero values to avoid log(0)
    S_squared = S_squared[S_squared .> eps(T)]
    
    return -sum(s2 * log(s2) for s2 in S_squared)
end

"""
    truncation_error(S::Vector, chi::Int)

Calculate truncation error from discarding singular values beyond chi.
"""
function truncation_error(S::Vector{T}, chi::Int) where T
    if chi >= length(S)
        return zero(T)
    end
    
    S_normalized = S ./ norm(S)
    return sum(S_normalized[chi+1:end].^2)
end