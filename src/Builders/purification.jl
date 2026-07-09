# Builders/purification.jl
#
# Ancilla-purification (LPTN, merged-index) primitives for finite temperature
# via imaginary-time TDVP — "Approach A".
#
# The purified state is an ORDINARY MPS with local dimension d² = physical ⊗
# ancilla, merged into one leg with index  k = (s-1)*d + a  (physical s outer,
# ancilla a inner). Tracing out the ancilla of |ψ⟩⟨ψ| gives the density matrix
#   ρ = Tr_anc |ψ⟩⟨ψ| ,
# and cooling |ψ⟩ by imaginary time τ = β/2 under H ⊗ I_anc realizes
#   ρ(β) ∝ e^{-βH}.
#
# This lets the entire (validated) MPS TDVP / environment / SVD / measurement
# machinery be reused unchanged: the ancilla identity lives in the MPO (H⊗I,
# built once by `purify_mpo`) and in the observables (O⊗I, `embed_physical`),
# NOT in the hot-loop contraction. Convention: with k=(s-1)*d+a, tensoring an
# identity onto the ancilla is exactly `kron(op_phys, I_d)`.

"""
    maximally_mixed_purification(N; d=2, T=Float64)

β = 0 (infinite-temperature) purified state: a product MPS with local dimension
d², bond dimension 1, each site the maximally entangled physical–ancilla pair
`(1/√d) Σ_s |s⟩_phys |s⟩_anc`. Tracing the ancilla gives ρ = (I/d)^{⊗N}
(maximally mixed), and ⟨ψ|ψ⟩ = 1.
"""
function maximally_mixed_purification(N::Int; d::Int = 2, T::Type = Float64)
    A = zeros(T, 1, d * d, 1)
    for s in 1:d
        A[1, (s - 1) * d + s, 1] = one(T) / sqrt(T(d))   # a = s
    end
    return MPS{T}([copy(A) for _ in 1:N])
end

"""
    purify_mpo(mpo::MPO; d=2)

Embed a physical Hamiltonian MPO (physical legs d×d) into the purified
local space (d²) as H ⊗ I_anc: each physical block becomes
`kron(W[l,r,:,:], I_d)`. The MPO bond dimension is unchanged — the ancilla
identity is a purely on-site factor. Matches the merged index k=(s-1)*d+a.
"""
function purify_mpo(mpo::MPO{T}; d::Int = 2) where {T}
    Id = Matrix{T}(I, d, d)
    new_tensors = Vector{Array{T,4}}(undef, length(mpo.tensors))
    for (n, W) in enumerate(mpo.tensors)
        Dl, Dr, do_, di_ = size(W)
        @assert do_ == d && di_ == d "MPO physical dims ($do_,$di_) ≠ d=$d"
        W2 = Array{T,4}(undef, Dl, Dr, d * d, d * d)
        for l in 1:Dl, r in 1:Dr
            @views W2[l, r, :, :] = kron(W[l, r, :, :], Id)
        end
        new_tensors[n] = W2
    end
    return MPO{T}(new_tensors)
end

"""
    embed_physical(op; d=2)

Embed a single-site physical operator into the purified local space as
`op ⊗ I_anc = kron(op, I_d)` (d² × d²). Feed this to the ordinary MPS
measurement routines to get ⟨O_phys⟩ on a purified MPS (then divide by
⟨ψ|ψ⟩ = Tr ρ for the normalized thermal expectation).
"""
function embed_physical(op::AbstractMatrix{T}; d::Int = 2) where {T}
    return kron(Matrix{T}(op), Matrix{T}(I, d, d))
end
