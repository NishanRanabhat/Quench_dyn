function spin_ops(d::Integer)
    @assert d ≥ 1 "d must be ≥ 1"
    # total spin S and its m‐values
    S = (d - 1)/2
    m_vals = collect(S:-1:-S)   # [S, S-1, …, -S]

    # Sz is just diagonal of m_vals
    Sz = Diagonal(m_vals)

    # Build S+ and S– by placing coef on the super/sub‐diagonal
    Sp = zeros(Float64, d, d)
    @inbounds for i in 1:d-1
        m_lower = m_vals[i+1]   # THIS is the m of the state being raised
        coef = sqrt((S - m_lower)*(S + m_lower + 1))
        Sp[i, i+1] = coef
    end
    Sm = Sp'  # adjoint

    # Now the cartesian components
    Sx = (Sp + Sm)/2
    Sy = (Sp - Sm) / (2im)

    return Dict(:X => Sx,
                :Y => Sy, 
                :Z => Sz,
                :Sp => Sp,
                :Sm => Sm, 
                :I => Matrix{Float64}(I, d, d))
end

# ──────────────────────────────────────────────────────────────────────────────
# Abstract supertype
#───────────────────────────────────────────────────────────────────────────────

"""
    AbstractSite{T}

A common supertype for all single‐site objects (spins, supersites, etc.).
"""
abstract type AbstractSite{T} end


struct SpinSite{T} <: AbstractSite{T}
    dim::Int
    ops::Dict             # :X,:Y,:Z,…
    spectra::Dict  # precomputed eigvals/vecs
end
  
function SpinSite(S::Real; T=ComplexF64)
    d   = Int(2S + 1)
    # Always store operators as ComplexF64 (σʸ is inherently complex)
    ops = Dict{Symbol,Matrix{ComplexF64}}()
    spectra = Dict{Symbol,Tuple{Vector{Float64},Matrix{ComplexF64}}}()
    raw = spin_ops(d)
    for ax in (:X,:Y,:Z)
        mat = ComplexF64.(raw[ax])
        E = eigen(Hermitian(mat))
        idx = sortperm(E.values, rev=true)
        ops[ax] = mat
        spectra[ax] = (real.(E.values[idx]), ComplexF64.(E.vectors[:,idx]))
    end
    return SpinSite{T}(d,ops,spectra)
end

"""
    state_tensor(site::SpinSite, label::Pair{Symbol,Int})

Return the (1,d,1) tensor for the `k`th eigenvector (ascending) of `axis`.
"""

function _state_tensor(site::SpinSite{T}, label::Tuple{Symbol,Int}) where T
    ax, k = label
    vals, vecs = site.spectra[ax]
    @assert 1 ≤ k ≤ length(vals)
    return reshape(vecs[:,k], 1, site.dim, 1)
end

# ──────────────────────────────────────────────────────────────────────────────
# SuperSite: composite site bundling n_sub spin-1/2 sites into d = 2^n_sub
# ──────────────────────────────────────────────────────────────────────────────

"""
    SuperSite{T}

A composite site formed by bundling `n_sub` spin-1/2 sites into a single
supersite with local Hilbert space dimension `d = 2^n_sub`.

Sublattice operators are stored as d×d matrices built via Kronecker products.
Each sublattice `k` has operators :Sz, :Sp, :Sm (8×8 for n_sub=3).

`basis_labels` maps basis index → tuple of per-sublattice spin labels,
e.g. index 1 → (:up, :up, :up) for n_sub=3.
"""
struct SuperSite{T} <: AbstractSite{T}
    dim::Int
    n_sub::Int
    sub_ops::Vector{Dict{Symbol, Matrix{T}}}   # sub_ops[k][:Sz], [:Sp], [:Sm]
    identity::Matrix{T}                         # d×d identity
    basis_labels::Vector{NTuple}                # human-readable basis labels
end

function SuperSite(n_sub::Int; T=Float64)
    @assert n_sub ≥ 2 "Need at least 2 sublattice sites"
    d = 2^n_sub

    # Fundamental spin-1/2 operators
    id2 = Matrix{T}(I, 2, 2)
    sz  = T[0.5 0; 0 -0.5]
    sp  = T[0 1; 0 0]
    sm  = T[0 0; 1 0]

    # Build sublattice operators via Kronecker products
    sub_ops = Vector{Dict{Symbol, Matrix{T}}}(undef, n_sub)
    for k in 1:n_sub
        # Operator on sublattice k = id ⊗ ... ⊗ op ⊗ ... ⊗ id
        mats_z  = [i == k ? sz : id2 for i in 1:n_sub]
        mats_p  = [i == k ? sp : id2 for i in 1:n_sub]
        mats_m  = [i == k ? sm : id2 for i in 1:n_sub]
        sub_ops[k] = Dict{Symbol, Matrix{T}}(
            :Sz => foldl(kron, mats_z),
            :Sp => foldl(kron, mats_p),
            :Sm => foldl(kron, mats_m),
        )
    end

    identity = Matrix{T}(I, d, d)

    # Build basis labels: enumerate all 2^n_sub states
    # Index ordering matches kron convention: first sublattice is most significant
    basis_labels = Vector{NTuple{n_sub, Symbol}}(undef, d)
    for idx in 0:d-1
        bits = ntuple(n_sub) do k
            # k=1 is most significant bit (leftmost in kron)
            bit = (idx >> (n_sub - k)) & 1
            bit == 0 ? :up : :dn
        end
        basis_labels[idx+1] = bits
    end

    return SuperSite{T}(d, n_sub, sub_ops, identity, basis_labels)
end

"""
    _state_tensor(site::SuperSite{T}, label::NTuple{N,Symbol})

Return the (1, d, 1) product-state tensor for a supersite configuration.
`label` is a tuple of :up/:dn for each sublattice, e.g. (:up, :dn, :up).
"""
function _state_tensor(site::SuperSite{T}, label::NTuple{N,Symbol}) where {T, N}
    @assert N == site.n_sub "Label must have $(site.n_sub) entries, got $N"
    idx = findfirst(==(label), site.basis_labels)
    @assert !isnothing(idx) "Invalid label $label"
    v = zeros(T, site.dim)
    v[idx] = one(T)
    return reshape(v, 1, site.dim, 1)
end

"""
    _state_tensor(site::SuperSite{T}, idx::Int)

Return the (1, d, 1) tensor for basis state `idx` (1-based).
"""
function _state_tensor(site::SuperSite{T}, idx::Int) where T
    @assert 1 ≤ idx ≤ site.dim "Index $idx out of range 1:$(site.dim)"
    v = zeros(T, site.dim)
    v[idx] = one(T)
    return reshape(v, 1, site.dim, 1)
end