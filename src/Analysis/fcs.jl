# Analysis/fcs.jl
#
# Full counting statistics on an MPS by the generating-function method of
# Ranabhat & Collura, SciPost Phys. 12, 126 (2022): for a subsystem (window)
# w and single-site Hermitian operators O_i, the observable
#
#     A = Σ_{i ∈ w} O_i
#
# has  G(θ) = ⟨Φ| ∏_{i∈w} e^{iθ O_i} |Φ⟩ / ⟨Φ|Φ⟩  — a product of SINGLE-SITE
# insertions, so each θ is one norm-style transfer contraction (no MPO, no
# sampling) — and the exact distribution follows by discrete Fourier
# inversion,  P(m) = (1/n) Σ_k e^{−iθ_k m} G(θ_k).
#
# The value grid is derived from the operator spectra: A's eigenvalues live
# on the uniform lattice  Σ_i min(O_i)  :  δ  :  Σ_i max(O_i)  whenever the
# per-site eigenvalue gaps are commensurate (always true for spin-1/2 and
# any ±S^z / S^x / staggered variant; asserted at runtime otherwise). With
# n grid points, the n angles θ_k = 2πk/(nδ) make the inversion EXACT.
#
# The operator is supplied BY THE CALLER (built in the script): a single
# d×d matrix applied on every window site, or a vector of matrices (one per
# window site) for site-dependent observables — e.g. the staggered
# magnetization is ops = [(−1)^i · S^z for i in window]. Convenience
# wrappers for that staggered case are kept below (`staggered_fcs`, ...);
# they and the exact ED pair (`staggered_values`/`staggered_fcs` in
# ED/fcs.jl) share the absolute-site (−1)^i convention.
#
# Efficiency: the environments left and right of the window are built once
# (θ-independent) and only the window is re-contracted per angle:
# O(N d χ³ + n ℓ d χ³) per distribution.
#
# Finite temperature: a purification is an ordinary MPS with merged local
# dimension d² (physical ⊗ ancilla). `embed = true` promotes each insertion
# with `embed_physical` (e^{iθO} ⊗ I_d) and the same contraction returns
# P(m) = Tr[ρ δ(A − m)] / Tr ρ — the θ = 0 normalization absorbs Tr ρ.

# ── generic machinery ──────────────────────────────────────────────────────

_as_ops(op::AbstractMatrix, ℓ::Int) = fill(Matrix(op), ℓ)
function _as_ops(ops::AbstractVector{<:AbstractMatrix}, ℓ::Int)
    @assert length(ops) == ℓ "need one operator per window site"
    return Matrix.(ops)
end

# uniform value lattice of Σ_i O_i from the per-site spectra
function _sum_grid(evals::Vector{Vector{Float64}})
    lo, hi = sum(first, evals), sum(last, evals)
    gaps = [e[j+1] - e[j] for e in evals for j in 1:length(e)-1 if e[j+1] - e[j] > 1e-12]
    isempty(gaps) && return [lo], 1.0                 # every O_i ∝ 𝟙
    δ = minimum(gaps)
    for g in vcat(gaps, hi - lo)
        r = g / δ
        abs(r - round(r)) < 1e-9 * max(1, r) ||
            error("FCS: operator eigenvalues are not commensurate — the sum " *
                  "spectrum is not a uniform grid, exact DFT inversion impossible")
    end
    n = round(Int, (hi - lo) / δ) + 1
    return collect(range(lo, hi; length = n)), δ
end

# overlap transfer step, with / without a one-site insertion
function _fcs_transfer(C::AbstractMatrix, A::AbstractArray{<:Number,3})
    @tensoropt Cn[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
    return Cn
end
function _fcs_transfer(C::AbstractMatrix, A::AbstractArray{<:Number,3}, op::AbstractMatrix)
    @tensoropt Cn[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * op[5, 6] * A[4, 6, -2]
    return Cn
end

"""
    fcs_generating_function(mps::MPS, sites, ops, θs; embed=false)

Normalized generating function G(θ) = ⟨Φ|∏_{i∈sites} e^{iθ O_i}|Φ⟩ / ⟨Φ|Φ⟩
for every θ in `θs`, for the windowed sum A = Σ_{i∈sites} O_i. `ops` is a
single d×d Hermitian matrix (same on every window site) or one matrix per
window site. The environments outside the window span are contracted once
and reused across all θ. Works for unnormalized MPS, and for purifications
with `embed = true` (G = Tr ρ e^{iθA} / Tr ρ).
"""
function fcs_generating_function(mps::MPS, sites::AbstractVector{<:Integer},
                                 ops, θs::AbstractVector{<:Real}; embed::Bool = false)
    N = length(mps.tensors)
    @assert issorted(sites) && 1 <= sites[1] && sites[end] <= N "window sites must be sorted within 1:$N"
    opv = _as_ops(ops, length(sites))
    eig = [eigen(Hermitian(op)) for op in opv]
    dphys = size(opv[1], 1)

    # e^{iθ O_i}, optionally promoted to the purification site (⊗ I_d)
    function insertion(k, θ)
        F = eig[k]
        U = F.vectors * Diagonal(cis.(θ .* F.values)) * F.vectors'
        return embed ? embed_physical(U; d = dphys) : U
    end

    lo, hi = sites[1], sites[end]
    pos = Dict(s => k for (k, s) in enumerate(sites))
    T = promote_type(eltype(mps.tensors[1]), ComplexF64)

    L = ones(T, 1, 1)
    for i in 1:lo-1
        L = _fcs_transfer(L, mps.tensors[i])
    end
    R = ones(T, 1, 1)
    for i in N:-1:hi+1
        A = mps.tensors[i]
        @tensoropt Rn[-1, -2] := conj(A)[-1, 5, 3] * A[-2, 5, 4] * R[3, 4]
        R = Rn
    end

    span(θ) = begin
        C = L
        for i in lo:hi
            C = haskey(pos, i) ?
                _fcs_transfer(C, mps.tensors[i], insertion(pos[i], θ)) :
                _fcs_transfer(C, mps.tensors[i])
        end
        sum(C .* R)
    end

    G0 = span(0.0)                      # = ⟨Φ|Φ⟩ (or Tr ρ)
    return [span(θ) / G0 for θ in θs]
end

"""
    fcs_distribution(mps::MPS, sites, ops; embed=false) -> (mgrid, P)

Exact distribution P(m) of the windowed sum A = Σ_{i∈sites} O_i measured on
an MPS (purification: `embed = true`): generating function on the n exact
angles followed by the exact inverse DFT, where n is the number of points of
A's uniform value lattice (derived from the operator spectra).
"""
function fcs_distribution(mps::MPS, sites::AbstractVector{<:Integer}, ops;
                          embed::Bool = false)
    opv = _as_ops(ops, length(sites))
    mgrid, δ = _sum_grid([sort(eigvals(Hermitian(op))) for op in opv])
    n = length(mgrid)
    n == 1 && return mgrid, [1.0]
    θs = [2π * k / (n * δ) for k in 0:n-1]
    G = fcs_generating_function(mps, sites, opv, θs; embed = embed)
    P = [real(sum(cis(-θs[k+1] * m) * G[k+1] for k in 0:n-1)) / n for m in mgrid]
    return mgrid, max.(P, 0.0)          # clip DFT-roundoff negatives ~1e-16
end

"""
    fcs_summary(mps::MPS, sites, ops; embed=false)

One-call FCS measurement: the caller supplies the MPS, the window (subsystem
sites), and the single-site operator(s) defining A = Σ_{i∈window} O_i —
one matrix for a uniform observable, or one matrix per window site
(e.g. `[(-1.0)^i * Sz for i in window]` for the staggered magnetization).
Returns a NamedTuple

    (m, P, extreme_weight, mean, variance, kurtosis)

with `m` the value grid, `P` the exact distribution on it,
`extreme_weight = P(m = m_min) + P(m = m_max)` (for the staggered operator
this is the ordered weight ≡ probability of a wall-free window), and the
central moments (kurtosis: 3 = Gaussian, → 1 = bimodal). This is the
intended entry point for post-processing scripts — the contraction and the
Fourier inversion underneath are implementation details.
"""
function fcs_summary(mps::MPS, sites::AbstractVector{<:Integer}, ops;
                     embed::Bool = false)
    mgrid, P = fcs_distribution(mps, sites, ops; embed = embed)
    μ, var, kurt = fcs_moments(mgrid, P)
    return (m = mgrid, P = P, extreme_weight = P[1] + P[end],
            mean = μ, variance = var, kurtosis = kurt)
end

# ── staggered-magnetization convenience wrappers ───────────────────────────
# M_s = Σ_{i∈w} (−1)^i S^z_i with ABSOLUTE-site parity, matching the ED pair
# `staggered_values` / `staggered_fcs` in ED/fcs.jl.

_staggered_ops(sites; d::Int = 2) =
    [Matrix((-1.0)^i * Diagonal(collect((d-1)/2:-1:-(d-1)/2))) for i in sites]

"""
    staggered_phase_op(θ, site; d=2, embed=false)

Single-site insertion e^{iθ (−1)^site S^z} of the staggered generating
function (diagonal; index 1 = up, the `spin_ops` convention).
"""
function staggered_phase_op(θ::Real, site::Int; d::Int = 2, embed::Bool = false)
    S = (d - 1) / 2
    op = Matrix(Diagonal([cis(θ * (-1.0)^site * m) for m in S:-1:-S]))
    return embed ? embed_physical(op; d = d) : op
end

"""
    staggered_generating_function(mps::MPS, sites, θs; embed=false)

G(θ) for the windowed staggered magnetization (wrapper around
[`fcs_generating_function`](@ref) with O_i = (−1)^i S^z).
"""
staggered_generating_function(mps::MPS, sites::AbstractVector{<:Integer},
                              θs::AbstractVector{<:Real}; embed::Bool = false) =
    fcs_generating_function(mps, sites, _staggered_ops(sites), θs; embed = embed)

"""
    staggered_fcs(mps::MPS, sites; embed=false) -> (mgrid, P)

Distribution of the windowed staggered magnetization on an MPS
(`mgrid = −ℓ/2 : 1 : ℓ/2`); directly comparable to the ED
`staggered_fcs(msvals, mgrid, p)`.
"""
staggered_fcs(mps::MPS, sites::AbstractVector{<:Integer}; embed::Bool = false) =
    fcs_distribution(mps, sites, _staggered_ops(sites); embed = embed)

"""
    fcs_summary(mps::MPS, sites; embed=false)

Staggered-magnetization default of the general [`fcs_summary`](@ref); the
returned NamedTuple additionally carries `ordered_weight` (= the
`extreme_weight` field, which for the staggered operator is the probability
of a wall-free, fully Néel-ordered window).
"""
function fcs_summary(mps::MPS, sites::AbstractVector{<:Integer}; embed::Bool = false)
    r = fcs_summary(mps, sites, _staggered_ops(sites); embed = embed)
    return merge(r, (ordered_weight = r.extreme_weight,))
end
