# Analysis/coarsening.jl
#
# Coarsening observables for a 1D quench, computed from
#   m_i  = ⟨S^z_i⟩         (one-point profile, length N)
#   C_ij = ⟨S^z_i S^z_j⟩   (two-point matrix, N×N)
# produced by EITHER the ED path (ed_local_profile + ed_correlation_matrix)
# or the MPS path (measure_local_profile + measure_correlation_matrix).
#
# Everything here is source-agnostic plain-array math: no tensor/state objects,
# no model assumptions beyond a 1D chain with a staggered (q = π) order
# parameter. The connected versions (built from `connected_matrix`) subtract the
# parity-odd, seed-dependent disconnected part and are the quantities to trust
# for coarsening.

"""
    connected_matrix(C, m)

Connected two-point matrix  Cᶜ[i,j] = ⟨S^z_i S^z_j⟩ − ⟨S^z_i⟩⟨S^z_j⟩.

Subtracting the disconnected part m_i m_j removes the parity-odd,
seed-dependent contribution, so every observable built from Cᶜ is
seed-independent.
"""
connected_matrix(C::AbstractMatrix, m::AbstractVector) = C .- m * transpose(m)

"""
    staggered_correlator(M; rmax, bulk)

Bulk-averaged staggered real-space correlator

    G(r) = (−1)^r · mean_{i ∈ bulk} M[i, i+r],   r = 0, …, rmax

from a two-point matrix `M`. Pass the connected matrix (`connected_matrix`)
for the seed-independent version. `bulk` is the set of reference sites i to
translation-average over (default: every i with i+r ≤ N; restrict to an
interior window, e.g. 11:90, to suppress open-boundary edge effects).

Returns `(rs, G)` with `rs = 0:rmax`. The (−1)^r factor maps the alternating
AFM correlations onto a positive, decaying envelope whose decay length is the
domain size.
"""
function staggered_correlator(M::AbstractMatrix;
                              rmax::Int = size(M, 1) ÷ 2,
                              bulk::AbstractVector{<:Integer} = 1:size(M, 1))
    N = size(M, 1)
    @assert 0 ≤ rmax ≤ N - 1 "rmax must be in 0:N-1"
    rs = 0:rmax
    G = zeros(Float64, rmax + 1)
    for r in rs
        refs = [i for i in bulk if 1 ≤ i && i + r ≤ N]
        @assert !isempty(refs) "no valid reference sites for r = $r"
        acc = 0.0
        for i in refs
            acc += real(M[i, i + r])
        end
        G[r + 1] = (-1.0)^r * acc / length(refs)
    end
    return rs, G
end

"""
    domain_length(rs, G; method=:integral)

Domain (correlation) length from the staggered correlator `G(r)` with
`rs = 0:rmax`.

- `:integral`  L = Σ_{r=0}^{r*−1} G(r) / G(0), where r* is the first r with
  G(r) ≤ 0 — the area under the normalized correlator up to its first zero.
- `:firstzero` L = linearly-interpolated r at which G(r) first crosses zero.

Both track the same length scale up to an O(1) prefactor. Compute it at each
time and read off growth (coarsening, slope = exponent) vs. saturation.
"""
function domain_length(rs, G::AbstractVector; method::Symbol = :integral)
    @assert G[1] > 0 "G(0) = $(G[1]) must be positive; check the staggering sign"
    zc = findfirst(<=(0.0), G)            # 1-based index of first G(r) ≤ 0
    if method == :integral
        last = zc === nothing ? length(G) : zc - 1
        return sum(G[1:last]) / G[1]
    elseif method == :firstzero
        zc === nothing && return float(rs[end])   # no zero within rmax
        # crossing between r = rs[zc-1] (G>0) and r = rs[zc] (G≤0)
        g_prev, g_here = G[zc - 1], G[zc]
        return float(rs[zc - 1]) + g_prev / (g_prev - g_here)
    else
        error("unknown method $method (use :integral or :firstzero)")
    end
end

"""
    structure_factor(M, qs)

Static structure factor

    S(q) = (1/N) Σ_{j,k} e^{iq(j−k)} M[j,k]   for each q in `qs`,

from a two-point matrix `M`. Pass the connected matrix for the connected
structure factor. The staggered order lives in the q = π peak; its inverse
width (HWHM⁻¹) is an alternative domain-length estimate, and the curves at
different times collapsing under S(q,t) = L(t)·F((q−π)L(t)) is the signature
of genuine coarsening. Returns a real Vector the length of `qs`.
"""
function structure_factor(M::AbstractMatrix, qs::AbstractVector)
    N = size(M, 1)
    S = zeros(Float64, length(qs))
    for (a, q) in enumerate(qs)
        ph = [exp(im * q * j) for j in 1:N]
        S[a] = real(dot(ph, M * ph)) / N
    end
    return S
end

"""
    staggered_magnetization_sq(M)

Squared staggered order parameter

    ⟨m_s²⟩ = (1/N²) Σ_{j,k} (−1)^{j−k} M[j,k] = S(q=π) / N,

from a two-point matrix `M`. Pass the connected matrix for the
seed-independent value, or the full C for the total second moment. Scales as
L(t)/N in a coarsening regime — a cheap global cross-check on the domain
length.
"""
function staggered_magnetization_sq(M::AbstractMatrix)
    N = size(M, 1)
    s = 0.0
    for k in 1:N, j in 1:N
        s += (-1.0)^(j - k) * real(M[j, k])
    end
    return s / N^2
end
