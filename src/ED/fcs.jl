# ED/fcs.jl
#
# Full counting statistics of the staggered magnetization, and domain-wall
# count statistics, from a config-basis probability vector p (see ED/cache.jl
# — quench, diagonal-ensemble, thermal, and eigenstate ensembles all reduce to
# such a p, so these functions serve every case).
#
# The windowed staggered magnetization
#     M_s^(w) = Σ_{i∈w} (−1)^i S^z_i
# is diagonal in the computational basis, so its full distribution P(m) is an
# exact histogram of the per-config values weighted by p — no generating
# function needed at ED level. (The MPS version of the same observable is the
# generating-function contraction; this file is its exact reference.)
#
# An AFM domain wall on bond (i,i+1) = parallel neighbors (S^z_i S^z_{i+1} > 0).
# P(0 walls in window) ≡ P(|m| = ℓ/2): the window is a single Néel domain.

"""
    staggered_values(sec; sites=1:sec.N)

Per-config values of M_s = Σ_{i∈sites} (−1)^i S^z_i (length = sector dim), and
the exact value grid they live on: `(msvals, mgrid)` with `mgrid` spanning
−ℓ/2 : 1 : ℓ/2 (spacing 1, ℓ = number of sites in the window).
"""
function staggered_values(sec::SzSector; sites::AbstractVector{<:Integer} = 1:sec.N)
    signs  = [(-1.0)^i for i in sites]
    msvals = sec.sz[:, sites] * signs
    ℓ      = length(sites)
    mgrid  = collect(-ℓ/2:1.0:ℓ/2)
    return msvals, mgrid
end

"""
    staggered_fcs(msvals, mgrid, p)

Exact distribution P(m) of the (windowed) staggered magnetization for the
ensemble with config probabilities `p`: a weighted histogram of the
per-config values `msvals` (from `staggered_values`) on `mgrid`.
"""
function staggered_fcs(msvals::AbstractVector{<:Real}, mgrid::AbstractVector{<:Real},
                       p::AbstractVector{<:Real})
    P = zeros(Float64, length(mgrid))
    for (k, m) in enumerate(msvals)
        P[round(Int, m - mgrid[1]) + 1] += p[k]
    end
    return P
end

"""
    fcs_moments(mgrid, P)

`(mean, var, kurtosis)` of a distribution on `mgrid`. Kurtosis ⟨δm⁴⟩/⟨δm²⟩²
(central): 3 = Gaussian, → 1 = symmetric bimodal (ordered, two Néel branches).
"""
function fcs_moments(mgrid::AbstractVector{<:Real}, P::AbstractVector{<:Real})
    μ  = sum(P .* mgrid)
    δ  = mgrid .- μ
    m2 = sum(P .* δ .^ 2)
    m4 = sum(P .* δ .^ 4)
    return μ, m2, m4 / m2^2
end

"""
    wall_counts(sec; bonds=1:sec.N-1)

Per-config number of AFM domain walls (parallel-neighbor bonds) on `bonds`.
Returns an integer vector of length = sector dim.
"""
function wall_counts(sec::SzSector; bonds::AbstractVector{<:Integer} = 1:sec.N-1)
    dim = length(sec)
    k = zeros(Int, dim)
    for c in 1:dim, b in bonds
        k[c] += sec.sz[c, b] * sec.sz[c, b+1] > 0 ? 1 : 0
    end
    return k
end

"""
    wall_distribution(kcounts, nbonds, p)

Distribution P(n), n = 0:nbonds, of the wall count for the ensemble with
config probabilities `p` (`kcounts` from `wall_counts`).
"""
function wall_distribution(kcounts::AbstractVector{<:Integer}, nbonds::Int,
                           p::AbstractVector{<:Real})
    P = zeros(Float64, nbonds + 1)
    for (c, k) in enumerate(kcounts)
        P[k + 1] += p[c]
    end
    return P
end

"""
    wall_stats(P, ℓ)

`(mean walls, wall density per bond, mean domain size)` of a wall-count
distribution `P` on a window of `ℓ` sites (`length(P) - 1` bonds). Mean domain
size = ⟨ℓ/(n+1)⟩, capped at ℓ for n = 0.
"""
function wall_stats(P::AbstractVector{<:Real}, ℓ::Int)
    nb = length(P) - 1
    n̄ = sum(P .* (0:nb))
    Ld = sum(P[n+1] * ℓ / (n + 1) for n in 0:nb)
    return n̄, n̄ / nb, Ld
end
