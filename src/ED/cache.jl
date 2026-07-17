# ED/cache.jl
#
# Loaders and ensemble constructors for the on-disk ED spectrum cache written
# by new_scripts/ed_diag_save.jl and new_scripts/ed_gsi_seeded_save.jl (data/*.jls).
#
# The unifying object is a PROBABILITY VECTOR p over the sector's
# computational-basis configs: because S^z is diagonal in that basis, every
# ensemble we compare — quench ψ(t), diagonal ensemble, thermal(β), any pure
# eigenstate — enters every S^z-built observable (correlators, structure
# factor, FCS, domain-wall counts) only through its p. The comparison
# "quench vs diagonal vs thermal vs GS_f" is therefore four p-vectors fed to
# the same observable functions (thermal_sz_correlations, staggered_fcs, ...).
#
# Cached eigenvectors are stored REAL (H is real symmetric); loaders return
# them as-is. Convert with `EDEigensystem(values, ComplexF64.(vectors))` only
# if a routine demands the complex container.

const ED_CACHE_DIR = normpath(joinpath(@__DIR__, "..", "..", "data"))

"""
    load_spectrum(N, Δ; datadir=ED_CACHE_DIR)

Load a cached full eigensystem of the S^z=0-sector XXZ Hamiltonian (J=1, h=0,
open boundaries) from `ed_spectrum_N{N}_delta{Δ}.jls`. Returns a NamedTuple

    (sec, values, vectors, meta)

with `sec::SzSector` rebuilt deterministically (basis order = coefficient
order), `values::Vector{Float64}` ascending, `vectors::Matrix{Float64}`
(eigenvectors as columns), and the file's metadata Dict.
"""
function load_spectrum(N::Int, Δ::Real; datadir::AbstractString = ED_CACHE_DIR)
    file = joinpath(datadir, "ed_spectrum_N$(N)_delta$(Δ).jls")
    isfile(file) || error("no cached spectrum $file — run new_scripts/ed_diag_save.jl")
    d = deserialize(file)
    sec = sz_sector(d[:N], d[:n_up])
    return (sec = sec, values = d[:values]::Vector{Float64},
            vectors = d[:vectors]::Matrix{Float64}, meta = d)
end

"""
    load_initial_state(N; h_s=0.0, Δ=0.8, datadir=ED_CACHE_DIR)

Load a cached quench initial state (ground state of H_i) from the cache:
`h_s = 0` → the clean GS (`ed_gsi_N{N}_delta{Δ}.jls`); `h_s > 0` → the GS
seeded with the staggered longitudinal field h_s·(−1)^i
(`ed_gsi_N{N}_delta{Δ}_stag{h_s}.jls`). Returns `(psi, energy, meta)`.
"""
function load_initial_state(N::Int; h_s::Real = 0.0, Δ::Real = 0.8,
                            datadir::AbstractString = ED_CACHE_DIR)
    file = h_s == 0 ? joinpath(datadir, "ed_gsi_N$(N)_delta$(Δ).jls") :
                      joinpath(datadir, "ed_gsi_N$(N)_delta$(Δ)_stag$(h_s).jls")
    isfile(file) || error("no cached initial state $file — run new_scripts/ed_diag_save.jl / ed_gsi_seeded_save.jl")
    d = deserialize(file)
    return (psi = d[:psi]::Vector{Float64}, energy = d[:energy]::Float64, meta = d)
end

"""
    thermal_weights(values, β)

Normalized Gibbs weights w_n ∝ e^{−βE_n} over the eigenvalues (log-sum-exp
shifted, stable for any β). `diagonal_probabilities(V, w)` turns them into the
config-basis thermal p (same quantity as `thermal_diagonal`, but for the
cache's real eigenvector matrix).
"""
function thermal_weights(values::AbstractVector{<:Real}, β::Real)
    w = exp.(-β .* values .- maximum(-β .* values))
    return w ./ sum(w)
end

"""
    diagonal_probabilities(V, w; block=2048)

Config-basis probability vector of a mixture that is diagonal in the
eigenbasis with weights `w`: p[c] = Σ_n V[c,n]² w_n. Covers the thermal state
(`w = thermal_weights(values, β)`), the diagonal ensemble (`w = abs2.(c0)`),
and single eigenstates (`w = e_k`). Processes `V` in column blocks so no
dim×dim temporary is formed (the N=16 matrix is 1.3 GB).
"""
function diagonal_probabilities(V::AbstractMatrix{<:Real}, w::AbstractVector{<:Real};
                                block::Int = 2048)
    dim, nvec = size(V)
    @assert length(w) == nvec "weight vector must match eigenvector count"
    p = zeros(Float64, dim)
    for lo in 1:block:nvec
        hi = min(lo + block - 1, nvec)
        @views p .+= abs2.(V[:, lo:hi]) * w[lo:hi]
    end
    return p
end

"""
    quench_amplitudes(V, ψi)

Eigenbasis amplitudes c0 = V′ψi of an initial state. `abs2.(c0)` are the
diagonal-ensemble weights; Σ|c0|² should be 1.
"""
quench_amplitudes(V::AbstractMatrix{<:Real}, ψi::AbstractVector{<:Real}) =
    transpose(V) * ψi

"""
    quench_probabilities(V, values, c0, ts; block=256)

Config-basis probability vectors of the time-evolved pure state,

    p_t[c] = |⟨c|ψ(t)⟩|²,   ψ(t) = V (c0 .* e^{−iE t}),

for every t in `ts` at once. Returns a dim × length(ts) matrix (column k =
p at ts[k]). Uses two real GEMMs per time block (V is real):
Re ψ = V(c0 cos Et), Im ψ = −V(c0 sin Et).
"""
function quench_probabilities(V::AbstractMatrix{<:Real}, values::AbstractVector{<:Real},
                              c0::AbstractVector{<:Real}, ts::AbstractVector{<:Real};
                              block::Int = 256)
    dim, nvec = size(V)
    P = Matrix{Float64}(undef, dim, length(ts))
    for lo in 1:block:length(ts)
        hi  = min(lo + block - 1, length(ts))
        tsb = ts[lo:hi]
        A = c0 .* cos.(values .* tsb')          # nvec × nt
        B = c0 .* sin.(values .* tsb')
        P[:, lo:hi] .= (V * A) .^ 2 .+ (V * B) .^ 2
    end
    return P
end

"""
    quench_state(V, values, c0, t)

The time-evolved pure state ψ(t) = V (c0 .* e^{−iEt}) itself (complex vector,
sector basis) — for observables that need more than the config-basis
probabilities, e.g. entanglement entropy.
"""
quench_state(V::AbstractMatrix{<:Real}, values::AbstractVector{<:Real},
             c0::AbstractVector{<:Real}, t::Real) =
    (V * (c0 .* cos.(values .* t))) .- im .* (V * (c0 .* sin.(values .* t)))
