# ED/thermal.jl
#
# Finite-temperature (canonical Gibbs) observables from a full sector
# eigensystem. Used to answer the equilibrium gating question for coarsening:
# at the effective temperature set by a quench's energy density, what is the
# staggered correlation length ξ(T_eff)?
#
# The quench deposits a fixed energy E_target = ⟨ψ_0|H_f|ψ_0⟩. Matching this to
# the Gibbs ensemble of H_f fixes β (hence T_eff = 1/β). Because S^z is diagonal
# in the computational basis, every S^z–S^z thermal correlator reduces to a
# weighted sum over basis configs with weights = the diagonal of the thermal
# density matrix — no operator matrices in Hilbert space are ever formed.
#
# All routines take the sector eigensystem (values sorted ascending, vectors as
# columns in the sector's computational basis) produced by `diagonalize`.

"""
    thermal_energy(evals, β)

Canonical energy ⟨H⟩_β = Σ_n E_n e^{−βE_n} / Σ_n e^{−βE_n} from the eigenvalues
`evals`. Uses a log-sum-exp shift so it is stable for any sign/size of `β`.
Monotonically decreasing in `β`.
"""
function thermal_energy(evals::AbstractVector{<:Real}, β::Real)
    shift = maximum(-β .* evals)
    w = exp.(-β .* evals .- shift)
    Z = sum(w)
    return sum(w .* evals) / Z
end

"""
    effective_beta(evals, E_target; tol=1e-12, βmax=1e6)

Inverse effective temperature β solving ⟨H⟩_β = `E_target`, by bisection on the
monotonic `thermal_energy`. `E_target` must lie strictly between the ground-state
energy `minimum(evals)` (β → +∞, T → 0⁺) and the infinite-temperature mean
`mean(evals)` (β = 0); values above the mean give negative β (negative
temperature). Returns `β`; take `1/β` for T_eff.

Warns and clamps to ±`βmax` if `E_target` is within `tol` of the spectrum edges
(cold/hot limits) where β diverges.
"""
function effective_beta(evals::AbstractVector{<:Real}, E_target::Real;
                        tol::Real = 1e-12, βmax::Real = 1e6)
    Emin, Emax = extrema(evals)
    if E_target <= Emin + tol
        @warn "E_target ≈ ground-state energy; T_eff → 0, clamping β = βmax" E_target Emin
        return float(βmax)
    end
    if E_target >= Emax - tol
        @warn "E_target ≈ top of spectrum; T_eff → 0⁻, clamping β = −βmax" E_target Emax
        return float(-βmax)
    end
    g(β) = thermal_energy(evals, β) - E_target   # decreasing in β; want g = 0
    # bracket: β_lo has g>0 (energy too high), β_hi has g<0 (energy too low)
    βlo, βhi = -1.0, 1.0
    while g(βlo) < 0
        βlo *= 2
        βlo < -βmax && (return float(-βmax))
    end
    while g(βhi) > 0
        βhi *= 2
        βhi > βmax && (return float(βmax))
    end
    for _ in 1:200
        βmid = (βlo + βhi) / 2
        gmid = g(βmid)
        abs(gmid) < tol && return βmid
        if gmid > 0
            βlo = βmid
        else
            βhi = βmid
        end
    end
    return (βlo + βhi) / 2
end

"""
    effective_temperature(evals, E_target; kwargs...)

Convenience wrapper returning `T_eff = 1 / effective_beta(...)` (units of J).
"""
effective_temperature(evals::AbstractVector{<:Real}, E_target::Real; kwargs...) =
    1 / effective_beta(evals, E_target; kwargs...)

"""
    thermal_diagonal(eig::EDEigensystem, β)

Diagonal of the thermal density matrix ρ = e^{−βH}/Z in the sector's
computational basis:

    ρ_diag[c] = Σ_n e^{−βE_n} |⟨c|n⟩|² / Z .

Returns a probability vector (length = sector dim, sums to 1). This is the only
quantity needed for any S^z–S^z thermal correlator.
"""
function thermal_diagonal(eig::EDEigensystem, β::Real)
    E = eig.values
    shift = maximum(-β .* E)
    w = exp.(-β .* E .- shift)
    Z = sum(w)
    ρ = (abs2.(eig.vectors) * w) ./ Z
    return real.(ρ)
end

"""
    thermal_sz_correlations(sec::SzSector, ρ_diag)

Thermal one- and two-point S^z data from a density-matrix diagonal `ρ_diag`
(from `thermal_diagonal`):

    m[j]    = ⟨S^z_j⟩       = Σ_c ρ_diag[c] · sz[c,j]
    C[j,k]  = ⟨S^z_j S^z_k⟩ = Σ_c ρ_diag[c] · sz[c,j] · sz[c,k]

Returns `(m, C)` with `m::Vector{Float64}` (length N) and `C::Matrix{Float64}`
(N×N). Feed straight into the `coarsening.jl` observables. In the S^z=0 sector
`m ≡ 0` by construction, so the connected and full C coincide.
"""
function thermal_sz_correlations(sec::SzSector, ρ_diag::AbstractVector{<:Real})
    @assert length(ρ_diag) == length(sec) "ρ_diag length must equal sector dim"
    return thermal_sz_correlations(sec.sz, ρ_diag)
end

# generic version: `sz` is a (dim × N) matrix of S^z eigenvalues per basis state
function thermal_sz_correlations(sz::AbstractMatrix{<:Real}, ρ_diag::AbstractVector{<:Real})
    @assert size(sz, 1) == length(ρ_diag) "sz rows must equal length(ρ_diag)"
    m = transpose(sz) * ρ_diag
    C = transpose(sz) * (ρ_diag .* sz)
    return Vector{Float64}(m), Matrix{Float64}(C)
end

"""
    full_sz_values(N; d=2)

`(d^N) × N` matrix of S^z eigenvalues, `sz[c,j]` = the S^z value of site j in
the c-th computational basis state, in the SAME ordering as
`build_xxz_hamiltonian` / `diagonalize` (site 1 most significant; local index
1 = up = +1/2, via the `basis_state` convention). For full-Hilbert-space
(all-sector) thermal averages — e.g. validating a full-ensemble purification,
which samples every magnetization sector rather than only S^z=0.
"""
function full_sz_values(N::Int; d::Int = 2)
    D = d^N
    mvals = collect((d - 1) / 2 : -1 : -(d - 1) / 2)   # index 1 = up (+1/2)
    sz = Matrix{Float64}(undef, D, N)
    for c in 0:D-1, j in 1:N
        digit = (c ÷ d^(N - j)) % d                    # 0 = up, 1 = down, ...
        sz[c + 1, j] = mvals[digit + 1]
    end
    return sz
end
