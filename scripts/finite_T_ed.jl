# scripts/finite_T_ed.jl
#
# Equilibrium gating check for XXZ quench coarsening, via exact diagonalization
# in the S^z=0 sector (the sector the quench conserves).
#
# For the reference quench Δ_i → Δ_f at each system size N:
#   1. ψ_0 = ground state of H_i (Δ_i) in the S^z=0 sector
#   2. E_target = ⟨ψ_0|H_f|ψ_0⟩          (energy the quench deposits)
#   3. T_eff from energy-matching to the Gibbs ensemble of H_f
#   4. thermal S^z–S^z correlator C_β  →  staggered G(r,T_eff)  →  ξ(T_eff)
#
# The question: is ξ(T_eff) small (~few sites, N-independent ⇒ no coarsening
# window) or large/growing with N (⇒ purification at larger N is warranted)?
# The T=0 ground-state correlator of H_f is printed alongside as the "ordered"
# reference the quench would have to approach for coarsening to be real.
#
# Run:  julia scripts/finite_T_ed.jl

using LinearAlgebra
using Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

const J  = 1.0
const Ns = (10, 12, 14)

# Tight crossing protocols: start gapless (Δ<1), end gapped (Δ>1), close to the
# Δ=1 transition to keep the deposited energy — hence T_eff — small. Symmetric
# pairs (1−δ, 1+δ) parametrized by half-width δ, plus the old wide reference.
const PROTOCOLS = [
    (0.95, 1.05),
    (0.90, 1.10),
    (0.80, 1.20),
    (0.70, 1.30),
    (0.80, 3.00),   # old wide reference, for contrast
]

# interior window to suppress open-boundary edge effects in the correlator
interior(N) = N >= 12 ? (3:(N-2)) : (2:(N-1))

# staggered correlator of a pure state ψ (in the sector computational basis)
function state_staggered_corr(sec::SzSector, ψ::AbstractVector)
    p = abs2.(ψ)
    return transpose(sec.sz) * (p .* sec.sz)   # C[j,k] = Σ_c |ψ_c|² sz sz
end

# ξ from the normalized staggered correlator, exponential fit G(r)/G(0) ~ e^{-r/ξ}
function xi_expfit(rs, G)
    G0 = G[1]
    xs = Float64[]; ys = Float64[]
    for (r, g) in zip(rs, G)
        r == 0 && continue
        g / G0 <= 1e-6 && break          # stop before the noise floor / first zero
        push!(xs, float(r)); push!(ys, log(g / G0))
    end
    length(xs) < 2 && return NaN
    slope = sum(xs .* ys) / sum(xs .^ 2) # origin-anchored least-squares slope
    return slope < 0 ? -1 / slope : NaN
end

# one (protocol, N) point → (T_eff, ξ_fit, ξ_int, S(π), staggered ξ_0 at T=0)
function run_point(Δi, Δf, N)
    sec = sz_sector(N, N ÷ 2)
    Hi  = build_xxz_hamiltonian_sector(sec, J, Δi)       # clean, h = 0
    Hf  = build_xxz_hamiltonian_sector(sec, J, Δf)

    _, ψ0    = ground_state(Hi)
    E_target = real(dot(ψ0, Hf * ψ0))

    eig  = diagonalize(Hf)
    β    = effective_beta(eig.values, E_target)
    Teff = 1 / β

    _, C = thermal_sz_correlations(sec, thermal_diagonal(eig, β))
    rmax, bulk = N ÷ 2, collect(interior(N))
    rs, G = staggered_correlator(C; rmax = rmax, bulk = bulk)

    # T=0 reference: ground-state correlator of H_f (order to coarsen into)
    _, ψf   = ground_state(Hf)
    _, G0   = staggered_correlator(state_staggered_corr(sec, ψf);
                                   rmax = rmax, bulk = bulk)
    return (; Teff, ξfit = xi_expfit(rs, G), ξint = domain_length(rs, G),
              Sπ = structure_factor(C, [Float64(π)])[1],
              ξ0 = xi_expfit(rs, G0), ms0 = sqrt(max(G0[1], 0.0)))
end

for (Δi, Δf) in PROTOCOLS
    @printf("\nProtocol  Δi=%.2f → Δf=%.2f   (δ=%.2f from Δ=1)\n",
            Δi, Δf, (Δf - Δi) / 2)
    @printf("%3s  %8s  %8s  %8s  %10s  %8s\n",
            "N", "T_eff", "ξ_fit", "ξ_int", "S(π)", "ξ0(T=0)")
    println("-"^54)
    for N in Ns
        p = run_point(Δi, Δf, N)
        @printf("%3d  %8.4f  %8.3f  %8.3f  %10.4f  %8.3f\n",
                N, p.Teff, p.ξfit, p.ξint, p.Sπ, p.ξ0)
    end
end
