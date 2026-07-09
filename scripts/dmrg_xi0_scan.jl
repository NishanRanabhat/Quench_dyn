# scripts/dmrg_xi0_scan.jl
#
# Ground-state staggered correlation length ξ₀(Δ_f) of the final XXZ Hamiltonian,
# via DMRG. ξ₀ is the coarsening CEILING for a low-T_eff (tight, transition-
# crossing) quench: at small T_eff the thermal state ≈ the ground state, so the
# largest domain a quench can build is set by the ground-state correlation
# length of H_f — a pure ground-state property, no thermal state or dynamics.
#
# Near the BKT point Δ=1, ξ₀ grows (exponentially) as Δ_f → 1⁺, so we scan
# Δ_f ∈ (1, 1.5] at N = 32, 64, 128 to (a) see how large ξ₀ can be made and
# (b) check whether it is converged in N and χ (χ ≤ 128). If ξ₀ keeps growing
# with N/χ near Δ=1, that Δ_f is under-resolved (long-range order not captured).
#
# Run:  julia scripts/dmrg_xi0_scan.jl

using LinearAlgebra
using Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", string(Sys.CPU_THREADS))))

const J        = 1.0
const d        = 2
const Δfs      = (1.05, 1.10, 1.20, 1.30, 1.50)
const Ns       = (32, 64, 128)
const CHI      = 128
const N_WARMUP = 14      # 2-site sweeps (grow χ)
const N_REFINE = 6       # 1-site sweeps (refine at fixed χ)
const CUTOFF   = 1e-8
const KRYLOV   = 4
const MAXITER  = 14
const RMAX_CAP = 50      # cap on correlator range
const MAX_REFS = 20      # cap on bulk reference sites (strided)

# ── DMRG ground state (2-site warmup + 1-site refine), returns (mps, E, maxχ, var)
function dmrg_ground_state(N, Δ)
    mpo   = build_xxz_mpo(N, J, Δ, zeros(N); d = d)
    sites = [SpinSite(0.5; T = ComplexF64) for _ in 1:N]
    mps   = product_state(sites, [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N])
    state = MPSState(mps, mpo; center = 1)
    solver = LanczosSolver(KRYLOV, MAXITER)
    opts   = DMRGOptions(CHI, CUTOFF, d)

    E = 0.0; maxχ = 0
    for s in 1:N_WARMUP
        res = dmrg_sweep(state, solver, opts, isodd(s) ? :right : :left)
        E = res.E; maxχ = res.max_chi
    end
    for _ in 1:N_REFINE
        res = dmrg_sweep_one_site(state, solver, opts,
                                  state.center == 1 ? :right : :left)
        E = res.E; maxχ = res.max_chi
    end
    return state, E, maxχ, energy_variance(state)
end

# ── bulk-averaged staggered correlator G(r) = (−1)^r mean_i ⟨S^z_i S^z_{i+r}⟩
function staggered_correlator_mps(mps, N, rmax)
    Sz = Matrix{ComplexF64}(spin_ops(d)[:Z])
    lo = max(3, N ÷ 4)
    hi = N - rmax - 2                       # so i+r stays interior for all r
    hi <= lo && (hi = lo)
    refs = collect(lo:hi)
    length(refs) > MAX_REFS &&
        (refs = refs[round.(Int, range(1, length(refs); length = MAX_REFS))])
    G = zeros(Float64, rmax + 1)
    for r in 0:rmax
        acc = 0.0
        for i in refs
            acc += (r == 0 ? real(measure_local_observable(mps, Sz * Sz, i))
                           : real(measure_correlation(mps, Sz, i, Sz, i + r)))
        end
        G[r + 1] = (-1.0)^r * acc / length(refs)
    end
    return 0:rmax, G
end

# ξ from exponential fit of the normalized staggered envelope
function xi_expfit(rs, G)
    G0 = G[1]
    xs = Float64[]; ys = Float64[]
    for (r, g) in zip(rs, G)
        r == 0 && continue
        g / G0 <= 1e-4 && break
        push!(xs, float(r)); push!(ys, log(g / G0))
    end
    length(xs) < 2 && return NaN
    slope = sum(xs .* ys) / sum(xs .^ 2)
    return slope < 0 ? -1 / slope : NaN
end

println("XXZ ground-state staggered correlation length ξ₀(Δf),  J=$J, χ≤$CHI\n")
@printf("%6s  %4s  %6s  %12s  %8s  %8s  %10s\n",
        "Δf", "N", "maxχ", "E/N", "ξ₀_fit", "G(rmax)", "var/N")
println("-"^64)

for Δf in Δfs
    for N in Ns
        rmax = min(N ÷ 2 - 2, RMAX_CAP)
        t = @elapsed begin
            state, E, maxχ, var = dmrg_ground_state(N, Δf)
            rs, G = staggered_correlator_mps(state.mps, N, rmax)
            ξ0 = xi_expfit(rs, G)
        end
        @printf("%6.2f  %4d  %6d  %12.6f  %8.3f  %8.4f  %10.2e   [%.0fs]\n",
                Δf, N, maxχ, E / N, ξ0, G[end] / G[1], var / N, t)
        flush(stdout)
    end
    println(); flush(stdout)
end
