# scripts/run_finite_T.jl
#
# Finite-temperature states of the 1D XXZ chain by IMAGINARY-TIME TDVP of an
# ancilla purification. Same pipeline as run_tdvp_2site.jl — an explicit loop
# over n_steps of 2-site TDVP sweeps — with two changes only:
#   * evol_type = "imaginary"  →  local propagator e^{-Δτ H_eff} (cooling),
#     instead of the real-time e^{-i Δt H_eff}.
#   * the state is a purification (local dim d²=4 = physical⊗ancilla) evolved
#     under H⊗I_anc; tracing the ancilla gives ρ ∝ e^{-2τH}, so β = 2τ.
# Reaching final imaginary time τ_max = β_max/2 gives the state at inverse
# temperature β_max.  2-site (not 1-site) because cooling starts from a
# bond-dimension-1 product seed and must GROW χ.
#
# Method + conventions: docs/finite_T_purification_protocol.md
#
# Hamiltonian:
#   H = J Σ_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1}) + Δ Σ_i Sz_i Sz_{i+1} + Σ_i h_i Sz_i

using LinearAlgebra
using Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", string(Sys.CPU_THREADS))))

# ═══════════════════════════════════════════════════════════════════════════
# MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

N     = 32        # number of sites
J     = 1.0       # XY coupling (energy scale)
Delta = 1.2       # Ising anisotropy of the Hamiltonian to thermalize
d     = 2         # local Hilbert space dimension (2 for spin-1/2)
h     = zeros(N)  # site-dependent longitudinal field (length N); zeros(N) = clean

# ═══════════════════════════════════════════════════════════════════════════
# IMAGINARY-TIME TDVP PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

dt          = 0.02      # imaginary-time step per round-trip; error O(dt²)
beta_max    = 8.0      # final inverse temperature (T_min = 1/beta_max); τ_max = β/2
chi_max     = 200      # maximum bond dimension for the purification
cutoff      = 1e-8     # SVD truncation cutoff

krylov_dim  = 14       # Krylov subspace dimension (local matrix exponential)
krylov_tol  = 1e-8     # convergence tolerance for the Krylov exponential

# ═══════════════════════════════════════════════════════════════════════════
# MEASUREMENT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

measure_every = 5      # measure observables every this many steps

# ═══════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════

n_steps = round(Int, (beta_max / 2) / dt)   # τ_max = β_max/2, one step = dt

println("="^70)
println("  Finite-T states: imaginary-time TDVP of a purification")
println("="^70)
println("\nModel: N=$N, J=$J, Δ=$Delta, d=$d")
println("Field: h = ", length(h) <= 10 ? "$h" : "[$(h[1]), $(h[2]), ..., $(h[end])]")
println("\nImag-time TDVP: dt=$dt, β_max=$beta_max (τ_max=$(beta_max/2)), n_steps=$n_steps")
println("  chi_max=$chi_max, cutoff=$cutoff")
println("  Krylov: krylov_dim=$krylov_dim, tol=$krylov_tol, evol_type=imaginary")
println("  Measure every $measure_every steps")
println("\nBLAS threads: $(BLAS.get_num_threads())")

# ── Build purified Hamiltonian H⊗I and the β=0 (infinite-T) seed ─────────
mpo_phys = build_xxz_mpo(N, J, Delta, h; d = d)   # physical MPO
mpo_pur  = purify_mpo(mpo_phys; d = d)            # H ⊗ I_anc  (bond dim unchanged)

psi0  = maximally_mixed_purification(N; d = d)    # ρ(β=0) = I/d^N, bond dim 1
state = MPSState(psi0, mpo_pur; center = 1)

solver = KrylovExponential(krylov_dim, krylov_tol, "imaginary")
opts   = TDVPOptions(dt, chi_max, cutoff, d * d)  # NOTE local dim = d² = 4

Sz = embed_physical(Matrix{Float64}(spin_ops(d)[:Z]); d = d)   # S^z ⊗ I_anc

# ═══════════════════════════════════════════════════════════════════════════
# COOL: explicit loop over imaginary-time steps  (β = 2·step·dt)
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  Step 1: imaginary-time evolution   (β = 2τ,  ρ ∝ e^{-βH})")
println("─"^70)

# Storage for the cooling trajectory
betas       = Float64[]
temps       = Float64[]
energies    = Float64[]     # ⟨H⟩/N
xis         = Float64[]     # staggered correlation length
ms2s        = Float64[]     # ⟨m_s²⟩ = S(π)/N
max_chis    = Int[]
max_truncs  = Float64[]

bulk = N >= 24 ? collect(round(Int, N/4):round(Int, 3N/4)) : collect(2:N-1)

println("  step      β         T          ⟨H⟩/N          χ       ξ        max_trunc")
println("  " * "─"^72)

for step in 1:n_steps
    global state
    tdvp_sweep(state, solver, opts, :right)
    res = tdvp_sweep(state, solver, opts, :left)
    # renormalize the orthogonality center (range control; observables are ratios)
    state.mps.tensors[state.center] ./= sqrt(measure_norm(state.mps))

    if step % measure_every == 0 || step == n_steps
        β    = 2 * step * dt
        Z    = measure_norm(state.mps)
        E    = measure_energy(state) / Z
        _, C = thermal_sz_from_purification(state; d = d)
        rs, G = staggered_correlator(C; rmax = N ÷ 2, bulk = bulk)
        ξ    = domain_length(rs, G)
        ms2  = staggered_magnetization_sq(C)
        mc   = maximum(size(t, 3) for t in state.mps.tensors[1:end-1])

        push!(betas, β); push!(temps, 1/β); push!(energies, E)
        push!(xis, ξ); push!(ms2s, ms2); push!(max_chis, mc)
        push!(max_truncs, res.max_trunc)

        @printf("  %4d   %7.3f   %8.4f   %12.6f   %5d   %7.3f   %9.2e\n",
                step, β, 1/β, E, mc, ξ, res.max_trunc)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# FINAL STATE (at β_max)
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  Final thermal state at β=$(betas[end])  (T=$(round(temps[end], digits=4)))")
println("─"^70)

_, C = thermal_sz_from_purification(state; d = d)
rs, G = staggered_correlator(C; rmax = N ÷ 2, bulk = bulk)

println("  Energy density ⟨H⟩/N = $(round(energies[end], digits=6))")
println("  Staggered correlation length ξ = $(round(xis[end], digits=4))")
println("  ⟨m_s²⟩ = S(π)/N = $(round(ms2s[end], sigdigits=5))")
println("  Max bond dimension used = $(max_chis[end])  (cap = $chi_max)")

println("\n  Staggered correlator  G(r)/G(0):")
for (r, g) in zip(rs, G)
    @printf("    r=%2d   %+8.5f\n", r, g / G[1])
end

# ═══════════════════════════════════════════════════════════════════════════
# CSV LOG (for offline plotting) — parity with the TDVP scripts, no state files
# ═══════════════════════════════════════════════════════════════════════════

csv_path = joinpath(@__DIR__, "finite_T_log.csv")
open(csv_path, "w") do io
    println(io, "beta,T,E_per_N,chi,xi,ms2,max_trunc")
    for k in eachindex(betas)
        println(io, "$(betas[k]),$(temps[k]),$(energies[k]),$(max_chis[k]),$(xis[k]),$(ms2s[k]),$(max_truncs[k])")
    end
end
println("\n  Cooling log CSV: $csv_path")

println("\n" * "="^70)
println("  Done.")
println("="^70)
