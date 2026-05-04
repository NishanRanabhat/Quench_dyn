# scripts/run_tdvp.jl
#
# Quench dynamics via TDVP time evolution of the 1D XXZ chain.
#
# Protocol:
#   1) Find the ground state of H(Delta_i) using DMRG.
#   2) Switch to H(Delta_f) and evolve with TDVP.
#   3) Measure observables at regular intervals.
#
# Hamiltonian:
#   H = J Σ_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1})
#     + Δ Σ_i  Sz_i Sz_{i+1}
#     + Σ_i    h_i  Sz_i

using LinearAlgebra
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

# BLAS multi-threading for dense linear algebra (SVD, matmul inside tensor
# contractions). Override by setting BLAS_THREADS in the environment, e.g.
#   BLAS_THREADS=8 julia run_tdvp.jl
#BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", string(Sys.CPU_THREADS))))

# ═══════════════════════════════════════════════════════════════════════════
# MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

N     = 32       # number of sites
J     = 1.0      # XY coupling strength (sets the energy scale)
d     = 2        # local Hilbert space dimension (2 for spin-1/2)

# Quench: Delta_i → Delta_f
# Reverse quench: start deep in Ising phase, quench into XY phase
Delta_i = 0.5    # initial Ising anisotropy (ground state is prepared here)
Delta_f = 3.0    # final Ising anisotropy (time evolution under this)

# Site-dependent longitudinal field h_i (shared by both pre- and post-quench H).
# Options:
#   zeros(N)                                      — no field
#   fill(0.5, N)                                  — uniform field
#   [0.5 * (-1.0)^i for i in 1:N]                 — staggered field
#   randn(N) * 0.1                                — random disorder
#h = zeros(N)
h = [0.1 * (-1.0)^i for i in 1:N]
# ═══════════════════════════════════════════════════════════════════════════
# DMRG PARAMETERS (for ground state preparation)
# ═══════════════════════════════════════════════════════════════════════════

chi_max_dmrg = 64    # maximum bond dimension for DMRG
n_sweeps     = 30     # total DMRG sweeps
cutoff_dmrg  = 1e-10  # SVD truncation cutoff for DMRG

# Lanczos eigensolver parameters (for the local effective Hamiltonian in DMRG)
lanczos_krylov_dim = 4     # Lanczos vectors per iteration
lanczos_max_iter   = 14   # maximum Lanczos restarts

# ═══════════════════════════════════════════════════════════════════════════
# TDVP PARAMETERS (for time evolution)
# ═══════════════════════════════════════════════════════════════════════════

dt          = 0.05     # time step (smaller = more accurate, slower)
t_max       = 5.0      # total evolution time
chi_max_tdvp = 256     # maximum bond dimension during TDVP
cutoff_tdvp  = 1e-8   # SVD truncation cutoff during TDVP

# Krylov exponential solver parameters (for the local time evolution in TDVP)
krylov_dim  = 14       # Krylov subspace dimension (larger = more accurate per step)
krylov_tol  = 1e-8    # convergence tolerance for Krylov exponential

# Evolution type:
#   "real"      — real-time evolution e^{-iHt} (unitary dynamics)
#   "imaginary" — imaginary-time evolution e^{-Ht} (cooling / ground state projection)
evol_type   = "real"

# ═══════════════════════════════════════════════════════════════════════════
# MEASUREMENT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# How often to measure observables (in number of time steps)
measure_every = 10     # measure every this many steps (set 1 for every step)

# ═══════════════════════════════════════════════════════════════════════════
# INITIAL STATE (via DMRG at Delta_i)
# ═══════════════════════════════════════════════════════════════════════════

n_steps = round(Int, t_max / dt)

println("="^70)
println("  Quench dynamics: TDVP time evolution")
println("="^70)
println("\nModel: N=$N, J=$J, d=$d")
println("Quench: Delta_i=$Delta_i → Delta_f=$Delta_f")
println("Field: h = ", length(h) <= 10 ? "$h" : "[$(h[1]), $(h[2]), ..., $(h[end])]")
println("\nDMRG (ground state): chi_max=$chi_max_dmrg, n_sweeps=$n_sweeps, cutoff=$cutoff_dmrg")
println("  Lanczos: krylov_dim=$lanczos_krylov_dim, max_iter=$lanczos_max_iter")
println("\nTDVP (evolution):    chi_max=$chi_max_tdvp, dt=$dt, t_max=$t_max, n_steps=$n_steps")
println("  Krylov: krylov_dim=$krylov_dim, tol=$krylov_tol, evol_type=$evol_type")
println("  Measure every $measure_every steps")
println("\nBLAS threads: $(BLAS.get_num_threads())")

# ── Build initial Hamiltonian and find ground state ────────────────────

println("\n" * "─"^70)
println("  Step 1: DMRG ground state at Delta_i=$Delta_i")
println("─"^70)

mpo_i = build_xxz_mpo(N, J, Delta_i, h; d=d)

sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]   # Neel state
mps = product_state(sites, labels)

state = MPSState(mps, mpo_i; center=1)
solver_dmrg = LanczosSolver(lanczos_krylov_dim, lanczos_max_iter)
opts_dmrg = DMRGOptions(chi_max_dmrg, cutoff_dmrg, d)

println("  Sweep   max_χ   max_trunc    total_trunc       energy")
println("  " * "─"^60)

E_gs = 0.0
for sweep in 1:n_sweeps
    global E_gs
    dir = isodd(sweep) ? :right : :left
    res = dmrg_sweep(state, solver_dmrg, opts_dmrg, dir)
    E_gs = res.E
    println("  $(lpad(sweep, 3))     $(lpad(res.max_chi, 4))   $(lpad(round(res.max_trunc, sigdigits=3), 10))   $(lpad(round(res.total_trunc, sigdigits=3), 10))    $E_gs")
end

var_E = energy_variance(state)
println("  " * "─"^60)
println("  Ground state energy: $E_gs")
println("  Norm: $(measure_norm(state.mps))")
println("  Energy variance ⟨H²⟩ − ⟨H⟩²: $var_E   (zero at exact eigenstate)")

# ── Measure initial state observables ──────────────────────────────────

ops = spin_ops(d)
Sz = Matrix{ComplexF64}(ops[:Z])

sz_init = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
m_stag_init = sum((-1.0)^i * sz_init[i] for i in 1:N) / N

println("\n  Initial staggered magnetization: M_stag = $m_stag_init")

# ═══════════════════════════════════════════════════════════════════════════
# QUENCH: Switch Hamiltonian to Delta_f and evolve with TDVP
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  Step 2: Quench to Delta_f=$Delta_f — TDVP evolution")
println("─"^70)

mpo_f = build_xxz_mpo(N, J, Delta_f, h; d=d)

# Rebuild MPSState with the new (post-quench) MPO
state = MPSState(state.mps, mpo_f; center=1)

solver_tdvp = KrylovExponential(krylov_dim, krylov_tol, evol_type)
tdvp_opts = TDVPOptions(dt, chi_max_tdvp, cutoff_tdvp, d)

# Storage for time series
times       = Float64[]
energies    = Float64[]
norms       = Float64[]
m_stag_t    = Float64[]
sz_profiles = Vector{Float64}[]
max_chis    = Int[]
max_truncs  = Float64[]
total_truncs = Float64[]

# Record t=0
push!(times, 0.0)
push!(energies, measure_energy(state))
push!(norms, real(measure_norm(state.mps)))
push!(m_stag_t, m_stag_init)
push!(sz_profiles, sz_init)
push!(max_chis, maximum(size(t, 3) for t in state.mps.tensors[1:end-1]))
push!(max_truncs, 0.0)
push!(total_truncs, 0.0)

E0 = energies[1]

println("  step     t       energy           dE          1-‖ψ‖²        χ_max    max_trunc")
println("  " * "─"^85)
println("  $(lpad(0, 4))   $(lpad("0.000", 6))   $(round(E0, digits=8))   $(round(0.0, digits=4))   $(round(1.0 - norms[1], sigdigits=3))    $(lpad(max_chis[1], 4))    $(round(0.0, sigdigits=3))")

for step in 1:n_steps
    global state

    res_r = tdvp_sweep(state, solver_tdvp, tdvp_opts, :right)
    res_l = tdvp_sweep(state, solver_tdvp, tdvp_opts, :left)

    t = step * dt

    if step % measure_every == 0 || step == n_steps
        E = measure_energy(state)
        nrm = real(measure_norm(state.mps))
        sz = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
        m_stag = sum((-1.0)^i * sz[i] for i in 1:N) / N
        mc = max(res_r.max_chi, res_l.max_chi)
        mt = max(res_r.max_trunc, res_l.max_trunc)
        tt = res_r.total_trunc + res_l.total_trunc

        push!(times, t)
        push!(energies, E)
        push!(norms, nrm)
        push!(m_stag_t, m_stag)
        push!(sz_profiles, sz)
        push!(max_chis, mc)
        push!(max_truncs, mt)
        push!(total_truncs, tt)

        println("  $(lpad(step, 4))   $(lpad(round(t, digits=3), 6))   $(round(E, digits=8))   $(round(E - E0, sigdigits=3))   $(round(1.0 - nrm, sigdigits=3))    $(lpad(mc, 4))    $(round(mt, sigdigits=3))")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# FINAL MEASUREMENTS
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  Final state at t=$t_max")
println("─"^70)

# ── Local Sz profile ───────────────────────────────────────────────────

println("\n  Local ⟨Sz_i⟩ profile")
println("  " * "─"^50)
sz_final = sz_profiles[end]
for i in 1:N
    bar = repeat("█", max(0, round(Int, (sz_final[i] + 0.5) * 40)))
    println("  Site $(lpad(i, 3)):  $(lpad(round(sz_final[i], digits=8), 12))  $bar")
end

# ── Bond entanglement entropies ────────────────────────────────────────

println("\n  Bond entanglement entropy S(i, i+1)")
println("  " * "─"^50)

mps_copy = MPS{eltype(state.mps.tensors[1])}(copy.(state.mps.tensors))
for bond in 1:N-1
    make_canonical(mps_copy, bond)
    A = mps_copy.tensors[bond]
    chi_l, d_loc, chi_r = size(A)
    F = svd(reshape(A, chi_l * d_loc, chi_r))
    S_ent = entropy(F.S)
    bar = repeat("█", max(0, round(Int, S_ent * 20)))
    println("  Bond $(lpad(bond, 3))-$(lpad(bond+1, 3)):  S = $(lpad(round(S_ent, digits=8), 12))  $bar")
end

# ── Staggered magnetization time series ────────────────────────────────

println("\n  Staggered magnetization M_stag(t)")
println("  " * "─"^50)
for k in eachindex(times)
    println("  t = $(lpad(round(times[k], digits=3), 8))   M_stag = $(round(m_stag_t[k], digits=8))")
end

# ── Energy conservation check ──────────────────────────────────────────

println("\n  Energy conservation (post-quench Hamiltonian)")
println("  " * "─"^50)
E_post_quench = energies[1]
max_E_drift = maximum(abs.(energies .- E_post_quench))
println("  E(t=0)  = $E_post_quench")
println("  max|E(t) - E(0)| = $max_E_drift")

# ── Norm conservation check ───────────────────────────────────────────

max_norm_drift = maximum(abs.(norms .- 1.0))
println("\n  Norm conservation")
println("  max|⟨ψ|ψ⟩ - 1| = $max_norm_drift")

# ── CSV log for offline plotting ──────────────────────────────────────
csv_path = joinpath(@__DIR__, "quench_log.csv")
open(csv_path, "w") do io
    println(io, "step,t,energy,dE,norm_sq,one_minus_norm_sq,m_stag,max_chi,max_trunc,total_trunc")
    for k in eachindex(times)
        step_k = (k == 1) ? 0 : (k - 1) * measure_every
        println(io, "$step_k,$(times[k]),$(energies[k]),$(energies[k]-E0),$(norms[k]),$(1.0-norms[k]),$(m_stag_t[k]),$(max_chis[k]),$(max_truncs[k]),$(total_truncs[k])")
    end
end
println("\n  Diagnostics CSV: $csv_path")

println("\n" * "="^70)
println("  Done.")
println("="^70)
