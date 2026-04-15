# scripts/run_dmrg.jl
#
# Find the ground state of the 1D XXZ chain with site-dependent field using DMRG.
#
#   H = J Σ_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1})
#     + Δ Σ_i  Sz_i Sz_{i+1}
#     + Σ_i    h_i  Sz_i
#
# Output: ground state energy, Sz profile, staggered magnetization,
#         Sz-Sz correlations, and bond entanglement entropies.

using LinearAlgebra
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

# ═══════════════════════════════════════════════════════════════════════════
# MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

N     = 20       # number of sites
J     = 1.0      # XY coupling strength (sets the energy scale)
Delta = 3.0      # Ising anisotropy Sz·Sz (Delta/J > 1 is the Ising phase)
d     = 2        # local Hilbert space dimension (2 for spin-1/2)

# Site-dependent longitudinal field h_i.
# Options:
#   zeros(N)                                      — no field
#   fill(0.5, N)                                  — uniform field
#   [0.5 * (-1.0)^i for i in 1:N]                 — staggered field
#   randn(N) * 0.1                                — random disorder
h = zeros(N)

# ═══════════════════════════════════════════════════════════════════════════
# DMRG PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

chi_max   = 128      # maximum bond dimension (controls accuracy vs cost)
n_sweeps  = 30       # total number of DMRG sweeps
cutoff    = 1e-10    # SVD truncation cutoff (discard singular values below this)

# SweepSchedule ramps chi linearly from chi_min to chi_max over the first
# half of sweeps, then holds at chi_max. Cutoff tightens similarly.
chi_min   = max(chi_max ÷ 8, 2)   # starting bond dimension for the ramp

# Lanczos eigensolver parameters (for the local effective Hamiltonian)
krylov_dim = 4       # Lanczos vectors per iteration (small is fine for ground state)
max_iter   = 100     # maximum Lanczos restarts

# ═══════════════════════════════════════════════════════════════════════════
# INITIAL STATE
# ═══════════════════════════════════════════════════════════════════════════

# Choose the initial MPS for DMRG.
# For the Ising phase (Delta > 1), the Neel state is a good starting point.
# For the XY phase (Delta < 1), a random state may converge faster.
#   :neel   — |↑↓↑↓...⟩ product state (bond dimension 1)
#   :random — random MPS with bond dimension = min(chi_max, 8)
init_state = :neel

# ═══════════════════════════════════════════════════════════════════════════
# BUILD HAMILTONIAN MPO
# ═══════════════════════════════════════════════════════════════════════════

println("="^70)
println("  DMRG ground state search")
println("="^70)
println("\nModel: N=$N, J=$J, Delta=$Delta")
println("Field: h = ", length(h) <= 10 ? "$h" : "[$(h[1]), $(h[2]), ..., $(h[end])]")
println("DMRG:  chi_max=$chi_max, n_sweeps=$n_sweeps, cutoff=$cutoff")
println("Lanczos: krylov_dim=$krylov_dim, max_iter=$max_iter")
println("Init:  $init_state\n")

mpo = build_xxz_mpo(N, J, Delta, h; d=d)

# ═══════════════════════════════════════════════════════════════════════════
# BUILD INITIAL MPS
# ═══════════════════════════════════════════════════════════════════════════

sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]

if init_state == :neel
    # (:Z, 1) = up (+1/2),  (:Z, 2) = down (-1/2)
    labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
    mps = product_state(sites, labels)
elseif init_state == :random
    mps = random_state(sites, min(chi_max, 8))
else
    error("Unknown init_state: $init_state. Use :neel or :random.")
end

# ═══════════════════════════════════════════════════════════════════════════
# RUN DMRG
# ═══════════════════════════════════════════════════════════════════════════

state = MPSState(mps, mpo; center=1)
solver = LanczosSolver(krylov_dim, max_iter)
schedule = SweepSchedule(chi_max, n_sweeps; chi_min=chi_min, cutoff_final=cutoff)

println("─"^70)
println("  Sweep   chi    cutoff         energy")
println("─"^70)

E_dmrg = 0.0
for sweep in 1:schedule.n_sweeps
    global E_dmrg
    opts = DMRGOptions(schedule.maxdims[sweep], schedule.cutoffs[sweep], d)
    dir = isodd(sweep) ? :right : :left
    E_dmrg = dmrg_sweep(state, solver, opts, dir)
    println("  $(lpad(sweep, 3))     $(lpad(schedule.maxdims[sweep], 4))   $(schedule.cutoffs[sweep])   $E_dmrg")
end

println("─"^70)
println("  Final ground state energy: $E_dmrg")
println("  Final norm: $(measure_norm(state.mps))")

# ═══════════════════════════════════════════════════════════════════════════
# MEASUREMENTS
# ═══════════════════════════════════════════════════════════════════════════

ops = spin_ops(d)
Sz = Matrix{ComplexF64}(ops[:Z])

# ── Local Sz profile ─────────────────────────────────────────────────────

println("\n" * "─"^70)
println("  Local ⟨Sz_i⟩ profile")
println("─"^70)

sz_profile = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
for i in 1:N
    bar = repeat("█", max(0, round(Int, (sz_profile[i] + 0.5) * 40)))
    println("  Site $(lpad(i, 3)):  $(lpad(round(sz_profile[i], digits=8), 12))  $bar")
end

# ── Staggered magnetization ──────────────────────────────────────────────

m_stag = sum((-1.0)^i * sz_profile[i] for i in 1:N) / N
println("\n  Staggered magnetization: M_stag = $m_stag")

# ── Bond entanglement entropies ──────────────────────────────────────────

println("\n" * "─"^70)
println("  Bond entanglement entropy S(i, i+1)")
println("─"^70)

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

# ── Sz-Sz correlations (from site 1) ─────────────────────────────────────

println("\n" * "─"^70)
println("  Sz-Sz correlations ⟨Sz_1 Sz_j⟩")
println("─"^70)

for j in 1:N
    if j == 1
        c = real(measure_local_observable(state.mps, Sz * Sz, 1))
    else
        c = real(measure_correlation(state.mps, Sz, 1, Sz, j))
    end
    println("  C(1, $(lpad(j, 3))) = $(round(c, digits=10))")
end

println("\n" * "="^70)
println("  Done.")
println("="^70)
