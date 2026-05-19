# tests/validate_tdvp_one_site.jl
#
# Validate one-site TDVP against ED for a quench protocol.
#
# Recipe (one-site TDVP cannot grow χ, so we need a non-trivial-χ start):
#   1. Two-site DMRG for the ground state of the pre-quench H.
#   2. Switch MPO to the post-quench H.
#   3. Evolve with one-site TDVP, compare to exact ED evolution of the same
#      initial state vector.
#
# With chi_work = 8 (natural max rank for N=6, d=2), the MPS manifold is the
# full Hilbert space, so one-site TDVP gives exact dynamics up to Krylov
# tolerance. Local Sz observables and the full-state fidelity are checked
# at intervals across the evolution.

using LinearAlgebra

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

println("="^70)
println("  VALIDATION: One-site TDVP vs ED (quench protocol)")
println("="^70)

# ── Parameters ───────────────────────────────────────────────────────────
N = 6
d = 2
J = 1.0

Delta_pre  = 3.0
h_pre      = zeros(N)

Delta_post = 0.3
h_post     = [0.1, -0.2, 0.15, -0.05, 0.3, -0.1]

chi_work = 8       # natural max rank for N=6, d=2

n_dmrg_sweeps = 10
opts_dmrg = DMRGOptions(chi_work, 1e-12, d)
solver_dmrg = LanczosSolver(4, 100)

dt = 0.05
t_max = 1.0
n_steps = round(Int, t_max / dt)
opts_tdvp = TDVPOptions(dt, chi_work, 1e-12, d)
solver_tdvp = KrylovExponential(30, 1e-12, "real")

tol = 1e-8

println("\nModel:        XXZ chain, N=$N, J=$J")
println("Pre-quench:   Delta=$Delta_pre, h=$h_pre")
println("Post-quench:  Delta=$Delta_post, h=$h_post")
println("Working χ:    $chi_work (full manifold for N=$N)")
println("Time:         dt=$dt, t_max=$t_max, n_steps=$n_steps")

# ── Hamiltonians and operators ───────────────────────────────────────────
mpo_pre  = build_xxz_mpo(N, J, Delta_pre,  h_pre)
mpo_post = build_xxz_mpo(N, J, Delta_post, h_post)

H_ed_pre  = build_xxz_hamiltonian(N, J, Delta_pre,  h_pre)
H_ed_post = build_xxz_hamiltonian(N, J, Delta_post, h_post)

E_pre_ed, _ = ground_state(H_ed_pre)
eig_post = diagonalize(H_ed_post)

ops = spin_ops(d)
Sz      = Matrix{ComplexF64}(ops[:Z])
Sz_real = Matrix{Float64}(real.(ops[:Z]))

# ── Step 1: DMRG ground state of pre-quench H ────────────────────────────
println("\n" * "─"^70)
println("  Step 1: DMRG ground state of pre-quench H")
println("─"^70)

sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps_init = product_state(sites, labels)

state = MPSState(mps_init, mpo_pre; center=1)

E_dmrg = 0.0
for sweep in 1:n_dmrg_sweeps
    global E_dmrg
    dir = isodd(sweep) ? :right : :left
    E_dmrg = dmrg_sweep(state, solver_dmrg, opts_dmrg, dir).E
end

println("  E_ED   = $(round(E_pre_ed, digits=12))")
println("  E_DMRG = $(round(E_dmrg,   digits=12))")
println("  |diff| = $(round(abs(E_dmrg - E_pre_ed), sigdigits=4))")

# Normalize MPS explicitly (DMRG with chi=full should leave norm=1, but be safe)
nrm_sq = real(measure_norm(state.mps))
state.mps.tensors[state.center] ./= sqrt(nrm_sq)

# Capture initial state vector for ED reference
psi0_vec = mps_to_vector(state.mps)
psi0_vec ./= norm(psi0_vec)

# Sanity: ⟨ψ|H_pre|ψ⟩ should match DMRG energy
E_psi0_ed = real(dot(psi0_vec, H_ed_pre * psi0_vec))
println("  ⟨ψ_DMRG|H_pre|ψ_DMRG⟩ via ED vector = $(round(E_psi0_ed, digits=12))")

# ── Step 2: Switch MPO to post-quench H, set up state for TDVP ──────────
state = MPSState(state.mps, mpo_post; center=1)

# ── Step 3: 1-site TDVP under post-quench H vs ED ───────────────────────
println("\n" * "─"^70)
println("  Step 2: 1-site TDVP vs ED time evolution")
println("─"^70)
println("  step    t       max|Sz_ED - Sz_TDVP|    norm_mps        infidelity")

max_err_obs = 0.0
max_infid   = 0.0
all_pass    = true

for step in 1:n_steps
    global max_err_obs, max_infid, all_pass
    t = step * dt

    # ED reference
    psi_ed_t = ed_time_evolve(eig_post, psi0_vec, t)

    # 1-site TDVP step: right sweep + left sweep
    tdvp_sweep_one_site(state, solver_tdvp, opts_tdvp, :right)
    tdvp_sweep_one_site(state, solver_tdvp, opts_tdvp, :left)

    if step % 4 == 0 || step == n_steps
        sz_ed   = ed_local_profile(psi_ed_t, Sz_real, N)
        sz_tdvp = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
        norm_mps = real(measure_norm(state.mps))

        psi_tdvp_vec = mps_to_vector(state.mps)
        overlap    = abs(dot(psi_ed_t, psi_tdvp_vec))^2
        infidelity = 1.0 - overlap

        max_err = maximum(abs.(sz_ed .- sz_tdvp))
        max_err_obs = max(max_err_obs, max_err)
        max_infid   = max(max_infid, infidelity)

        status = (max_err < tol && infidelity < tol) ? " " : "!"
        println("  $status $(lpad(step, 4))    $(round(t, digits=3))    $(round(max_err, sigdigits=4))                 $(round(norm_mps, digits=8))    $(round(infidelity, sigdigits=4))")

        if max_err >= tol || infidelity >= tol
            all_pass = false
        end
    end
end

# ── Summary ──────────────────────────────────────────────────────────────
println("\n" * "="^70)
println("  Max Sz error across measured times: $(round(max_err_obs, sigdigits=4))")
println("  Max infidelity:                      $(round(max_infid,   sigdigits=4))")
println("  Tolerance: $tol")
if all_pass
    println("  ONE-SITE TDVP VALIDATION PASSED")
else
    println("  ONE-SITE TDVP VALIDATION FAILED")
end
println("="^70)
