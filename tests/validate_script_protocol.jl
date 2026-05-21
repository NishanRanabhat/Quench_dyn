# tests/validate_script_protocol.jl
#
# Validate the exact protocol used by scripts/run_tdvp_1site.jl against ED,
# at the same physics (XXZ quench Delta_i=0.5 → Delta_f=3.0 with staggered
# field), reduced to a size where ED is feasible.
#
# N=12 → ED dim = 4096, middle-bond natural max χ = 2^6 = 64.
# With chi_work=64 the MPS manifold IS the full Hilbert space, so 1-site
# TDVP gives exact dynamics up to Krylov tolerance. Any disagreement
# with ED is a bug in the protocol code path.

using LinearAlgebra
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

Random.seed!(0xBEEF_F00D)

println("="^70)
println("  VALIDATION: scripts/run_tdvp_1site.jl protocol vs ED")
println("="^70)

# ── Same physics as the script ──────────────────────────────────────────
N = 12
d = 2
J = 1.0
Delta_i = 0.5
Delta_f = 3.0
h = [0.1 * (-1.0)^i for i in 1:N]

# ── Reduced to ED-feasible size; chi_work = full manifold for middle bond
chi_dmrg = 16
chi_work = 64       # = min(d^(N/2), d^(N/2)) = full manifold
pad_noise = 1e-6

n_dmrg_sweeps = 20
dt = 0.05
t_max = 1.0
n_steps = round(Int, t_max / dt)

tol_pad  = 1e-8
tol_tdvp = 1e-8

println("\nModel: N=$N, J=$J, Delta_i=$Delta_i → Delta_f=$Delta_f")
println("Field: h = $h")
println("chi_dmrg = $chi_dmrg, chi_work = $chi_work (full manifold = $(d^(div(N,2))))")
println("Padding noise = $pad_noise")
println("Time: dt=$dt, t_max=$t_max, n_steps=$n_steps")

# ── Build MPOs and ED Hamiltonians ──────────────────────────────────────
mpo_pre  = build_xxz_mpo(N, J, Delta_i, h)
mpo_post = build_xxz_mpo(N, J, Delta_f, h)

H_ed_pre  = build_xxz_hamiltonian(N, J, Delta_i, h)
H_ed_post = build_xxz_hamiltonian(N, J, Delta_f, h)

E_ed_pre, _ = ground_state(H_ed_pre)
eig_post = diagonalize(H_ed_post)

ops = spin_ops(d)
Sz = Matrix{ComplexF64}(ops[:Z])
Sz_real = Matrix{Float64}(real.(ops[:Z]))

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: 2-site DMRG (Néel init, like the script)
# ═══════════════════════════════════════════════════════════════════════════
println("\n" * "─"^70)
println("  Step 1: 2-site DMRG ground state at Delta_i=$Delta_i, chi_dmrg=$chi_dmrg")
println("─"^70)

sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps_init = product_state(sites, labels)
state = MPSState(mps_init, mpo_pre; center=1)

solver_dmrg = LanczosSolver(4, 100)
opts_dmrg = DMRGOptions(chi_dmrg, 1e-12, d)

E_dmrg = 0.0
for sweep in 1:n_dmrg_sweeps
    global E_dmrg
    dir = isodd(sweep) ? :right : :left
    E_dmrg = dmrg_sweep(state, solver_dmrg, opts_dmrg, dir).E
end

nrm = real(measure_norm(state.mps))
state.mps.tensors[state.center] ./= sqrt(nrm)

dmrg_dE = abs(E_dmrg - E_ed_pre)
println("  E_ED   = $(round(E_ed_pre, digits=12))")
println("  E_DMRG = $(round(E_dmrg,   digits=12))")
println("  |ΔE|   = $(round(dmrg_dE, sigdigits=4))  (DMRG truncation at chi_dmrg=$chi_dmrg)")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Pad chi_dmrg → chi_work
# ═══════════════════════════════════════════════════════════════════════════
println("\n" * "─"^70)
println("  Step 2: Pad MPS chi_dmrg=$chi_dmrg → chi_work=$chi_work (noise=$pad_noise)")
println("─"^70)

bonds_pre = [size(state.mps.tensors[i], 3) for i in 1:N-1]
padded = pad_mps(state.mps, chi_work; noise=pad_noise)
bonds_post = [size(padded.tensors[i], 3) for i in 1:N-1]

state_pad_check = MPSState(padded, mpo_pre; center=1)
E_padded = measure_energy(state_pad_check)
pad_dE = abs(E_padded - E_dmrg)

println("  Bond profile: $bonds_pre → $bonds_post")
println("  E_DMRG               = $(round(E_dmrg,   digits=12))")
println("  ⟨ψ_pad|H_pre|ψ_pad⟩   = $(round(E_padded, digits=12))")
println("  |ΔE|                  = $(round(pad_dE,  sigdigits=4))  (expected O(noise²)=$(pad_noise^2))")

pad_pass = pad_dE < tol_pad
println("  [$(pad_pass ? "PASS" : "FAIL")] padding perturbation < tol_pad=$tol_pad")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: 1-site TDVP under H_post vs ED, same initial state
# ═══════════════════════════════════════════════════════════════════════════
println("\n" * "─"^70)
println("  Step 3: 1-site TDVP vs ED time evolution (same psi0 = padded MPS)")
println("─"^70)

state = MPSState(padded, mpo_post; center=1)
psi0_vec = mps_to_vector(state.mps)
psi0_vec ./= norm(psi0_vec)

E0_tdvp = measure_energy(state)
E0_ed   = real(dot(psi0_vec, H_ed_post * psi0_vec))
println("  E0 (TDVP measure)         = $(round(E0_tdvp, digits=12))")
println("  E0 (ED ⟨psi0|H_post|psi0⟩) = $(round(E0_ed,   digits=12))")
println("  |ΔE0|                      = $(round(abs(E0_tdvp - E0_ed), sigdigits=4))")

solver_tdvp = KrylovExponential(30, 1e-12, "real")
opts_tdvp = TDVPOptions(dt, chi_work, 1e-12, d)

println("\n  step    t      max|Sz_ED − Sz_TDVP|    norm_mps     1-fidelity")
println("  " * "─"^75)

max_err_obs = 0.0
max_infid = 0.0
max_E_drift = 0.0
tdvp_pass = true

for step in 1:n_steps
    global max_err_obs, max_infid, max_E_drift, tdvp_pass
    t = step * dt

    psi_ed_t = ed_time_evolve(eig_post, psi0_vec, t)

    tdvp_sweep_one_site(state, solver_tdvp, opts_tdvp, :right)
    tdvp_sweep_one_site(state, solver_tdvp, opts_tdvp, :left)

    if step % 2 == 0 || step == n_steps
        sz_ed   = ed_local_profile(psi_ed_t, Sz_real, N)
        sz_tdvp = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
        norm_mps = real(measure_norm(state.mps))
        E_tdvp = measure_energy(state)

        psi_tdvp_vec = mps_to_vector(state.mps)
        overlap = abs(dot(psi_ed_t, psi_tdvp_vec))^2
        infidelity = 1.0 - overlap

        max_err = maximum(abs.(sz_ed .- sz_tdvp))
        max_err_obs = max(max_err_obs, max_err)
        max_infid   = max(max_infid, infidelity)
        max_E_drift = max(max_E_drift, abs(E_tdvp - E0_tdvp))

        status = (max_err < tol_tdvp && infidelity < tol_tdvp) ? " " : "!"
        println("  $status $(lpad(step,4))   $(lpad(round(t, digits=3),5))    $(lpad(round(max_err, sigdigits=4),18))   $(round(norm_mps, digits=10))    $(round(infidelity, sigdigits=4))")

        if max_err >= tol_tdvp || infidelity >= tol_tdvp
            tdvp_pass = false
        end
    end
end

# ── Summary ──────────────────────────────────────────────────────────
println("\n" * "="^70)
println("  SUMMARY (script protocol at N=$N, chi_work=$chi_work=full manifold)")
println("="^70)
println("  Padding sanity   |ΔE| = $(round(pad_dE,      sigdigits=4))   tol $tol_pad   [$(pad_pass  ? "PASS" : "FAIL")]")
println("  TDVP vs ED      max|ΔSz|  = $(round(max_err_obs, sigdigits=4))   tol $tol_tdvp  [$(tdvp_pass ? "PASS" : "FAIL")]")
println("                  max infid = $(round(max_infid,   sigdigits=4))")
println("                  max E drift = $(round(max_E_drift, sigdigits=4))")
println()
overall = pad_pass && tdvp_pass
println("  $(overall ? "PROTOCOL VALIDATED AGAINST ED" : "PROTOCOL FAILED ED COMPARISON")")
println("="^70)
