# tests/validate_tdvp.jl
#
# Validate TDVP time evolution against exact ED evolution.
# Start from the same product state, evolve under the same Hamiltonian,
# compare local <Sz_i> at each time step.

using LinearAlgebra

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

println("="^70)
println("  VALIDATION: TDVP vs ED time evolution")
println("="^70)

# ── Parameters ───────────────────────────────────────────────────────────

N = 6
d = 2
J = 1.0
Delta = 1.5
h = [0.1, -0.2, 0.15, -0.05, 0.3, -0.1]

# time evolution
dt = 0.05
t_max = 1.0
n_steps = round(Int, t_max / dt)

# TDVP solver params
krylov_dim = 30
krylov_tol = 1e-12
chi_max = 64
cutoff = 1e-12

println("\nModel: XXZ chain, N=$N, J=$J, Delta=$Delta")
println("Field: h = $h")
println("Time:  dt=$dt, t_max=$t_max, n_steps=$n_steps")
println("TDVP:  krylov_dim=$krylov_dim, chi_max=$chi_max, cutoff=$cutoff")

# ── Build Hamiltonians ───────────────────────────────────────────────────

mpo = build_xxz_mpo(N, J, Delta, h)
H_ed = build_xxz_hamiltonian(N, J, Delta, h)
eig = diagonalize(H_ed)

ops = spin_ops(d)
Sz = Matrix{ComplexF64}(ops[:Z])
Sz_real = Matrix{Float64}(real.(ops[:Z]))

# ── Initial state: Neel |↑↓↑↓↑↓⟩ ────────────────────────────────────────

# ED state vector
psi0_ed = neel_state(N; start_up=true)

# MPS product state
sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]   # 1=up, 2=down
mps = product_state(sites, labels)

# ── Verify initial states match ──────────────────────────────────────────

println("\n" * "─"^70)
println("  Initial state check: <Sz_i> from ED vs MPS")
println("─"^70)

sz_ed_init = ed_local_profile(psi0_ed, Sz_real, N)
sz_mps_init = [real(measure_local_observable(mps, Sz, i)) for i in 1:N]

println("  Site   ED         MPS        diff")
for i in 1:N
    diff = abs(sz_ed_init[i] - sz_mps_init[i])
    println("  $i      $(round(sz_ed_init[i], digits=6))   $(round(sz_mps_init[i], digits=6))   $diff")
end

init_match = maximum(abs.(sz_ed_init .- sz_mps_init)) < 1e-12
println("  Initial states match: $init_match")

if !init_match
    println("  >> FATAL: Initial states do not match. Aborting.")
    exit(1)
end

# ── Set up TDVP ──────────────────────────────────────────────────────────

state = MPSState(mps, mpo; center=1)
solver = KrylovExponential(krylov_dim, krylov_tol, "real")
options = TDVPOptions(dt, chi_max, cutoff, d)

# ── Evolve and compare ───────────────────────────────────────────────────

println("\n" * "─"^70)
println("  Time evolution: TDVP vs ED")
println("─"^70)
println("  step   t        max|Sz_ed - Sz_tdvp|   norm_mps")

psi_ed = copy(psi0_ed)
max_error_all = 0.0
all_pass = true
tol = 1e-4

for step in 1:n_steps
    global psi_ed, max_error_all, all_pass
    t = step * dt

    # ED: exact evolution from initial state
    psi_ed = ed_time_evolve(eig, psi0_ed, t)

    # TDVP: one full step = right sweep + left sweep
    tdvp_sweep(state, solver, options, :right)
    tdvp_sweep(state, solver, options, :left)

    # measure every 5 steps to keep output readable
    if step % 5 == 0 || step == n_steps
        sz_ed = ed_local_profile(psi_ed, Sz_real, N)
        sz_tdvp = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
        norm_mps = measure_norm(state.mps)

        max_err = maximum(abs.(sz_ed .- sz_tdvp))
        max_error_all = max(max_error_all, max_err)

        status = max_err < tol ? " " : "!"
        println("  $status $(lpad(step, 4))   $(round(t, digits=3))    $(round(max_err, sigdigits=4))                $(round(real(norm_mps), digits=8))")

        if max_err >= tol
            all_pass = false
        end
    end
end

# ── Final detailed comparison ────────────────────────────────────────────

println("\n" * "─"^70)
println("  Final state comparison at t=$(t_max)")
println("─"^70)

psi_ed_final = ed_time_evolve(eig, psi0_ed, t_max)
sz_ed_final = ed_local_profile(psi_ed_final, Sz_real, N)
sz_tdvp_final = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]

println("  Site   Sz_ED           Sz_TDVP         diff")
for i in 1:N
    diff = abs(sz_ed_final[i] - sz_tdvp_final[i])
    println("  $i      $(round(sz_ed_final[i], digits=10))   $(round(sz_tdvp_final[i], digits=10))   $(round(diff, sigdigits=4))")
end

# ── State overlap: |<psi_ED|psi_TDVP>|^2 ─────────────────────────────────

println("\n" * "─"^70)
println("  State overlap (fidelity) at t=$(t_max)")
println("─"^70)

psi_tdvp_vec = mps_to_vector(state.mps)
overlap = abs(dot(psi_ed_final, psi_tdvp_vec))^2
infidelity = 1.0 - overlap

println("  |<psi_ED|psi_TDVP>|^2 = $(round(overlap, digits=12))")
println("  Infidelity             = $(round(infidelity, sigdigits=4))")

# ── Summary ──────────────────────────────────────────────────────────────

println("\n" * "="^70)
println("  Max Sz error across all times: $(round(max_error_all, sigdigits=4))")
println("  State fidelity at t=$t_max:    $(round(overlap, digits=10))")
println("  Tolerance: $tol")
if all_pass && overlap > 1.0 - tol
    println("  TDVP VALIDATION PASSED")
else
    println("  TDVP VALIDATION FAILED")
end
println("="^70)
