# tests/validate_mpo_and_gs.jl
#
# Two-stage validation:
#   1) N=4: build XXZ Hamiltonian as dense matrix (ED) and as MPO,
#      contract MPO to dense, compare element-by-element.
#   2) N=6: compare ground state energy from DMRG vs ED for 3 parameter sets.

using LinearAlgebra

# ── Load module ──────────────────────────────────────────────────────────
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

println("="^70)
println("  VALIDATION: MPO vs ED Hamiltonian + DMRG vs ED ground state")
println("="^70)


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: MPO ↔ ED Hamiltonian consistency (N=4)
# ══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  TEST 1: MPO vs ED dense Hamiltonian (N=4)")
println("─"^70)

N_test1 = 4
d = 2

test_params = [
    (J=1.0, Delta=2.0,  h=[0.0, 0.0, 0.0, 0.0],            label="Ising-like, no field"),
    (J=1.0, Delta=0.5,  h=[0.3, -0.1, 0.2, -0.4],          label="XY-like, random field"),
    (J=1.3, Delta=1.0,  h=[0.5, 0.5, 0.5, 0.5],            label="Heisenberg + uniform field"),
]

all_pass_test1 = true

for (idx, p) in enumerate(test_params)
    # ED dense Hamiltonian
    H_ed = build_xxz_hamiltonian(N_test1, p.J, p.Delta, p.h)

    # MPO → dense
    mpo = build_xxz_mpo(N_test1, p.J, p.Delta, p.h)
    H_mpo = mpo_to_matrix(mpo)

    max_diff = maximum(abs.(Matrix(H_ed) .- real.(H_mpo)))
    pass = max_diff < 1e-14

    status = pass ? "PASS" : "FAIL"
    println("  [$status] Set $idx: $(p.label)")
    println("         J=$(p.J), Delta=$(p.Delta), h=$(p.h)")
    println("         max|H_ed - H_mpo| = $max_diff")

    if !pass
        all_pass_test1 = false
    end
end

println()
if all_pass_test1
    println("  >> TEST 1 PASSED: MPO and ED Hamiltonians are identical.")
else
    println("  >> TEST 1 FAILED: Mismatch detected!")
    exit(1)
end

# ══════════════════════════════════════════════════════════════════════════
# TEST 2: DMRG vs ED ground state energy (N=6)
# ══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  TEST 2: DMRG vs ED ground state energy (N=6)")
println("─"^70)

N_test2 = 6

test_params_gs = [
    (J=1.0, Delta=3.0,  h=zeros(N_test2),                         label="Deep Ising phase, no field"),
    (J=1.0, Delta=0.3,  h=[0.1, -0.2, 0.15, -0.05, 0.3, -0.1],   label="XY-dominated, random field"),
    (J=1.0, Delta=1.0,  h=[0.5, -0.5, 0.5, -0.5, 0.5, -0.5],     label="Heisenberg + staggered field"),
]

all_pass_test2 = true
tol_gs = 1e-8

for (idx, p) in enumerate(test_params_gs)
    # ── ED ground state ──
    H_ed = build_xxz_hamiltonian(N_test2, p.J, p.Delta, p.h)
    eig = diagonalize(H_ed)
    E_ed, _ = ground_state(eig)

    # ── DMRG ground state ──
    mpo = build_xxz_mpo(N_test2, p.J, p.Delta, p.h)
    sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N_test2]

    # start from Neel state: (:Z,1)=up, (:Z,2)=down
    labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N_test2]
    mps = product_state(sites, labels)

    state = MPSState(mps, mpo; center=1)
    solver = LanczosSolver(4, 100)   # krylov_dim, max_iter
    schedule = SweepSchedule(32, 20; cutoff_final=1e-12)

    E_dmrg = 0.0
    for sweep in 1:schedule.n_sweeps
        opts = DMRGOptions(schedule.maxdims[sweep], schedule.cutoffs[sweep], d)
        dir = isodd(sweep) ? :right : :left
        E_dmrg = dmrg_sweep(state, solver, opts, dir)
    end

    diff = abs(E_dmrg - E_ed)
    pass = diff < tol_gs

    status = pass ? "PASS" : "FAIL"
    println("  [$status] Set $idx: $(p.label)")
    println("         J=$(p.J), Delta=$(p.Delta), h=$(p.h)")
    println("         E_ed   = $(round(E_ed, digits=12))")
    println("         E_dmrg = $(round(E_dmrg, digits=12))")
    println("         |diff| = $(round(diff, sigdigits=4))")

    if !pass
        all_pass_test2 = false
    end
end

println()
if all_pass_test2
    println("  >> TEST 2 PASSED: DMRG ground state energies match ED.")
else
    println("  >> TEST 2 FAILED: Energy mismatch beyond tolerance $tol_gs!")
end

# ── Summary ──────────────────────────────────────────────────────────────
println("\n" * "="^70)
if all_pass_test1 && all_pass_test2
    println("  ALL TESTS PASSED")
else
    println("  SOME TESTS FAILED")
end
println("="^70)
