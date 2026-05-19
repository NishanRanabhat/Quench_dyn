# tests/validate_dmrg_one_site.jl
#
# Validate one-site DMRG against ED. One-site DMRG cannot grow χ from a
# product state, so the standard recipe is two-site warmup → one-site
# refinement. The test checks:
#   (a) one-site does not raise the variational energy (within fp noise),
#   (b) one-site energy matches ED.
#
# With chi_work = 8 (natural max rank for N=6, d=2), the MPS manifold is the
# full Hilbert space, so DMRG converges to the exact ED ground state.

using LinearAlgebra

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

println("="^70)
println("  VALIDATION: One-site DMRG vs ED (after two-site warmup)")
println("="^70)

N = 6
d = 2
chi_work = 8       # natural max rank for N=6, d=2

n_warmup = 8       # two-site sweeps (even → ends with center=1)
n_refine = 8       # one-site sweeps (even → ends with center=1)

tol_ed   = 1e-10
tol_drop = 1e-10   # one-site refinement must not raise E beyond fp noise

test_params = [
    (J=1.0, Delta=3.0, h=zeros(N),
        label="Deep Ising"),
    (J=1.0, Delta=0.3, h=[0.1, -0.2, 0.15, -0.05, 0.3, -0.1],
        label="XY + random field"),
    (J=1.0, Delta=1.0, h=[0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
        label="Heisenberg + staggered field"),
]

all_pass = true

for (idx, p) in enumerate(test_params)
    global all_pass
    println("\n" * "─"^70)
    println("  Set $idx: $(p.label)")
    println("         J=$(p.J), Delta=$(p.Delta), h=$(p.h)")
    println("─"^70)

    # ── ED reference ──────────────────────────────────────────────────
    H_ed = build_xxz_hamiltonian(N, p.J, p.Delta, p.h)
    E_ed, _ = ground_state(H_ed)

    # ── Set up DMRG from Néel ─────────────────────────────────────────
    mpo = build_xxz_mpo(N, p.J, p.Delta, p.h)
    sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
    labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
    mps = product_state(sites, labels)

    state = MPSState(mps, mpo; center=1)
    solver = LanczosSolver(4, 100)
    opts = DMRGOptions(chi_work, 1e-12, d)

    # ── Two-site warmup ───────────────────────────────────────────────
    E_warmup = 0.0
    for sweep in 1:n_warmup
        dir = isodd(sweep) ? :right : :left
        E_warmup = dmrg_sweep(state, solver, opts, dir).E
    end
    diff_warmup = abs(E_warmup - E_ed)

    # ── One-site refinement ───────────────────────────────────────────
    E_refine = 0.0
    for sweep in 1:n_refine
        dir = isodd(sweep) ? :right : :left
        E_refine = dmrg_sweep_one_site(state, solver, opts, dir).E
    end
    diff_refine = abs(E_refine - E_ed)

    # ── Checks ────────────────────────────────────────────────────────
    delta_refine = E_refine - E_warmup        # must be ≤ tol_drop (variational)

    pass_var = delta_refine <= tol_drop
    pass_ed  = diff_refine < tol_ed
    pass     = pass_var && pass_ed

    println("  E_ED              = $(round(E_ed,     digits=12))")
    println("  E_DMRG  (2-site)  = $(round(E_warmup, digits=12))   |Δ_ED| = $(round(diff_warmup, sigdigits=4))")
    println("  E_DMRG  (1-site)  = $(round(E_refine, digits=12))   |Δ_ED| = $(round(diff_refine, sigdigits=4))")
    println("  E(1-site) - E(2-site) = $(round(delta_refine, sigdigits=4))   (must be ≤ $tol_drop)")
    println("  [$(pass ? "PASS" : "FAIL")] variational=$(pass_var)   matches ED=$(pass_ed)")

    if !pass
        all_pass = false
    end
end

println("\n" * "="^70)
if all_pass
    println("  ONE-SITE DMRG VALIDATION PASSED")
else
    println("  ONE-SITE DMRG VALIDATION FAILED")
end
println("="^70)
