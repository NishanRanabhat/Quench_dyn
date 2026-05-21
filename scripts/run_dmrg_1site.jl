# scripts/run_dmrg_1site.jl
#
# Ground state of the 1D XXZ chain via 2-site DMRG warmup + 1-site DMRG
# refinement.
#
# Why this protocol:
#   1-site DMRG is bond-preserving (QR/LQ only, no SVD/truncation), so it
#   cannot grow χ from a product state. 2-site DMRG handles the χ-growth
#   phase via SVD; once at working χ, 1-site refines the energy without
#   truncation noise.

using LinearAlgebra
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", string(Sys.CPU_THREADS))))

# ═══════════════════════════════════════════════════════════════════════════
# MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

N     = 32
J     = 1.0
Delta = 0.5
d     = 2

h = zeros(N)

# ═══════════════════════════════════════════════════════════════════════════
# DMRG PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

chi_max     = 64
n_warmup    = 20      # 2-site sweeps (grow χ, truncate at cutoff)
n_refine    = 10      # 1-site sweeps (no truncation, refine within fixed χ)
cutoff      = 1e-8

krylov_dim  = 4
max_iter    = 14

init_state  = :neel

# ═══════════════════════════════════════════════════════════════════════════
# BUILD HAMILTONIAN AND INITIAL MPS
# ═══════════════════════════════════════════════════════════════════════════

println("="^70)
println("  DMRG ground state (2-site warmup + 1-site refinement)")
println("="^70)
println("\nModel: N=$N, J=$J, Delta=$Delta")
println("Field: h = ", length(h) <= 10 ? "$h" : "[$(h[1]), $(h[2]), ..., $(h[end])]")
println("DMRG:  chi_max=$chi_max, n_warmup=$n_warmup, n_refine=$n_refine, cutoff=$cutoff")
println("Lanczos: krylov_dim=$krylov_dim, max_iter=$max_iter")
println("Init:  $init_state")
println("BLAS threads: $(BLAS.get_num_threads())\n")

mpo = build_xxz_mpo(N, J, Delta, h; d=d)

sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
if init_state == :neel
    labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
    mps = product_state(sites, labels)
elseif init_state == :random
    mps = random_state(sites, min(chi_max, 8))
else
    error("Unknown init_state: $init_state. Use :neel or :random.")
end

state = MPSState(mps, mpo; center=1)
solver = LanczosSolver(krylov_dim, max_iter)
opts = DMRGOptions(chi_max, cutoff, d)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: 2-site warmup (grows χ from 1 to chi_max)
# ═══════════════════════════════════════════════════════════════════════════

println("─"^70)
println("  Phase 1: 2-site warmup ($n_warmup sweeps)")
println("─"^70)
println("  Sweep   max_χ   max_trunc    total_trunc       energy")
println("─"^70)

E_warmup = 0.0
for sweep in 1:n_warmup
    global E_warmup
    dir = isodd(sweep) ? :right : :left
    res = dmrg_sweep(state, solver, opts, dir)
    E_warmup = res.E
    println("  $(lpad(sweep, 3))     $(lpad(res.max_chi, 4))   $(lpad(round(res.max_trunc, sigdigits=3), 10))   $(lpad(round(res.total_trunc, sigdigits=3), 10))    $E_warmup")
end

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: 1-site refinement (no truncation, χ fixed at warmup value)
# ═══════════════════════════════════════════════════════════════════════════

println("\n─"^1 * "─"^69)
println("  Phase 2: 1-site refinement ($n_refine sweeps, χ fixed)")
println("─"^70)
println("  Sweep   max_χ         energy                   dE_from_warmup")
println("─"^70)

E_dmrg = E_warmup
for sweep in 1:n_refine
    global E_dmrg
    # Pick direction from current center, so this is robust to even/odd
    # warmup count.
    dir = state.center == 1 ? :right : :left
    res = dmrg_sweep_one_site(state, solver, opts, dir)
    E_dmrg = res.E
    dE = E_dmrg - E_warmup
    println("  $(lpad(sweep, 3))     $(lpad(res.max_chi, 4))     $E_dmrg     $(round(dE, sigdigits=3))")
end

var_E = energy_variance(state)
println("─"^70)
println("  Final ground state energy: $E_dmrg")
println("  Final norm: $(measure_norm(state.mps))")
println("  Energy variance ⟨H²⟩ − ⟨H⟩²: $var_E   (zero at exact eigenstate)")
println("  Warmup → refine energy drop: $(round(E_dmrg - E_warmup, sigdigits=4))")

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

# ── Sz-Sz correlations ──────────────────────────────────────────────────

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
