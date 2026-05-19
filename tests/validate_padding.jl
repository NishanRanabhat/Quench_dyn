# tests/validate_padding.jl
#
# Validate the pad_mps utility:
#   TEST 1: Pad a random MPS, check bond dims, norm, observables preserved.
#   TEST 2: End-to-end workflow — DMRG at χ_DMRG, pad to χ_TDVP, run one-site
#           TDVP, compare to ED. Uses N=8 so χ_TDVP=16 (= natural max for the
#           middle bond) gives the full Hilbert manifold → dynamics is exact
#           up to Krylov tolerance.

using LinearAlgebra
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

Random.seed!(0xC0DE_BEEF)   # reproducibility for both random_state and pad_mps noise

println("="^70)
println("  VALIDATION: pad_mps utility + DMRG → pad → 1-site TDVP workflow")
println("="^70)

# ══════════════════════════════════════════════════════════════════════════
# TEST 1: padding correctness on a random MPS
# ══════════════════════════════════════════════════════════════════════════
println("\n" * "─"^70)
println("  TEST 1: pad_mps on a Néel product state  (N=8, χ: 1 → 16)")
println("─"^70)

N = 8
d = 2
chi_target = 16    # = natural max for the middle bond of N=8

sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels_neel = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps_src = product_state(sites, labels_neel)

# Already normalized (product state, bond 1, single coefficient = 1).
# Pad to target.
mps_padded = pad_mps(mps_src, chi_target; noise=1e-6)

# Check 1a: bond dimensions hit the position-aware ceiling (starting from
# chi=1, no over-parameterization, so padded = natural max everywhere).
expected_bonds = [min(chi_target, d^min(i, N-i)) for i in 1:N-1]
actual_bonds   = [size(mps_padded.tensors[i], 3)  for i in 1:N-1]
bonds_pass = expected_bonds == actual_bonds

# Check 1b: norm is exactly 1.
norm_padded = real(measure_norm(mps_padded))
norm_pass = abs(norm_padded - 1.0) < 1e-12

# Check 1c: local observables match the source (perturbation ~ noise² ~ 1e-12).
ops = spin_ops(d)
Sz = Matrix{ComplexF64}(ops[:Z])
sz_src    = [real(measure_local_observable(mps_src,    Sz, i)) for i in 1:N]
sz_padded = [real(measure_local_observable(mps_padded, Sz, i)) for i in 1:N]
max_obs_diff = maximum(abs.(sz_src .- sz_padded))
obs_pass = max_obs_diff < 1e-10

println("  Expected bonds:  $expected_bonds")
println("  Actual bonds:    $actual_bonds")
println("  Bonds match:     $bonds_pass")
println("  ‖padded‖² = $(round(norm_padded, digits=14))   (target 1.0)   pass: $norm_pass")
println("  max|⟨Sz_i⟩_padded − ⟨Sz_i⟩_src| = $(round(max_obs_diff, sigdigits=4))   pass: $obs_pass")

test1_pass = bonds_pass && norm_pass && obs_pass
println("\n  TEST 1: $(test1_pass ? "PASS" : "FAIL")")

# ══════════════════════════════════════════════════════════════════════════
# TEST 2: DMRG (chi=4) → pad to chi=16 → 1-site TDVP → compare to ED
# ══════════════════════════════════════════════════════════════════════════
println("\n" * "─"^70)
println("  TEST 2: DMRG → pad → 1-site TDVP   (N=$N, χ_DMRG=4, χ_TDVP=16)")
println("─"^70)

J          = 1.0
Delta_pre  = 2.0       # Ising-like → ground state has modest entanglement
h_pre      = zeros(N)
Delta_post = 0.5       # XY-dominated → state spreads
h_post     = [0.1, -0.05, 0.15, -0.1, 0.05, -0.15, 0.1, -0.05]

chi_DMRG = 4
chi_TDVP = 16   # = natural max for middle bond → full manifold

dt        = 0.05
t_max     = 1.0
n_steps   = round(Int, t_max / dt)
tol       = 1e-8

mpo_pre   = build_xxz_mpo(N, J, Delta_pre,  h_pre)
mpo_post  = build_xxz_mpo(N, J, Delta_post, h_post)
H_ed_pre  = build_xxz_hamiltonian(N, J, Delta_pre,  h_pre)
H_ed_post = build_xxz_hamiltonian(N, J, Delta_post, h_post)
eig_post  = diagonalize(H_ed_post)

labels   = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps_neel = product_state(sites, labels)

# ── DMRG ──────────────────────────────────────────────────────────────────
state_dmrg = MPSState(mps_neel, mpo_pre; center=1)
solver_dmrg = LanczosSolver(4, 100)
opts_dmrg = DMRGOptions(chi_DMRG, 1e-12, d)

E_dmrg = 0.0
for sweep in 1:10
    global E_dmrg
    dir = isodd(sweep) ? :right : :left
    E_dmrg = dmrg_sweep(state_dmrg, solver_dmrg, opts_dmrg, dir).E
end

# Normalize DMRG state.
nrm_sq = real(measure_norm(state_dmrg.mps))
state_dmrg.mps.tensors[state_dmrg.center] ./= sqrt(nrm_sq)

pre_pad_bonds = [size(state_dmrg.mps.tensors[i], 3) for i in 1:N-1]
E_pre_dmrg_check = real(dot(mps_to_vector(state_dmrg.mps),
                            H_ed_pre * mps_to_vector(state_dmrg.mps)))

# ── Pad to chi_TDVP ───────────────────────────────────────────────────────
mps_padded = pad_mps(state_dmrg.mps, chi_TDVP; noise=1e-6)
post_pad_bonds = [size(mps_padded.tensors[i], 3) for i in 1:N-1]

# Energy under H_pre after padding — must match DMRG energy to O(noise²).
psi_padded_vec = mps_to_vector(mps_padded)
psi_padded_vec ./= norm(psi_padded_vec)
E_pre_padded_check = real(dot(psi_padded_vec, H_ed_pre * psi_padded_vec))

println("  E_DMRG (χ=$chi_DMRG)                = $(round(E_dmrg,            digits=12))")
println("  ⟨ψ_DMRG  |H_pre|ψ_DMRG ⟩ via vec   = $(round(E_pre_dmrg_check,   digits=12))")
println("  ⟨ψ_padded|H_pre|ψ_padded⟩ via vec  = $(round(E_pre_padded_check, digits=12))")
println("  Bonds before padding: $pre_pad_bonds")
println("  Bonds after  padding: $post_pad_bonds")

E_pre_shift = abs(E_pre_padded_check - E_pre_dmrg_check)
energy_pass = E_pre_shift < 1e-8
println("  |ΔE| after padding = $(round(E_pre_shift, sigdigits=4))   (must be < 1e-8)  pass: $energy_pass")

# ── 1-site TDVP under H_post, ED reference uses the padded vector ────────
state_tdvp  = MPSState(mps_padded, mpo_post; center=1)
solver_tdvp = KrylovExponential(30, 1e-12, "real")
opts_tdvp   = TDVPOptions(dt, chi_TDVP, 1e-12, d)

# Re-fetch the post-canonicalization vector so ED and TDVP truly start equal.
psi0_vec = mps_to_vector(state_tdvp.mps)
psi0_vec ./= norm(psi0_vec)

Sz_real = Matrix{Float64}(real.(ops[:Z]))
max_err_obs = 0.0
max_infid   = 0.0
tdvp_pass   = true

println("\n  step    t       max|Sz_ED − Sz_TDVP|    norm_mps        infidelity")
for step in 1:n_steps
    global max_err_obs, max_infid, tdvp_pass
    t = step * dt

    psi_ed_t = ed_time_evolve(eig_post, psi0_vec, t)

    tdvp_sweep_one_site(state_tdvp, solver_tdvp, opts_tdvp, :right)
    tdvp_sweep_one_site(state_tdvp, solver_tdvp, opts_tdvp, :left)

    if step % 4 == 0 || step == n_steps
        sz_ed    = ed_local_profile(psi_ed_t, Sz_real, N)
        sz_tdvp  = [real(measure_local_observable(state_tdvp.mps, Sz, i)) for i in 1:N]
        norm_mps = real(measure_norm(state_tdvp.mps))

        psi_tdvp_vec = mps_to_vector(state_tdvp.mps)
        overlap    = abs(dot(psi_ed_t, psi_tdvp_vec))^2
        infidelity = 1.0 - overlap

        max_err = maximum(abs.(sz_ed .- sz_tdvp))
        max_err_obs = max(max_err_obs, max_err)
        max_infid   = max(max_infid, infidelity)

        status = (max_err < tol && infidelity < tol) ? " " : "!"
        println("  $status $(lpad(step, 4))    $(round(t, digits=3))    $(round(max_err, sigdigits=4))                 $(round(norm_mps, digits=8))    $(round(infidelity, sigdigits=4))")

        if max_err >= tol || infidelity >= tol
            tdvp_pass = false
        end
    end
end

test2_pass = energy_pass && tdvp_pass
println("\n  TEST 2: $(test2_pass ? "PASS" : "FAIL")")
println("  Max Sz error during evolution: $(round(max_err_obs, sigdigits=4))")
println("  Max infidelity:                 $(round(max_infid,   sigdigits=4))")

# ── Summary ──────────────────────────────────────────────────────────────
println("\n" * "="^70)
if test1_pass && test2_pass
    println("  PADDING VALIDATION PASSED")
else
    println("  PADDING VALIDATION FAILED")
end
println("="^70)
