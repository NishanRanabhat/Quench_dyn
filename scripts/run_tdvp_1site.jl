# scripts/run_tdvp_1site.jl
#
# Quench dynamics via 1-site TDVP.
#
# Protocol:
#   1) 2-site DMRG for the ground state of H(Delta_i) at chi_dmrg.
#   2) Pad the MPS up to chi_work using small random noise. The noise
#      seeds amplitude in the new bond directions so 1-site TDVP can
#      rotate into them under H(Delta_f); without it, those directions
#      would be identically zero and quench-driven correlations could
#      not grow there. Padded-state ⟨H_pre⟩ differs from E_gs by
#      O(noise²), which is checked below.
#   3) Switch MPO to H(Delta_f); build fresh environments at chi_work.
#   4) Evolve with 1-site TDVP (bond-preserving QR/LQ, no SVD →
#      exactly unitary on the fixed-χ manifold, so norm and energy are
#      conserved up to Krylov tolerance).
#
# Hamiltonian:
#   H = J Σ_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1})
#     + Δ Σ_i  Sz_i Sz_{i+1}
#     + Σ_i    h_i  Sz_i

using LinearAlgebra
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

#BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", string(Sys.CPU_THREADS))))

# ═══════════════════════════════════════════════════════════════════════════
# MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

N     = 32
J     = 1.0
d     = 2

Delta_i = 0.5
Delta_f = 3.0

h = [0.1 * (-1.0)^i for i in 1:N]

# ═══════════════════════════════════════════════════════════════════════════
# DMRG PARAMETERS (2-site, ground state preparation)
# ═══════════════════════════════════════════════════════════════════════════

chi_dmrg     = 64
n_sweeps     = 30
cutoff_dmrg  = 1e-10

lanczos_krylov_dim = 4
lanczos_max_iter   = 14

# ═══════════════════════════════════════════════════════════════════════════
# PADDING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

chi_work     = 128       # working bond dimension for TDVP (typically ≥ chi_dmrg)
pad_noise    = 1e-6      # noise amplitude in padded directions

# ═══════════════════════════════════════════════════════════════════════════
# TDVP PARAMETERS (1-site, evolution)
# ═══════════════════════════════════════════════════════════════════════════

dt          = 0.05
t_max       = 5.0

krylov_dim  = 14
krylov_tol  = 1e-8
evol_type   = "real"

# Note: chi_max and cutoff in TDVPOptions are IGNORED by 1-site TDVP
# (bond-preserving). They are passed only to satisfy the type signature.

# ═══════════════════════════════════════════════════════════════════════════
# MEASUREMENT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

measure_every = 10

# ═══════════════════════════════════════════════════════════════════════════

n_steps = round(Int, t_max / dt)

println("="^70)
println("  Quench dynamics: 1-site TDVP")
println("  (2-site DMRG warmup → pad → 1-site TDVP)")
println("="^70)
println("\nModel: N=$N, J=$J, d=$d")
println("Quench: Delta_i=$Delta_i → Delta_f=$Delta_f")
println("Field: h = ", length(h) <= 10 ? "$h" : "[$(h[1]), $(h[2]), ..., $(h[end])]")
println("\nDMRG (2-site): chi=$chi_dmrg, n_sweeps=$n_sweeps, cutoff=$cutoff_dmrg")
println("  Lanczos: krylov_dim=$lanczos_krylov_dim, max_iter=$lanczos_max_iter")
println("Padding: chi_dmrg=$chi_dmrg → chi_work=$chi_work, noise=$pad_noise")
println("TDVP (1-site): dt=$dt, t_max=$t_max, n_steps=$n_steps, χ_work=$chi_work (fixed)")
println("  Krylov: krylov_dim=$krylov_dim, tol=$krylov_tol, evol_type=$evol_type")
println("  Measure every $measure_every steps")
println("\nBLAS threads: $(BLAS.get_num_threads())")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: 2-site DMRG ground state at Delta_i
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  Step 1: 2-site DMRG ground state at Delta_i=$Delta_i (χ=$chi_dmrg)")
println("─"^70)

mpo_i = build_xxz_mpo(N, J, Delta_i, h; d=d)

sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps = product_state(sites, labels)

state = MPSState(mps, mpo_i; center=1)
solver_dmrg = LanczosSolver(lanczos_krylov_dim, lanczos_max_iter)
opts_dmrg = DMRGOptions(chi_dmrg, cutoff_dmrg, d)

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
println("  Energy variance ⟨H²⟩ − ⟨H⟩²: $var_E")

# Capture pre-padding diagnostics
ops = spin_ops(d)
Sz = Matrix{ComplexF64}(ops[:Z])
bonds_pre = [size(state.mps.tensors[i], 3) for i in 1:N-1]
sz_pre = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Pad MPS up to chi_work
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  Step 2: Pad MPS chi_dmrg=$chi_dmrg → chi_work=$chi_work (noise=$pad_noise)")
println("─"^70)

padded = pad_mps(state.mps, chi_work; noise=pad_noise)
bonds_post = [size(padded.tensors[i], 3) for i in 1:N-1]

println("  Bond profile: max χ before = $(maximum(bonds_pre)), max χ after = $(maximum(bonds_post))")
println("  Per-bond (first/last 8):")
for i in 1:min(8, N-1)
    println("    bond $(lpad(i,3)):  $(lpad(bonds_pre[i],3)) → $(bonds_post[i])")
end
if N-1 > 16
    println("    ...")
    for i in (N-8):(N-1)
        println("    bond $(lpad(i,3)):  $(lpad(bonds_pre[i],3)) → $(bonds_post[i])")
    end
end

# Sanity check: ⟨H_pre⟩ on padded state should match E_gs to ~O(noise²).
state_pad_check = MPSState(padded, mpo_i; center=1)
E_padded = measure_energy(state_pad_check)
sz_post = [real(measure_local_observable(padded, Sz, i)) for i in 1:N]
max_sz_shift = maximum(abs.(sz_post .- sz_pre))

println("\n  Padding sanity check (padded state still represents |ψ_gs⟩):")
println("    E_gs         (DMRG)        = $(round(E_gs, digits=12))")
println("    ⟨ψ_pad|H_pre|ψ_pad⟩         = $(round(E_padded, digits=12))")
println("    |ΔE|                         = $(round(abs(E_padded - E_gs), sigdigits=4))")
println("    expected O(noise²)           = $(round(pad_noise^2, sigdigits=2))")
println("    max |ΔSz_i|                  = $(round(max_sz_shift, sigdigits=4))")
println("    ‖ψ_pad‖²                     = $(round(real(measure_norm(padded)), digits=12))")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Switch MPO to post-quench H, build fresh environments
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  Step 3: Quench to Delta_f=$Delta_f — 1-site TDVP evolution")
println("─"^70)

mpo_f = build_xxz_mpo(N, J, Delta_f, h; d=d)
state = MPSState(padded, mpo_f; center=1)

solver_tdvp = KrylovExponential(krylov_dim, krylov_tol, evol_type)
tdvp_opts = TDVPOptions(dt, chi_work, 1e-12, d)

# Storage
times       = Float64[]
energies    = Float64[]
norms       = Float64[]
m_stag_t    = Float64[]
sz_profiles = Vector{Float64}[]
max_chis    = Int[]

# Record t=0 (post-quench)
E0 = measure_energy(state)
sz0 = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
m_stag0 = sum((-1.0)^i * sz0[i] for i in 1:N) / N

push!(times, 0.0)
push!(energies, E0)
push!(norms, real(measure_norm(state.mps)))
push!(m_stag_t, m_stag0)
push!(sz_profiles, sz0)
push!(max_chis, maximum(size(t, 3) for t in state.mps.tensors[1:end-1]))

println("  step     t       energy           dE            1-‖ψ‖²        χ_max")
println("  " * "─"^75)
println("  $(lpad(0, 4))   $(lpad("0.000", 6))   $(round(E0, digits=8))   $(round(0.0, digits=4))    $(round(1.0 - norms[1], sigdigits=3))    $(lpad(max_chis[1], 4))")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: 1-site TDVP loop
# ═══════════════════════════════════════════════════════════════════════════

for step in 1:n_steps
    global state

    res_r = tdvp_sweep_one_site(state, solver_tdvp, tdvp_opts, :right)
    res_l = tdvp_sweep_one_site(state, solver_tdvp, tdvp_opts, :left)

    t = step * dt

    if step % measure_every == 0 || step == n_steps
        E = measure_energy(state)
        nrm = real(measure_norm(state.mps))
        sz = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
        m_stag = sum((-1.0)^i * sz[i] for i in 1:N) / N
        mc = max(res_r.max_chi, res_l.max_chi)

        push!(times, t)
        push!(energies, E)
        push!(norms, nrm)
        push!(m_stag_t, m_stag)
        push!(sz_profiles, sz)
        push!(max_chis, mc)

        println("  $(lpad(step, 4))   $(lpad(round(t, digits=3), 6))   $(round(E, digits=8))   $(round(E - E0, sigdigits=3))    $(round(1.0 - nrm, sigdigits=3))    $(lpad(mc, 4))")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# FINAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  Final state at t=$t_max")
println("─"^70)

# ── Local Sz profile ────────────────────────────────────────────────────
println("\n  Local ⟨Sz_i⟩ profile")
println("  " * "─"^50)
sz_final = sz_profiles[end]
for i in 1:N
    bar = repeat("█", max(0, round(Int, (sz_final[i] + 0.5) * 40)))
    println("  Site $(lpad(i, 3)):  $(lpad(round(sz_final[i], digits=8), 12))  $bar")
end

# ── Bond entanglement entropies ─────────────────────────────────────────
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

# ── Staggered magnetization time series ─────────────────────────────────
println("\n  Staggered magnetization M_stag(t)")
println("  " * "─"^50)
for k in eachindex(times)
    println("  t = $(lpad(round(times[k], digits=3), 8))   M_stag = $(round(m_stag_t[k], digits=8))")
end

# ── Conservation diagnostics ────────────────────────────────────────────
max_E_drift     = maximum(abs.(energies .- E0))
max_norm_drift  = maximum(abs.(norms .- 1.0))

println("\n  Conservation (1-site TDVP: both should hold up to Krylov tol)")
println("  " * "─"^50)
println("  E(t=0)                = $E0")
println("  max|E(t) - E(0)|      = $max_E_drift")
println("  max|⟨ψ|ψ⟩ - 1|         = $max_norm_drift")
println("  Krylov tol             = $krylov_tol")

# ── CSV log ─────────────────────────────────────────────────────────────
csv_path = joinpath(@__DIR__, "quench_log_1site.csv")
open(csv_path, "w") do io
    println(io, "step,t,energy,dE,norm_sq,one_minus_norm_sq,m_stag,max_chi")
    for k in eachindex(times)
        step_k = (k == 1) ? 0 : (k - 1) * measure_every
        println(io, "$step_k,$(times[k]),$(energies[k]),$(energies[k]-E0),$(norms[k]),$(1.0-norms[k]),$(m_stag_t[k]),$(max_chis[k])")
    end
end
println("\n  Diagnostics CSV: $csv_path")

println("\n" * "="^70)
println("  Done.")
println("="^70)
