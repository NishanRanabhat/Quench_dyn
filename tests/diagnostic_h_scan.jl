# tests/diagnostic_h_scan.jl
#
# Follow-up to diagnostic_parity_h0.jl. Two scans at N=12, χ=64, Δi=0.5 → Δf=2.0:
#
#   Part (a): h-scan over disorder strength.
#       Manuscript protocol: random h only in H_i, clean (h=0) H_f.
#       For each h_amp in {0, 1e-8, 1e-6, 1e-4, 1e-2}, run:
#         ED ground state of H_i  →  ED time evolution under H_f
#         DMRG  →  2-site TDVP (cutoff=0)
#       Report: max_i |⟨S^z⟩|_ED(t),  max_i |⟨S^z⟩|_TDVP(t),  max_i|ED-TDVP|.
#       Expect signal ∝ h and TDVP floor ~1e-8 (from diagnostic_parity_h0.jl).
#
#   Part (b): DMRG-tightness scan at h=0.
#       Vary (cutoff, n_sweeps). Report max_i|⟨S^z_i⟩| at t=0 on the DMRG state.
#       Tests whether tightening DMRG lowers the noise floor.

using LinearAlgebra
using Printf
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

# ── Common ──────────────────────────────────────────────────────────────────
const N       = 12
const J       = 1.0
const d       = 2
const Δi      = 0.5
const Δf      = 2.0
const χ       = 64
const dt      = 0.05
const t_max   = 2.0
const krylov_dim = 30
const krylov_tol = 1e-14

ops      = spin_ops(d)
Sz_mat   = Matrix{ComplexF64}(ops[:Z])
times    = collect(0.0:dt:t_max)
n_steps  = length(times) - 1

# Fixed random spatial pattern of unit max amplitude.
Random.seed!(42)
h_pattern = randn(N)
h_pattern ./= maximum(abs.(h_pattern))

println(repeat("=", 90))
println("  Random h spatial pattern (scaled so max|h_i| = 1):")
print("    "); for v in h_pattern; @printf "%+.2f " v; end; println()
println()

# ── Helpers ────────────────────────────────────────────────────────────────

function run_dmrg(mps_init, mpo; cutoff, n_sweeps)
    state = MPSState(mps_init, mpo; center=1)
    solver = LanczosSolver(4, 14)
    opts   = DMRGOptions(χ, cutoff, d)
    E = 0.0
    for sweep in 1:n_sweeps
        dir = isodd(sweep) ? :right : :left
        res = dmrg_sweep(state, solver, opts, dir)
        E = res.E
    end
    return state, E
end

sz_profile_mps(mps) = [real(measure_local_observable(mps, Sz_mat, i)) for i in 1:N]

function tdvp_evolution!(state, mpo_f; cutoff=0.0)
    solver = KrylovExponential(krylov_dim, krylov_tol, "real")
    tdvp_opts = TDVPOptions(dt, χ, cutoff, d)
    # Move state to post-quench Hamiltonian.
    state = MPSState(state.mps, mpo_f; center=1)
    profiles = Vector{Vector{Float64}}()
    push!(profiles, sz_profile_mps(state.mps))
    for _ in 1:n_steps
        tdvp_sweep(state, solver, tdvp_opts, :right)
        tdvp_sweep(state, solver, tdvp_opts, :left)
        push!(profiles, sz_profile_mps(state.mps))
    end
    return profiles
end

# Build Pauli-X parity operator once (constant, h-independent).
const σx = 2 .* Matrix{ComplexF64}(ops[:X])
const P_op = embed_multi_site([σx for _ in 1:N], collect(1:N), N; d=d)

# Build the post-quench Hamiltonian once (always clean, h=0 in H_f).
println("[setup] Building clean H_f and its eigensystem …")
mpo_f_clean = build_xxz_mpo(N, J, Δf, zeros(N); d=d)
H_f_clean   = build_xxz_hamiltonian(N, J, Δf, zeros(N); d=d)
eig_f       = diagonalize(H_f_clean)

# Initial product state (Neel branch A) used as DMRG seed for all runs.
sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels_A = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps_A0 = product_state(sites, labels_A)

# ═══════════════════════════════════════════════════════════════════════════
# PART (a): h-scan
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * repeat("=", 90))
println("  Part (a): h-scan  (clean H_f; h only in H_i, scaled spatial pattern)")
println(repeat("=", 90))

h_amps = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]

@printf "  %-10s  %-15s  %-15s  %-15s  %-15s  %-15s\n" "h_amp" "max|Sz| ED(t)" "max|Sz| TDVP(t)" "max|ED-TDVP|" "DMRG seed |Sz|(0)" "rel err"
println(repeat("─", 100))

results_a = Vector{NamedTuple}()
for h_amp in h_amps
    h = h_amp .* h_pattern

    # ED side: ground state of H_i (with field), evolve under H_f (clean).
    H_i = build_xxz_hamiltonian(N, J, Δi, h; d=d)
    eig_i = diagonalize(H_i)
    ψ_ED  = Vector{ComplexF64}(eig_i.vectors[:, 1]); ψ_ED ./= norm(ψ_ED)

    sz_ED_t = [ed_local_profile(ed_time_evolve(eig_f, ψ_ED, t),
                                Matrix{ComplexF64}(ops[:Z]), N; d=d) for t in times]

    # MPS side: DMRG with H_i (with field), TDVP under H_f (clean).
    mpo_i = build_xxz_mpo(N, J, Δi, h; d=d)
    state, _ = run_dmrg(mps_A0, mpo_i; cutoff=1e-12, n_sweeps=40)
    sz_DMRG_init = sz_profile_mps(state.mps)
    sz_TDVP_t = tdvp_evolution!(state, mpo_f_clean; cutoff=0.0)

    max_ed   = maximum(maximum(abs.(s)) for s in sz_ED_t)
    max_tdvp = maximum(maximum(abs.(s)) for s in sz_TDVP_t)
    max_err  = maximum(maximum(abs.(sz_ED_t[k] .- sz_TDVP_t[k])) for k in eachindex(times))
    max_seed = maximum(abs.(sz_DMRG_init))
    rel_err  = max_ed > 1e-20 ? max_err / max_ed : NaN

    push!(results_a, (h_amp=h_amp, max_ed=max_ed, max_tdvp=max_tdvp, max_err=max_err, max_seed=max_seed, rel_err=rel_err))
    @printf "  %-10.0e  %-15.3e  %-15.3e  %-15.3e  %-15.3e  %-15.3e\n" h_amp max_ed max_tdvp max_err max_seed rel_err
end

# ═══════════════════════════════════════════════════════════════════════════
# PART (b): DMRG-tightness scan at h=0
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * repeat("=", 90))
println("  Part (b): DMRG-tightness scan at h=0  (does tighter DMRG lower the floor?)")
println(repeat("=", 90))

settings = [
    (1e-10, 30),
    (1e-12, 40),
    (1e-14, 40),
    (1e-14, 100),
    (1e-16, 100),
    (0.0,   100),
]

mpo_i_clean = build_xxz_mpo(N, J, Δi, zeros(N); d=d)
H_i_clean   = build_xxz_hamiltonian(N, J, Δi, zeros(N); d=d)
eig_i_clean = diagonalize(H_i_clean)
E_ED        = eig_i_clean.values[1]
ψ_ED_clean  = Vector{ComplexF64}(eig_i_clean.vectors[:, 1]); ψ_ED_clean ./= norm(ψ_ED_clean)

@printf "  %-10s  %-10s  %-15s  %-15s  %-15s  %-15s\n" "cutoff" "n_sweeps" "ΔE (DMRG-ED)" "1-|⟨ED|DMRG⟩|²" "1-⟨P⟩" "max|⟨Sz⟩|(t=0)"
println(repeat("─", 100))

results_b = Vector{NamedTuple}()
for (cut, ns) in settings
    state, E = run_dmrg(mps_A0, mpo_i_clean; cutoff=cut, n_sweeps=ns)
    ψ_DMRG = mps_to_vector(state.mps; d=d); ψ_DMRG ./= norm(ψ_DMRG)
    overlap  = abs2(ψ_ED_clean' * ψ_DMRG)
    P_DMRG   = real(ψ_DMRG' * P_op * ψ_DMRG)
    sz_init  = sz_profile_mps(state.mps)
    max_sz   = maximum(abs.(sz_init))
    push!(results_b, (cutoff=cut, n_sweeps=ns, dE=E-E_ED, one_minus_ov=1.0-overlap, one_minus_P=1.0-P_DMRG, max_sz=max_sz))
    @printf "  %-10.0e  %-10d  %+15.3e  %15.3e  %15.3e  %15.3e\n" cut ns (E-E_ED) (1.0-overlap) (1.0-P_DMRG) max_sz
end

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * repeat("=", 90))
println("  TAKEAWAYS")
println(repeat("=", 90))
println("  Part (a):")
println("    • Signal  max|⟨Sz⟩|_ED grows linearly with h_amp (perturbative O(h) regime).")
println("    • Floor   max|ED−TDVP| should be roughly constant ≈ DMRG seed magnitude.")
println("    • Crossover at h_amp where signal ≈ floor.")
println("  Part (b):")
println("    • If max|⟨Sz⟩|(t=0) shrinks with tighter cutoff/more sweeps → fixable.")
println("    • If it stays at the same magnitude → structural floor of SVD sign noise.")
