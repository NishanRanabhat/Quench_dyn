# tests/diagnostic_parity_h0.jl
#
# Diagnostic to localize the source of the ED-vs-TDVP discrepancy at h=0.
#
# Three tests at N=12, chi=64, h=0:
#   A. Standard protocol: DMRG from |Neel A> init, then 2-site TDVP vs ED.
#   B. Cheat: convert exact ED ground state to MPS, then TDVP vs ED.
#      Isolates whether TDVP itself is at fault or it is the DMRG seed.
#   C. Parity-symmetric init: DMRG from (|A>+|B>)/sqrt(2), then TDVP vs ED.
#      Tests whether a parity-pure starting MPS survives DMRG sweeps.
#
# Measurements at every step:
#   - Ground-state energy ED vs DMRG
#   - Overlap |<ED|DMRG>|^2
#   - <P> = expectation of global spin-flip parity P = prod_i sigma^x_i
#   - <S^z_i> per-site profile at t=0 and at several t>0 (ED and TDVP)
#   - max_i |<S^z_i(t)>_ED - <S^z_i(t)>_TDVP|

using LinearAlgebra
using Printf

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

const N      = 12
const J      = 1.0
const d      = 2
const Δi     = 0.5
const Δf     = 2.0
const h      = zeros(N)              # h=0: this is the troublesome case
const χ      = 64                    # full Hilbert: 2^(N/2) = 64
const n_sweeps_dmrg = 40
const cutoff_dmrg   = 1e-12
const dt            = 0.05
const t_max         = 2.0
const cutoff_tdvp   = 0.0            # no truncation
const krylov_dim    = 30
const krylov_tol    = 1e-14

const TIMES_TO_PRINT = [0.0, 0.5, 1.0, 1.5, 2.0]

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

ops = spin_ops(d)
const Sz_mat = Matrix{ComplexF64}(ops[:Z])
const σx     = 2 .* Matrix{ComplexF64}(ops[:X])     # Pauli X

"""Global parity operator P = prod_i σ^x_i as a dense 2^N x 2^N matrix."""
function build_parity_operator(N; d=2)
    return embed_multi_site([σx for _ in 1:N], collect(1:N), N; d=d)
end

"""Convert dense state vector psi (length d^N) to a left-canonical MPS via
successive SVDs. Site 1 = MSB, matching mps_to_vector / ED convention."""
function vector_to_mps(psi::AbstractVector, N::Int; d::Int=2)
    @assert length(psi) == d^N
    tensors = Vector{Array{ComplexF64,3}}(undef, N)
    psi_mat = reshape(Vector{ComplexF64}(psi), (1, d^N))
    chi_l = 1
    for i in 1:N-1
        remaining = d^(N - i)
        psi_mat = reshape(psi_mat, (chi_l * d, remaining))
        F = svd(psi_mat)
        chi_r = length(F.S)
        tensors[i] = reshape(F.U, (chi_l, d, chi_r))
        psi_mat   = Diagonal(F.S) * F.Vt
        chi_l = chi_r
    end
    tensors[N] = reshape(psi_mat, (chi_l, d, 1))
    return MPS{ComplexF64}(tensors)
end

"""Build the parity-symmetric (|A>+|B>)/sqrt(2) Neel superposition as a
bond-dim-2 MPS, where |A>=|up dn up dn...> and |B>=|dn up dn up...>."""
function neel_symmetric_mps(N::Int; d::Int=2)
    @assert iseven(N)
    T = ComplexF64
    s = 1 / sqrt(T(2))

    tensors = Vector{Array{T,3}}(undef, N)

    # First site: maps bond=1 to bond=(branch label).
    W1 = zeros(T, 1, d, 2)
    W1[1, 1, 1] = s         # up,  branch A
    W1[1, 2, 2] = s         # dn,  branch B
    tensors[1] = W1

    # Bulk sites: branch label preserved; physical spin set by branch + parity of site.
    for i in 2:N-1
        Wi = zeros(T, 2, d, 2)
        if isodd(i)
            Wi[1, 1, 1] = 1      # branch A at odd site → up
            Wi[2, 2, 2] = 1      # branch B at odd site → dn
        else
            Wi[1, 2, 1] = 1      # branch A at even site → dn
            Wi[2, 1, 2] = 1      # branch B at even site → up
        end
        tensors[i] = Wi
    end

    # Last site (N even): A ends in dn, B ends in up. Bond → 1.
    WN = zeros(T, 2, d, 1)
    WN[1, 2, 1] = 1
    WN[2, 1, 1] = 1
    tensors[N] = WN

    return MPS{T}(tensors)
end

"""Run DMRG from a given initial MPS, return (state, E_history)."""
function run_dmrg(mps_init, mpo, χ, n_sweeps, cutoff)
    state = MPSState(mps_init, mpo; center=1)
    solver = LanczosSolver(4, 14)
    opts   = DMRGOptions(χ, cutoff, d)
    E_hist = Float64[]
    for sweep in 1:n_sweeps
        dir = isodd(sweep) ? :right : :left
        res = dmrg_sweep(state, solver, opts, dir)
        push!(E_hist, res.E)
    end
    return state, E_hist
end

"""Per-site <S^z_i> on a given MPS."""
sz_profile_mps(mps) = [real(measure_local_observable(mps, Sz_mat, i)) for i in 1:N]

"""Print a per-site S^z table."""
function print_sz_profile(label, sz_vec)
    @printf "  %-22s " label
    for v in sz_vec
        @printf "%+9.2e " v
    end
    @printf "\n"
end

# ═══════════════════════════════════════════════════════════════════════════
# COMMON SETUP
# ═══════════════════════════════════════════════════════════════════════════

println(repeat("=", 90))
println("  Diagnostic: ED vs TDVP at h=0, N=$N, χ=$χ, Δi=$Δi → Δf=$Δf")
println(repeat("=", 90))

# Build dense + MPO Hamiltonians.
println("\n[setup] Building Hamiltonians …")
mpo_i = build_xxz_mpo(N, J, Δi, h; d=d)
mpo_f = build_xxz_mpo(N, J, Δf, h; d=d)
H_i   = build_xxz_hamiltonian(N, J, Δi, h; d=d)
H_f   = build_xxz_hamiltonian(N, J, Δf, h; d=d)

# ED ground state of H_i.
println("[setup] Diagonalizing H_i (size $(size(H_i,1)) × $(size(H_i,1))) …")
eig_i = diagonalize(H_i)
E_ED  = eig_i.values[1]
ψ_ED  = Vector{ComplexF64}(eig_i.vectors[:, 1])
ψ_ED ./= norm(ψ_ED)

# ED eigensystem of H_f for time evolution.
println("[setup] Diagonalizing H_f for time evolution …")
eig_f = diagonalize(H_f)

# Parity operator (full 2^N × 2^N dense; cheap at N=12).
println("[setup] Building parity operator P = ⊗_i σ^x …")
P_op = build_parity_operator(N; d=d)

# Diagnostics on ED ground state.
P_ED = real(ψ_ED' * P_op * ψ_ED)
sz_ED_init = ed_local_profile(ψ_ED, Matrix{ComplexF64}(ops[:Z]), N; d=d)
gap_ED = eig_i.values[2] - eig_i.values[1]

println("\n--- ED ground state of H_i ---")
@printf "  E_GS (ED)         = %.10f\n" E_ED
@printf "  Gap to 1st excited = %.6e\n" gap_ED
@printf "  ⟨P⟩ on ED GS      = %+.6e   (should be ±1)\n" P_ED
@printf "  max_i |⟨S^z_i⟩|   = %.3e\n" maximum(abs.(sz_ED_init))
print_sz_profile("ED ⟨S^z_i⟩ (t=0):", sz_ED_init)

# Pre-compute ED time-evolved profiles.
times = collect(0.0:dt:t_max)
sz_ED_t = Vector{Vector{Float64}}()
for t in times
    ψt = ed_time_evolve(eig_f, ψ_ED, t)
    push!(sz_ED_t, ed_local_profile(ψt, Matrix{ComplexF64}(ops[:Z]), N; d=d))
end

# ═══════════════════════════════════════════════════════════════════════════
# TEST A: STANDARD PROTOCOL — DMRG from |Neel A>
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * repeat("─", 90))
println(" TEST A — DMRG from |↑↓↑↓…⟩ (current default in scripts/run_tdvp_2site.jl)")
println(repeat("─", 90))

sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels_A = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps_A0 = product_state(sites, labels_A)

state_A, E_hist_A = run_dmrg(mps_A0, mpo_i, χ, n_sweeps_dmrg, cutoff_dmrg)
E_DMRG_A = E_hist_A[end]
ψ_DMRG_A = mps_to_vector(state_A.mps; d=d)
ψ_DMRG_A ./= norm(ψ_DMRG_A)
overlap_A = abs2(ψ_ED' * ψ_DMRG_A)
P_DMRG_A  = real(ψ_DMRG_A' * P_op * ψ_DMRG_A)
sz_DMRG_A_init = sz_profile_mps(state_A.mps)

@printf "  E_GS (DMRG)         = %.10f\n" E_DMRG_A
@printf "  ΔE  (DMRG − ED)     = %+.3e   (should be ~0)\n" (E_DMRG_A - E_ED)
@printf "  |⟨ED|DMRG⟩|²        = %.10f   (should be 1)\n" overlap_A
@printf "  ⟨P⟩ on DMRG GS      = %+.6e   (should be ±1)\n" P_DMRG_A
@printf "  max_i |⟨S^z_i⟩|     = %.3e   (should be ~%.0e)\n" maximum(abs.(sz_DMRG_A_init)) maximum(abs.(sz_ED_init))
print_sz_profile("ED   ⟨S^z_i⟩ (t=0):", sz_ED_init)
print_sz_profile("DMRG ⟨S^z_i⟩ (t=0):", sz_DMRG_A_init)

println("\n  TDVP time evolution (start from DMRG state, 2-site, cutoff=0):")
state_tdvp_A = MPSState(state_A.mps, mpo_f; center=1)
solver_tdvp = KrylovExponential(krylov_dim, krylov_tol, "real")
tdvp_opts   = TDVPOptions(dt, χ, cutoff_tdvp, d)

sz_TDVP_A_t = Vector{Vector{Float64}}()
push!(sz_TDVP_A_t, sz_profile_mps(state_tdvp_A.mps))
n_steps = length(times) - 1
for step in 1:n_steps
    tdvp_sweep(state_tdvp_A, solver_tdvp, tdvp_opts, :right)
    tdvp_sweep(state_tdvp_A, solver_tdvp, tdvp_opts, :left)
    push!(sz_TDVP_A_t, sz_profile_mps(state_tdvp_A.mps))
end

println("\n  Per-site ⟨S^z_i(t)⟩ at selected times (ED vs TDVP):")
for tp in TIMES_TO_PRINT
    k = findfirst(t -> isapprox(t, tp; atol=1e-9), times)
    k === nothing && continue
    println("\n  t = $(times[k]):")
    print_sz_profile("    ED   ⟨S^z_i⟩:", sz_ED_t[k])
    print_sz_profile("    TDVP ⟨S^z_i⟩:", sz_TDVP_A_t[k])
    @printf "    max_i |ED-TDVP| = %.3e\n" maximum(abs.(sz_ED_t[k] .- sz_TDVP_A_t[k]))
end

# ═══════════════════════════════════════════════════════════════════════════
# TEST B: TDVP starting from EXACT ED ground state (converted to MPS)
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * repeat("─", 90))
println(" TEST B — TDVP from exact ED ground state (vector→MPS via SVD)")
println(repeat("─", 90))

mps_ED = vector_to_mps(ψ_ED, N; d=d)
ψ_back = mps_to_vector(mps_ED; d=d)
ψ_back ./= norm(ψ_back)
@printf "  Round-trip overlap |⟨ED|MPS(ED)⟩|² = %.12f   (should be 1)\n" abs2(ψ_ED' * ψ_back)
sz_init_B = sz_profile_mps(mps_ED)
@printf "  max_i |⟨S^z_i⟩| on MPS(ED) = %.3e\n" maximum(abs.(sz_init_B))

state_tdvp_B = MPSState(mps_ED, mpo_f; center=1)
sz_TDVP_B_t = Vector{Vector{Float64}}()
push!(sz_TDVP_B_t, sz_profile_mps(state_tdvp_B.mps))
for step in 1:n_steps
    tdvp_sweep(state_tdvp_B, solver_tdvp, tdvp_opts, :right)
    tdvp_sweep(state_tdvp_B, solver_tdvp, tdvp_opts, :left)
    push!(sz_TDVP_B_t, sz_profile_mps(state_tdvp_B.mps))
end

println("\n  Per-site ⟨S^z_i(t)⟩ at selected times (ED vs TDVP-from-ED-init):")
for tp in TIMES_TO_PRINT
    k = findfirst(t -> isapprox(t, tp; atol=1e-9), times)
    k === nothing && continue
    @printf "  t = %.2f   max_i |ED-TDVP| = %.3e\n" times[k] maximum(abs.(sz_ED_t[k] .- sz_TDVP_B_t[k]))
end

# ═══════════════════════════════════════════════════════════════════════════
# TEST C: DMRG from (|A>+|B>)/sqrt(2) (parity-even product superposition)
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * repeat("─", 90))
println(" TEST C — DMRG from parity-symmetric (|A⟩+|B⟩)/√2  init")
println(repeat("─", 90))

mps_C0 = neel_symmetric_mps(N; d=d)
ψ_C0 = mps_to_vector(mps_C0; d=d); ψ_C0 ./= norm(ψ_C0)
@printf "  ⟨P⟩ on (|A⟩+|B⟩)/√2 (t=0) = %+.6e   (should be +1)\n" real(ψ_C0' * P_op * ψ_C0)

state_C, E_hist_C = run_dmrg(mps_C0, mpo_i, χ, n_sweeps_dmrg, cutoff_dmrg)
E_DMRG_C = E_hist_C[end]
ψ_DMRG_C = mps_to_vector(state_C.mps; d=d); ψ_DMRG_C ./= norm(ψ_DMRG_C)
overlap_C = abs2(ψ_ED' * ψ_DMRG_C)
P_DMRG_C  = real(ψ_DMRG_C' * P_op * ψ_DMRG_C)
sz_DMRG_C_init = sz_profile_mps(state_C.mps)

@printf "  E_GS (DMRG)         = %.10f\n" E_DMRG_C
@printf "  ΔE  (DMRG − ED)     = %+.3e\n" (E_DMRG_C - E_ED)
@printf "  |⟨ED|DMRG⟩|²        = %.10f\n" overlap_C
@printf "  ⟨P⟩ on DMRG GS      = %+.6e   (should be +1)\n" P_DMRG_C
@printf "  max_i |⟨S^z_i⟩|     = %.3e\n" maximum(abs.(sz_DMRG_C_init))

state_tdvp_C = MPSState(state_C.mps, mpo_f; center=1)
sz_TDVP_C_t = Vector{Vector{Float64}}()
push!(sz_TDVP_C_t, sz_profile_mps(state_tdvp_C.mps))
for step in 1:n_steps
    tdvp_sweep(state_tdvp_C, solver_tdvp, tdvp_opts, :right)
    tdvp_sweep(state_tdvp_C, solver_tdvp, tdvp_opts, :left)
    push!(sz_TDVP_C_t, sz_profile_mps(state_tdvp_C.mps))
end

println("\n  Per-site ⟨S^z_i(t)⟩ at selected times (ED vs TDVP):")
for tp in TIMES_TO_PRINT
    k = findfirst(t -> isapprox(t, tp; atol=1e-9), times)
    k === nothing && continue
    @printf "  t = %.2f   max_i |ED-TDVP| = %.3e\n" times[k] maximum(abs.(sz_ED_t[k] .- sz_TDVP_C_t[k]))
end

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * repeat("=", 90))
println("  SUMMARY")
println(repeat("=", 90))
@printf "  %-50s %-20s\n"  "Quantity"  "Value"
println(repeat("─", 90))
@printf "  %-50s %.10f\n"  "E_GS (ED)"                E_ED
@printf "  %-50s %.10f\n"  "E_GS (DMRG, |A⟩ init)"    E_DMRG_A
@printf "  %-50s %.10f\n"  "E_GS (DMRG, sym init)"    E_DMRG_C
@printf "  %-50s %+.3e\n"  "⟨P⟩ ED"                   P_ED
@printf "  %-50s %+.3e\n"  "⟨P⟩ DMRG (|A⟩ init)"      P_DMRG_A
@printf "  %-50s %+.3e\n"  "⟨P⟩ DMRG (sym init)"      P_DMRG_C
@printf "  %-50s %.10f\n"  "|⟨ED|DMRG⟩|² (|A⟩ init)"  overlap_A
@printf "  %-50s %.10f\n"  "|⟨ED|DMRG⟩|² (sym init)"  overlap_C
@printf "  %-50s %.3e\n"   "max_t max_i |ED-TDVP_A|"   maximum(maximum(abs.(sz_ED_t[k] .- sz_TDVP_A_t[k])) for k in eachindex(times))
@printf "  %-50s %.3e\n"   "max_t max_i |ED-TDVP_B|"   maximum(maximum(abs.(sz_ED_t[k] .- sz_TDVP_B_t[k])) for k in eachindex(times))
@printf "  %-50s %.3e\n"   "max_t max_i |ED-TDVP_C|"   maximum(maximum(abs.(sz_ED_t[k] .- sz_TDVP_C_t[k])) for k in eachindex(times))
println(repeat("=", 90))
