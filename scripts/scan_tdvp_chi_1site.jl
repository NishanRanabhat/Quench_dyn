# scripts/scan_tdvp_chi_1site.jl
#
# χ-scan for the 1-site TDVP quench protocol.
#
# Why this script: 1-site TDVP cannot report a per-step truncation error
# (QR/LQ only, no SVD). The standard correctness check is therefore
# convergence: run at several χ_work values and watch observables
# stabilize. Two diagnostics per run:
#   • Schmidt-spectrum tail: smallest retained singular value across bonds
#     at each measurement time. If this is still large (e.g. > 1e-3) the
#     dynamics is bumping the manifold ceiling and χ is too small.
#   • Energy / norm drift: should always be ~Krylov tol regardless of χ
#     (the algorithm is unitary on a fixed manifold by construction), so
#     drift is a Krylov diagnostic, NOT a manifold-adequacy diagnostic.
#
# Workflow:
#   1) 2-site DMRG once at chi_dmrg → ground state of H(Delta_i).
#   2) For each chi in chi_ladder:
#        - pad the DMRG ground state to chi (fresh pad each time,
#          not chained — different chi values get independent noise).
#        - switch MPO to H(Delta_f), run 1-site TDVP, log diagnostics.
#   3) Cross-χ convergence summary on M_stag(t; chi).

using LinearAlgebra
using Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", string(Sys.CPU_THREADS))))

# ═══════════════════════════════════════════════════════════════════════════
# MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

N        = 24
J        = 1.0
d        = 2
Delta_i  = 0.5
Delta_f  = 3.0
h        = [0.1 * (-1.0)^i for i in 1:N]

# ═══════════════════════════════════════════════════════════════════════════
# DMRG (ground state, run ONCE)
# ═══════════════════════════════════════════════════════════════════════════

chi_dmrg     = 64
n_sweeps_gs  = 25
cutoff_dmrg  = 1e-10
lanczos_dim  = 4
lanczos_iter = 14

# ═══════════════════════════════════════════════════════════════════════════
# PADDING + TDVP (one trajectory per χ in the ladder)
# ═══════════════════════════════════════════════════════════════════════════

chi_ladder  = [64, 96, 128]
pad_noise   = 1e-6
dt          = 0.05
t_max       = 3.0
krylov_dim  = 14
krylov_tol  = 1e-8

n_steps     = round(Int, t_max / dt)

out_dir = joinpath(@__DIR__, "scan_results_1site")
isdir(out_dir) || mkdir(out_dir)

println("="^70)
println("  1-site TDVP χ-scan (2-site DMRG → pad → 1-site TDVP)")
println("="^70)
println("Model: N=$N  J=$J  Δ_i=$Delta_i → Δ_f=$Delta_f")
println("DMRG : χ=$chi_dmrg, $n_sweeps_gs sweeps, cutoff=$cutoff_dmrg")
println("Padding: noise=$pad_noise")
println("TDVP : dt=$dt, t_max=$t_max ($n_steps steps), 1-site (no truncation)")
println("χ ladder: $chi_ladder")
println("Output dir: $out_dir")

# ═══════════════════════════════════════════════════════════════════════════
# Ground state at Δ_i (one shared DMRG run)
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  2-site DMRG ground state at Δ_i = $Delta_i")
println("─"^70)

mpo_i  = build_xxz_mpo(N, J, Delta_i, h)
sites  = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps_neel = product_state(sites, labels)

state_gs    = MPSState(mps_neel, mpo_i; center=1)
solver_dmrg = LanczosSolver(lanczos_dim, lanczos_iter)
opts_dmrg   = DMRGOptions(chi_dmrg, cutoff_dmrg, d)

E_gs = 0.0
for sweep in 1:n_sweeps_gs
    global E_gs
    dir = isodd(sweep) ? :right : :left
    E_gs = dmrg_sweep(state_gs, solver_dmrg, opts_dmrg, dir).E
end
var_E = energy_variance(state_gs)
@printf("  E_gs = %.10f   ‖ψ‖² = %.12f   var(H) = %.3e\n",
        E_gs, real(measure_norm(state_gs.mps)), var_E)

gs_tensors = [copy(t) for t in state_gs.mps.tensors]
gs_eltype  = eltype(state_gs.mps.tensors[1])

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

ops = spin_ops(d)
Sz  = Matrix{ComplexF64}(ops[:Z])

mpo_f = build_xxz_mpo(N, J, Delta_f, h)

# Schmidt spectrum tail per bond. Computes SVD at every bond and returns
# (min_smallest_sigma_across_bonds, max_smallest_sigma_across_bonds).
# The min is a "how much margin in the worst-converged bond" measure;
# the max is a "how much margin in the most-converged bond" sanity number.
function schmidt_tails(mps::MPS{T}) where T
    mps_copy = MPS{T}(copy.(mps.tensors))
    N = length(mps_copy.tensors)
    smallest = Float64[]
    for bond in 1:N-1
        make_canonical(mps_copy, bond)
        A = mps_copy.tensors[bond]
        chi_l, d_loc, chi_r = size(A)
        F = svd(reshape(A, chi_l * d_loc, chi_r))
        push!(smallest, F.S[end])
    end
    return minimum(smallest), maximum(smallest), smallest
end

# ═══════════════════════════════════════════════════════════════════════════
# Per-χ TDVP runs
# ═══════════════════════════════════════════════════════════════════════════

results = Dict{Int, NamedTuple}()

for chi in chi_ladder
    println("\n" * "─"^70)
    println("  TDVP run, χ_work = $chi")
    println("─"^70)

    mps_fresh = MPS{gs_eltype}([copy(t) for t in gs_tensors])

    # ── Pad to chi ──────────────────────────────────────────────────────
    padded = pad_mps(mps_fresh, chi; noise=pad_noise)

    # Padding sanity vs DMRG energy
    state_pad_check = MPSState(padded, mpo_i; center=1)
    E_pad_check = measure_energy(state_pad_check)
    @printf("  Padding: |ΔE| under H_pre = %.3e  (expect ~ noise² = %.0e)\n",
            abs(E_pad_check - E_gs), pad_noise^2)

    # ── Set up TDVP run on post-quench H ───────────────────────────────
    state  = MPSState(padded, mpo_f; center=1)
    solver = KrylovExponential(krylov_dim, krylov_tol, "real")
    opts   = TDVPOptions(dt, chi, 1e-12, d)   # chi/cutoff ignored by 1-site

    E0  = measure_energy(state)
    sz0 = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
    m0  = sum((-1.0)^i * sz0[i] for i in 1:N) / N
    sm_min0, sm_max0, _ = schmidt_tails(state.mps)

    times      = [0.0]
    energies   = [E0]
    m_stag     = [m0]
    one_m_norm = [1.0 - real(measure_norm(state.mps))]
    sm_tails   = [sm_min0]
    chis       = [maximum(size(t, 3) for t in state.mps.tensors[1:end-1])]

    csv_path = joinpath(out_dir, "tdvp_1site_chi$(chi).csv")
    io = open(csv_path, "w")
    println(io, "step,t,energy,dE,one_minus_norm_sq,m_stag,max_chi,min_schmidt_tail,max_schmidt_tail")
    @printf(io, "0,0.0,%.12f,0.0,%.3e,%.12f,%d,%.3e,%.3e\n",
            E0, one_m_norm[1], m0, chis[1], sm_min0, sm_max0)

    @printf("  step    t      dE          1-‖ψ‖²       max_chi   min(σ_min over bonds)\n")
    @printf("  %4d  %5.2f  %+.3e   %+.3e   %4d    %.3e\n",
            0, 0.0, 0.0, one_m_norm[1], chis[1], sm_min0)

    for step in 1:n_steps
        tdvp_sweep_one_site(state, solver, opts, :right)
        tdvp_sweep_one_site(state, solver, opts, :left)

        t   = step * dt
        E   = measure_energy(state)
        nrm = real(measure_norm(state.mps))
        sz  = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
        ms  = sum((-1.0)^i * sz[i] for i in 1:N) / N
        mc  = maximum(size(t, 3) for t in state.mps.tensors[1:end-1])

        # Schmidt tails are expensive (SVDs on every bond); compute only on
        # measurement steps.
        if step % 10 == 0 || step == n_steps
            sm_min, sm_max, _ = schmidt_tails(state.mps)
        else
            sm_min = NaN
            sm_max = NaN
        end

        push!(times, t); push!(energies, E); push!(m_stag, ms)
        push!(one_m_norm, 1.0 - nrm); push!(chis, mc); push!(sm_tails, sm_min)

        @printf(io, "%d,%f,%.12f,%.3e,%.3e,%.12f,%d,%.3e,%.3e\n",
                step, t, E, E - E0, 1.0 - nrm, ms, mc, sm_min, sm_max)

        if step % 10 == 0 || step == n_steps
            @printf("  %4d  %5.2f  %+.3e   %+.3e   %4d    %.3e\n",
                    step, t, E - E0, 1.0 - nrm, mc, sm_min)
        end
    end
    close(io)
    println("  CSV: $csv_path")

    results[chi] = (times=times, m_stag=m_stag, one_m_norm=one_m_norm,
                    chis=chis, energies=energies, sm_tails=sm_tails, E0=E0)
end

# ═══════════════════════════════════════════════════════════════════════════
# Cross-χ convergence summary
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("  Cross-χ convergence summary  (reference χ = $(maximum(chi_ladder)))")
println("="^70)

ref_chi   = maximum(chi_ladder)
ref       = results[ref_chi]
ref_times = ref.times

sample_idxs = unique([1; round.(Int, range(2, length(ref_times); length=8))])

let
    header = "  t      "
    for chi in chi_ladder
        header *= @sprintf("|χ=%-3d  M_stag       |dM_ref|    σ_tail_min  ", chi)
    end
    println(header)
    println("  " * "─"^(8 + 50 * length(chi_ladder)))

    for k in sample_idxs
        line = @sprintf("  %5.2f ", ref_times[k])
        m_ref = ref.m_stag[k]
        for chi in chi_ladder
            r = results[chi]
            m  = r.m_stag[k]
            diff = abs(m - m_ref)
            tail = r.sm_tails[k]
            tail_str = isnan(tail) ? "      —    " : @sprintf("%.2e  ", tail)
            line *= @sprintf("| %+.6f   %.2e    %s", m, diff, tail_str)
        end
        println(line)
    end
end

# Energy/norm drift summary per χ — should be uniformly small (~Krylov tol)
println("\n  Drift summary (1-site is unitary on fixed manifold; drift = Krylov error)")
@printf("    %4s   %-15s   %-15s\n", "χ", "max|E-E0|", "max|1-‖ψ‖²|")
for chi in chi_ladder
    r = results[chi]
    max_dE = maximum(abs.(r.energies .- r.E0))
    max_dn = maximum(abs.(r.one_m_norm))
    @printf("    %4d   %.3e        %.3e\n", chi, max_dE, max_dn)
end

println("\n  Manifold-adequacy summary (Schmidt tail at t=t_max per χ)")
@printf("    %4s   σ_tail_min @ t_max\n", "χ")
for chi in chi_ladder
    r = results[chi]
    tail_end = NaN
    for k in length(r.sm_tails):-1:1
        if !isnan(r.sm_tails[k]); tail_end = r.sm_tails[k]; break; end
    end
    @printf("    %4d   %.3e\n", chi, tail_end)
end

println("\n" * "="^70)
println("  Done. CSVs in $out_dir")
println("="^70)
