# scripts/scan_tdvp_chi.jl
#
# χ-scan for TDVP quench dynamics. Prepares the ground state once, then
# evolves it under the post-quench Hamiltonian for a ladder of bond-
# dimension caps. One CSV per χ + a cross-χ convergence summary at the
# end.
#
# Usage: edit the parameter block, then `julia scripts/scan_tdvp_chi.jl`.

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
# DMRG (ground state, run once)
# ═══════════════════════════════════════════════════════════════════════════

chi_dmrg     = 64
n_sweeps_gs  = 25
cutoff_dmrg  = 1e-10
lanczos_dim  = 4
lanczos_iter = 14

# ═══════════════════════════════════════════════════════════════════════════
# TDVP (one trajectory per χ in the ladder)
# ═══════════════════════════════════════════════════════════════════════════

chi_ladder  = [64, 96, 128]
dt          = 0.05
t_max       = 3.0
cutoff_tdvp = 1e-10
krylov_dim  = 14
krylov_tol  = 1e-8

n_steps     = round(Int, t_max / dt)

out_dir = joinpath(@__DIR__, "scan_results")
isdir(out_dir) || mkdir(out_dir)

println("="^70)
println("  TDVP χ-scan")
println("="^70)
println("Model: N=$N  J=$J  Δ_i=$Delta_i → Δ_f=$Delta_f")
println("DMRG : χ=$chi_dmrg, $n_sweeps_gs sweeps, cutoff=$cutoff_dmrg")
println("TDVP : dt=$dt, t_max=$t_max ($n_steps steps), cutoff=$cutoff_tdvp")
println("χ ladder: $chi_ladder")
println("Output dir: $out_dir")

# ═══════════════════════════════════════════════════════════════════════════
# Ground state at Δ_i (run once, snapshot tensors for each χ run)
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "─"^70)
println("  DMRG ground state at Δ_i = $Delta_i")
println("─"^70)

mpo_i  = build_xxz_mpo(N, J, Delta_i, h)
sites  = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps    = product_state(sites, labels)

state_gs    = MPSState(mps, mpo_i; center=1)
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
# Per-χ TDVP runs
# ═══════════════════════════════════════════════════════════════════════════

ops = spin_ops(d)
Sz  = Matrix{ComplexF64}(ops[:Z])

mpo_f = build_xxz_mpo(N, J, Delta_f, h)

# Storage: chi → (times, m_stag(t), 1-‖ψ‖²(t), max_chi(t), max_trunc(t))
results = Dict{Int, NamedTuple}()

for chi in chi_ladder
    println("\n" * "─"^70)
    println("  TDVP run, χ_max = $chi")
    println("─"^70)

    mps_fresh = MPS{gs_eltype}([copy(t) for t in gs_tensors])
    state     = MPSState(mps_fresh, mpo_f; center=1)
    solver    = KrylovExponential(krylov_dim, krylov_tol, "real")
    opts      = TDVPOptions(dt, chi, cutoff_tdvp, d)

    E0 = measure_energy(state)
    sz0 = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
    m0 = sum((-1.0)^i * sz0[i] for i in 1:N) / N

    times      = [0.0]
    m_stag     = [m0]
    one_m_norm = [1.0 - real(measure_norm(state.mps))]
    chis       = [maximum(size(t, 3) for t in state.mps.tensors[1:end-1])]
    truncs     = [0.0]
    energies   = [E0]

    csv_path = joinpath(out_dir, "tdvp_chi$(chi).csv")
    io = open(csv_path, "w")
    println(io, "step,t,energy,dE,one_minus_norm_sq,m_stag,max_chi,max_trunc,total_trunc")
    println(io, "0,0.0,$E0,0.0,$(one_m_norm[1]),$m0,$(chis[1]),0.0,0.0")

    @printf("  step    t     dE          1-‖ψ‖²       max_chi   max_trunc\n")
    @printf("  %4d  %5.2f  %+.3e   %+.3e   %4d    %.3e\n",
            0, 0.0, 0.0, one_m_norm[1], chis[1], 0.0)

    for step in 1:n_steps
        res_r = tdvp_sweep(state, solver, opts, :right)
        res_l = tdvp_sweep(state, solver, opts, :left)

        t   = step * dt
        E   = measure_energy(state)
        nrm = real(measure_norm(state.mps))
        sz  = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
        ms  = sum((-1.0)^i * sz[i] for i in 1:N) / N
        mc  = max(res_r.max_chi, res_l.max_chi)
        mt  = max(res_r.max_trunc, res_l.max_trunc)
        tt  = res_r.total_trunc + res_l.total_trunc

        push!(times, t); push!(m_stag, ms); push!(one_m_norm, 1.0 - nrm)
        push!(chis, mc); push!(truncs, mt); push!(energies, E)

        println(io, "$step,$t,$E,$(E-E0),$(1.0-nrm),$ms,$mc,$mt,$tt")

        if step % 10 == 0 || step == n_steps
            @printf("  %4d  %5.2f  %+.3e   %+.3e   %4d    %.3e\n",
                    step, t, E - E0, 1.0 - nrm, mc, mt)
        end
    end
    close(io)
    println("  CSV: $csv_path")

    results[chi] = (times=times, m_stag=m_stag, one_m_norm=one_m_norm,
                    chis=chis, truncs=truncs, energies=energies, E0=E0)
end

# ═══════════════════════════════════════════════════════════════════════════
# Cross-χ convergence summary
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("  Cross-χ convergence summary")
println("="^70)

ref_chi   = maximum(chi_ladder)
ref       = results[ref_chi]
ref_times = ref.times

# Pick a sparse subset of times for the table
sample_idxs = unique([1; round.(Int, range(2, length(ref_times); length=8))])

let
    header = "  t      "
    for chi in chi_ladder
        header *= @sprintf("|χ=%-3d  Mₛ        |dM_ref|     1-‖ψ‖²    ", chi)
    end
    println(header)
    println("  " * "─"^(8 + 50 * length(chi_ladder)))

    for k in sample_idxs
        line = @sprintf("  %5.2f ", ref_times[k])
        m_ref = ref.m_stag[k]
        for chi in chi_ladder
            r = results[chi]
            m  = r.m_stag[k]
            on = r.one_m_norm[k]
            diff = abs(m - m_ref)
            line *= @sprintf("| %+.6f   %.2e    %+.2e  ", m, diff, on)
        end
        println(line)
    end
end

# t_reliable per χ: last t where 1-‖ψ‖² < 1e-6
println("\n  t_reliable(χ) = last t where 1-‖ψ‖² < 1e-6")
for chi in chi_ladder
    r = results[chi]
    idx = findlast(x -> abs(x) < 1e-6, r.one_m_norm)
    t_rel = isnothing(idx) ? 0.0 : r.times[idx]
    @printf("    χ=%3d : t_reliable = %.3f\n", chi, t_rel)
end

println("\n" * "="^70)
println("  Done. CSVs in $out_dir")
println("="^70)
