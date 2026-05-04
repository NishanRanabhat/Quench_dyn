# tests/benchmark_tdvp_drift.jl
#
# Per-step diagnostic for 2-site TDVP: tracks norm, energy, max bond dim
# at every full step (right + left sweep). Uses the public API only —
# no source modifications. Writes CSV for offline plotting.

using LinearAlgebra
using Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

# ── Parameters (mirrors run_tdvp.jl, but smaller N for a quick run) ──────

N        = 20
J        = 1.0
Delta_i  = 0.5
Delta_f  = 3.0
h        = [0.1 * (-1.0)^i for i in 1:N]

chi_dmrg     = 64
n_sweeps_gs  = 20
cutoff_dmrg  = 1e-10

dt           = 0.05
t_max        = 3.0
n_steps      = round(Int, t_max / dt)
chi_tdvp     = 128
cutoff_tdvp  = 1e-8
krylov_dim   = 14
krylov_tol   = 1e-8

csv_path = joinpath(@__DIR__, "benchmark_tdvp_drift.csv")

println("="^70)
println("  TDVP drift diagnostic")
println("="^70)
println("N=$N  J=$J  Δi=$Delta_i → Δf=$Delta_f  h=staggered 0.1")
println("DMRG: χ=$chi_dmrg, $n_sweeps_gs sweeps, cutoff=$cutoff_dmrg")
println("TDVP: dt=$dt, t_max=$t_max ($n_steps steps), χ=$chi_tdvp, cutoff=$cutoff_tdvp")
println("Krylov: dim=$krylov_dim, tol=$krylov_tol")

# ── Ground state at Δi via DMRG ──────────────────────────────────────────

mpo_i  = build_xxz_mpo(N, J, Delta_i, h)
sites  = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps    = product_state(sites, labels)

state       = MPSState(mps, mpo_i; center=1)
solver_dmrg = LanczosSolver(4, 14)
opts_dmrg   = DMRGOptions(chi_dmrg, cutoff_dmrg, 2)

println("\nDMRG ground state ...")
E_gs = 0.0
for sweep in 1:n_sweeps_gs
    global E_gs
    dir = isodd(sweep) ? :right : :left
    E_gs = dmrg_sweep(state, solver_dmrg, opts_dmrg, dir).E
end
@printf("  E_gs = %.10f   ‖ψ‖² = %.12f\n", E_gs, real(measure_norm(state.mps)))

# ── Quench → TDVP ────────────────────────────────────────────────────────

mpo_f       = build_xxz_mpo(N, J, Delta_f, h)
state       = MPSState(state.mps, mpo_f; center=1)
solver      = KrylovExponential(krylov_dim, krylov_tol, "real")
opts        = TDVPOptions(dt, chi_tdvp, cutoff_tdvp, 2)

max_bond(m) = maximum(size(t, 3) for t in m.tensors[1:end-1])

E0 = measure_energy(state)
n0 = real(measure_norm(state.mps))
chi0 = max_bond(state.mps)

println("\nTDVP evolution → CSV at $csv_path")
println("step    t      ‖ψ‖²            E             dE          1-‖ψ‖²       χ_max   max_trunc")

io = open(csv_path, "w")
println(io, "step,t,norm_sq,energy,dE,one_minus_norm_sq,max_chi,max_trunc,total_trunc")
println(io, "0,0.0,$n0,$E0,0.0,$(1.0-n0),$chi0,0.0,0.0")
@printf("%4d  %5.2f  %.12f  %.8f  %+.3e  %+.3e  %4d  %.3e\n",
        0, 0.0, n0, E0, 0.0, 1.0-n0, chi0, 0.0)

for step in 1:n_steps
    res_r = tdvp_sweep(state, solver, opts, :right)
    res_l = tdvp_sweep(state, solver, opts, :left)

    t   = step * dt
    E   = measure_energy(state)
    nrm = real(measure_norm(state.mps))
    chi = max_bond(state.mps)
    mt  = max(res_r.max_trunc, res_l.max_trunc)
    tt  = res_r.total_trunc + res_l.total_trunc

    println(io, "$step,$t,$nrm,$E,$(E-E0),$(1.0-nrm),$chi,$mt,$tt")

    if step % 5 == 0 || step == n_steps
        @printf("%4d  %5.2f  %.12f  %.8f  %+.3e  %+.3e  %4d  %.3e\n",
                step, t, nrm, E, E-E0, 1.0-nrm, chi, mt)
    end
end
close(io)

println("\nDone. CSV written to $csv_path")
