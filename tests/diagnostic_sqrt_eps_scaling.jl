# tests/diagnostic_sqrt_eps_scaling.jl
#
# DECISIVE test of the claimed source of the DMRG <S^z> floor at h=0.
#
# Claim: the floor is the *variational* relation between energy error and state
# error. At a minimum, E - E0 ≈ g·‖δ‖² (quadratic), while ⟨S^z_i⟩ ~ ‖δ‖ (linear).
# Therefore the parity-odd staggered residue should obey
#       max_i |⟨S^z_i⟩|  ∝  sqrt( E_DMRG - E_ED ),   i.e. slope 1/2 on log-log.
#
# We get many (ΔE, |⟨S^z⟩|) points from ONE DMRG run by recording after every
# sweep: early sweeps are poorly converged (large ΔE), late sweeps tight (small
# ΔE). If the points fall on a slope-1/2 line over several decades, the floor IS
# the variational tail of the energy error — not a coding bug, not truncation.
#
# Distinguishing accumulation vs variational ceiling: we also report the floor
# per-sweep once ΔE has plateaued. If ⟨S^z⟩ keeps climbing while ΔE is flat,
# that is sweep-accumulation on top of the variational floor.

using LinearAlgebra, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

const N, J, d = 12, 1.0, 2
const Δi = 0.5
const h  = zeros(N)
const χ  = 64
const cutoff_dmrg = 1e-12
const N_SWEEPS = 60

ops = spin_ops(d)
const Sz_mat = Matrix{ComplexF64}(ops[:Z])
sz_profile_mps(mps) = [real(measure_local_observable(mps, Sz_mat, i)) for i in 1:N]

println(repeat("=", 70))
println("  √ε scaling test: DMRG ⟨S^z⟩ floor vs energy error   (N=$N, χ=$χ, h=0)")
println(repeat("=", 70))

H_i   = build_xxz_hamiltonian(N, J, Δi, h; d=d)
mpo_i = build_xxz_mpo(N, J, Δi, h; d=d)
E_ED  = diagonalize(H_i).values[1]

sites  = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
mps_A0 = product_state(sites, [isodd(i) ? (:Z,1) : (:Z,2) for i in 1:N])
state  = MPSState(mps_A0, mpo_i; center=1)
solver = LanczosSolver(4, 14)
opts   = DMRGOptions(χ, cutoff_dmrg, d)

dE   = Float64[]
szmx = Float64[]
@printf "\n  %-5s  %-14s  %-14s  %-s\n" "sweep" "ΔE=E-E_ED" "max|⟨S^z⟩|" "running slope"
for sweep in 1:N_SWEEPS
    res = dmrg_sweep(state, solver, opts, isodd(sweep) ? :right : :left)
    ΔE  = res.E - E_ED
    smx = maximum(abs.(sz_profile_mps(state.mps)))
    push!(dE, ΔE); push!(szmx, smx)
    if ΔE > 0 && length(dE) >= 2 && dE[end-1] > 0 && dE[end] != dE[end-1]
        slope = log(szmx[end]/szmx[end-1]) / log(dE[end]/dE[end-1])
        @printf "  %-5d  %-14.3e  %-14.3e  %.2f\n" sweep ΔE smx slope
    else
        @printf "  %-5d  %-14.3e  %-14.3e  %s\n" sweep ΔE smx "—"
    end
end

# Global least-squares slope over the regime where ΔE is a clean signal
# (ΔE > 10·machine floor, so we are above the energy plateau).
mask = dE .> 1e-13
if count(mask) >= 2
    x = log10.(dE[mask]); y = log10.(szmx[mask])
    A = hcat(x, ones(length(x)))
    coef = A \ y
    println(repeat("─", 70))
    @printf "  Least-squares slope (over ΔE > 1e-13, %d pts):  %.3f\n" count(mask) coef[1]
    @printf "  Prediction if floor = variational √ΔE:          0.500\n"
    @printf "  Plateau floor (last 10 sweeps): ΔE ~ %.1e,  max|⟨S^z⟩| ~ %.1e\n" dE[end] szmx[end]
end
println(repeat("=", 70))
