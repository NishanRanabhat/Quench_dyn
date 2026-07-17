# tests/benchmark_fcs_n16.jl — end-to-end N=16 benchmark of the MPS
# generating-function FCS (Analysis/fcs.jl) against exact ED, on BOTH
# production paths:
#
#   A. FINITE T: purification cooled to β_eff by 2-site imaginary-time TDVP
#      (the run_finite_T.jl pipeline; 2-site because the β=0 seed has χ=1),
#      thermal FCS via staggered_fcs(..., embed=true)
#      vs the exact FULL-ENSEMBLE ED thermal distribution — built by
#      diagonalizing every S^z sector (spin-flip symmetry maps sector k to
#      16−k, so only k = 0…8 are computed; k = 8 is the cached one) and
#      Boltzmann-weighting the sector FCS with a common energy shift.
#      NOTE the purification is the FULL canonical ensemble; the S^z=0
#      SECTOR ensemble of the quench differs from it at finite N (the
#      ensemble offset is printed — physics, not error; see the ISSUE-1
#      note in the finite-T protocol doc).
#
#   B. REAL TIME: quench Δi=0.8 → Δf=3.0 (h=0), DMRG ground state → pad to
#      the FULL N=16 manifold (χ=256, so 1-site TDVP has zero projection
#      error and the only error is the O(dt²) sweep splitting + Krylov tol)
#      → 1-site TDVP to t=20, FCS(t) vs the exact ED quench distribution.
#
# Window: centered ℓ=8 (sites 5–12), matching docs/ed_analysis/. Figure with
# the overlay written to docs/ed_analysis/fcs_mps_benchmark_n16.{png,pdf}.
#
# Run: julia tests/benchmark_fcs_n16.jl   (~15–30 min, dominated by part B)

using LinearAlgebra, Printf
ENV["GKSwstype"] = "100"
using Plots; gr()
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

const N, J, Δi, Δf = 16, 1.0, 0.8, 3.0
const d   = 2
const win = collect(5:12)                       # ℓ = 8

# ── exact ED references from the spectrum cache ───────────────────────────
S   = load_spectrum(N, Δf)
gsi = load_initial_state(N)
c0  = quench_amplitudes(S.vectors, gsi.psi)
βe  = effective_beta(S.values, dot(c0 .^ 2, S.values))
msv, mgE = staggered_values(S.sec; sites = win)
P_ed_th = staggered_fcs(msv, mgE, diagonal_probabilities(S.vectors,
                            thermal_weights(S.values, βe)))
ts_meas = collect(0.0:1.0:20.0)
Pq = quench_probabilities(S.vectors, S.values, c0, ts_meas)
P_ed_t = [staggered_fcs(msv, mgE, @view Pq[:, k]) for k in eachindex(ts_meas)]
E_th_sector = thermal_energy(S.values, βe)
S = Pq = nothing; GC.gc()
@printf("ED sector reference ready: β_eff = %.6f (T_eff = %.4f)\n", βe, 1 / βe)

# full-ensemble thermal reference: all S^z sectors, spin-flip symmetry k↔N−k
# (identical spectra; configs map by bit complement ⇒ m → −m in the FCS)
println("building FULL-ensemble ED thermal reference (sectors k = 0…8) ...")
sector_E  = Vector{Vector{Float64}}(undef, 9)      # eigenvalues, k = 0…8
sector_P  = Vector{Vector{Float64}}(undef, 9)      # UNNORMALIZED sector FCS
E0_global = gsi.energy                              # any common shift works
for k in 0:8
    seck = sz_sector(N, k)
    Hk   = build_xxz_hamiltonian_sector(seck, J, Δf)
    Fk   = eigen!(Hk)
    sector_E[k+1] = Fk.values
    wk = exp.(-βe .* (Fk.values .- E0_global))      # common shift ⇒ relative Z_k
    pk = diagonal_probabilities(real.(Fk.vectors), wk ./ sum(wk))
    msvk, _ = staggered_values(seck; sites = win)
    sector_P[k+1] = sum(wk) .* staggered_fcs(msvk, mgE, pk)
    Fk = nothing; GC.gc()
end
Zs = [sum(exp.(-βe .* (sector_E[k+1] .- E0_global))) for k in 0:8]
Zfull = sum(Zs[1:8]) * 2 + Zs[9]                    # k and N−k pair up; k=8 once
P_ed_full = (sum(sector_P[k+1] .+ reverse(sector_P[k+1]) for k in 0:7) .+
             sector_P[9]) ./ Zfull
E_th_full = (2 * sum(sum(E .* exp.(-βe .* (E .- E0_global))) for E in sector_E[1:8]) +
             sum(sector_E[9] .* exp.(-βe .* (sector_E[9] .- E0_global)))) / Zfull
@printf("full ensemble at β_eff: ⟨H⟩ = %.8f (sector ⟨H⟩ = %.8f — the finite-N ensemble offset)\n\n",
        E_th_full, E_th_sector)

# ═══ A. purification → thermal FCS ═════════════════════════════════════════
println("─"^70)
println("  A. purification cooling to β_eff (2-site imaginary-time TDVP)")
println("─"^70)

n_cool = 30
dτ     = (βe / 2) / n_cool                      # β = 2·n_cool·dτ = β_eff exactly
mpo_pur = purify_mpo(build_xxz_mpo(N, J, Δf, zeros(N); d = d); d = d)
pur     = maximally_mixed_purification(N; d = d)
state_p = MPSState(pur, mpo_pur; center = 1)
solver_p = KrylovExponential(14, 1e-10, "imaginary")
opts_p   = TDVPOptions(dτ, 256, 1e-12, d * d)

for step in 1:n_cool
    tdvp_sweep(state_p, solver_p, opts_p, :right)
    tdvp_sweep(state_p, solver_p, opts_p, :left)
    state_p.mps.tensors[state_p.center] ./= sqrt(measure_norm(state_p.mps))
end
Zp  = measure_norm(state_p.mps)
E_p = measure_energy(state_p) / Zp
χp  = maximum(size(t, 3) for t in state_p.mps.tensors[1:end-1])
@printf("  cooled to β = %.6f : χ_max = %d, ⟨H⟩ = %.8f (full-ens exact %.8f, dev %.2e)\n",
        2 * n_cool * dτ, χp, E_p, E_th_full, abs(E_p - E_th_full))

_, P_mps_th = staggered_fcs(state_p.mps, win; embed = true)
dev_th = maximum(abs.(P_mps_th .- P_ed_full))
@printf("  THERMAL FCS max |P_mps − P_ed(FULL ensemble)|  = %.3e   ← the benchmark\n", dev_th)
@printf("  ensemble offset |P_ed(full) − P_ed(sector)|max = %.3e   ← physics, not error\n\n",
        maximum(abs.(P_ed_full .- P_ed_th)))

# ═══ B. quench → FCS(t) via 1-site TDVP ════════════════════════════════════
println("─"^70)
println("  B. quench 0.8 → 3.0: DMRG → pad to full manifold → 1-site TDVP")
println("─"^70)

sites_ = [SpinSite(0.5; T = ComplexF64) for _ in 1:N]
mps0 = product_state(sites_, [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N])
state = MPSState(mps0, build_xxz_mpo(N, J, Δi, zeros(N); d = d); center = 1)
opts_d = DMRGOptions(128, 1e-12, d)
solver_d = LanczosSolver(4, 14)
E_gs = 0.0
for sweep in 1:24
    global E_gs = dmrg_sweep(state, solver_d, opts_d, isodd(sweep) ? :right : :left).E
end
@printf("  DMRG GS: E = %.10f (ED %.10f, dev %.2e), var = %.2e\n",
        E_gs, gsi.energy, abs(E_gs - gsi.energy), energy_variance(state))

padded = pad_mps(state.mps, 256; noise = 1e-7)   # full manifold at N=16
state = MPSState(padded, build_xxz_mpo(N, J, Δf, zeros(N); d = d); center = 1)
solver_t = KrylovExponential(14, 1e-10, "real")
opts_t   = TDVPOptions(0.05, 256, 1e-12, d)
n_per    = round(Int, 1.0 / 0.05)                # measurements at integer t

E0 = measure_energy(state)
dev_t = zeros(length(ts_meas))
ow_mps = zeros(length(ts_meas)); ow_ed = zeros(length(ts_meas))
_, P0 = staggered_fcs(state.mps, win)
dev_t[1] = maximum(abs.(P0 .- P_ed_t[1]))
ow_mps[1], ow_ed[1] = P0[1] + P0[end], P_ed_t[1][1] + P_ed_t[1][end]
@printf("  t = %4.1f : FCS dev %.3e\n", 0.0, dev_t[1])

P20_mps = Float64[]
for k in 2:length(ts_meas)
    for _ in 1:n_per
        tdvp_sweep_one_site(state, solver_t, opts_t, :right)
        tdvp_sweep_one_site(state, solver_t, opts_t, :left)
    end
    _, P = staggered_fcs(state.mps, win)
    dev_t[k] = maximum(abs.(P .- P_ed_t[k]))
    ow_mps[k], ow_ed[k] = P[1] + P[end], P_ed_t[k][1] + P_ed_t[k][end]
    k == length(ts_meas) && (global P20_mps = P)
    @printf("  t = %4.1f : FCS dev %.3e   (E drift %.2e, 1−‖ψ‖² %.2e)\n",
            ts_meas[k], dev_t[k], measure_energy(state) - E0,
            1 - measure_norm(state.mps))
end
@printf("\n  REAL-TIME FCS max dev over all (m, t) = %.3e\n", maximum(dev_t))

# ═══ figure ════════════════════════════════════════════════════════════════
p1 = plot(mgE, P_ed_full; marker = :circle, color = :black,
          label = "ED thermal (full ens.)", xlabel = "m", ylabel = "P(m)",
          title = "thermal at β_eff")
plot!(p1, mgE, P_ed_th; marker = :utriangle, color = :gray60, ls = :dot,
      label = "ED thermal (S^z=0 sector)")
plot!(p1, mgE, P_mps_th; marker = :xcross, ms = 6, ls = :dash, color = :firebrick,
      label = "purification FCS")
p2 = plot(mgE, P_ed_t[end]; marker = :circle, color = :black, label = "ED t=20",
          xlabel = "m", ylabel = "P(m)", title = "quench at t = 20")
plot!(p2, mgE, P20_mps; marker = :xcross, ms = 6, ls = :dash, color = :firebrick,
      label = "1-site TDVP FCS")
p3 = plot(ts_meas, ow_ed; color = :black, marker = :circle, label = "ED",
          xlabel = "t [1/J]", ylabel = "P(|m|=4)", title = "ordered weight vs t")
plot!(p3, ts_meas, ow_mps; color = :firebrick, marker = :xcross, ms = 6,
      ls = :dash, label = "MPS")
p4 = plot(ts_meas, max.(dev_t, 1e-12); yscale = :log10, marker = :circle,
          color = :firebrick, label = "", xlabel = "t [1/J]",
          ylabel = "max_m |ΔP|", title = "FCS deviation vs ED")
fig = plot(p1, p2, p3, p4; layout = (2, 2), size = (1100, 750),
           plot_title = "MPS FCS benchmark vs ED, N=16 (window ℓ=8)",
           left_margin = 5Plots.mm)
outdir = normpath(joinpath(@__DIR__, "..", "docs", "ed_analysis"))
for ext in ("png", "pdf")
    savefig(fig, joinpath(outdir, "fcs_mps_benchmark_n16.$ext"))
end
println("  wrote docs/ed_analysis/fcs_mps_benchmark_n16.{png,pdf}")

println("\nSUMMARY: thermal dev = $(round(dev_th, sigdigits=3)), " *
        "real-time dev = $(round(maximum(dev_t), sigdigits=3))")
