# tests/diagnostic_seed_integration.jl
#
# Goal: explain WHY the DMRG-seed dt-curve in fig2 is flat at ~2.8e-8 while the
# clean (ED→MPS) seed shows a clean dt^3 integration error reaching 4.4e-6 at
# dt=1 — even though both are (nearly) the same physical state evolved by the
# SAME 2-site TDVP integrator with the SAME parameters.
#
# The naive "errors add" model predicts the DMRG curve should lift off and
# track the clean curve once dt^3 exceeds the 2.8e-8 floor (dt >= 0.25). It does
# not. So the integration error is NOT seed-independent. This script decomposes
# why, with three controls:
#
#   (1) Confirm the TDVP parameters are byte-for-byte identical for both seeds.
#   (2) Report the retained bond spectra (dims + smallest singular value) of
#       each seed — does DMRG's cutoff regularize the Schmidt tail vs the exact
#       full-rank clean seed?
#   (3) The decisive decomposition. For each seed S and step dt, compare TDVP
#       against the EXACT ED evolution of S's OWN initial vector (not the exact
#       ground state). This separates two things the fig2 metric conflates:
#         • integ_total = state error ‖ψ_TDVP − e^{-iH t}ψ_S(0)‖  (both parities)
#         • sz_odd      = max_i|⟨S^z_i⟩_TDVP − ⟨S^z_i⟩_exact(S)|  (odd channel only)
#       If integ_total is the same for both seeds but sz_odd is not, the
#       integration error is the same SIZE and only its parity PROJECTION differs
#       by seed. If integ_total itself differs, the DMRG representation genuinely
#       integrates more accurately.
#
# A 4th control (B2) recompresses the exact (parity-pure) vector at the DMRG
# cutoff: same spectrum regularization as DMRG, but zero parity contamination.

using LinearAlgebra
using Printf

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

# ─── parameters (identical to diagnostic_dt_convergence.jl) ──────────────────
const N      = 12
const J      = 1.0
const d      = 2
const Δi     = 0.5
const Δf     = 2.0
const h      = zeros(N)
const χ      = 64                    # full Hilbert space: 2^(N/2)
const n_sweeps_dmrg = 40
const cutoff_dmrg   = 1e-12
const t_max         = 2.0
const cutoff_tdvp   = 0.0            # NO truncation → no projection error
const krylov_dim    = 30
const krylov_tol    = 1e-14

const DTS  = [1.0, 0.5, 0.25, 0.125]
const GRID = [1.0, 2.0]

ops = spin_ops(d)
const Sz_mat = Matrix{ComplexF64}(ops[:Z])

sz_profile_mps(mps) = [real(measure_local_observable(mps, Sz_mat, i)) for i in 1:N]

"""Exact ED vector → left-canonical MPS via successive SVDs, keeping only
singular values above a *relative* cutoff (cutoff=0 keeps full rank)."""
function vector_to_mps(psi::AbstractVector, N::Int; d::Int=2, cutoff::Float64=0.0)
    @assert length(psi) == d^N
    tensors = Vector{Array{ComplexF64,3}}(undef, N)
    psi_mat = reshape(Vector{ComplexF64}(psi), (1, d^N))
    chi_l = 1
    for i in 1:N-1
        remaining = d^(N - i)
        psi_mat = reshape(psi_mat, (chi_l * d, remaining))
        F = svd(psi_mat)
        keep = cutoff > 0 ? max(count(F.S ./ norm(F.S) .>= cutoff), 1) : length(F.S)
        chi_r = keep
        tensors[i] = reshape(F.U[:, 1:chi_r], (chi_l, d, chi_r))
        psi_mat   = Diagonal(F.S[1:chi_r]) * F.Vt[1:chi_r, :]
        chi_l = chi_r
    end
    tensors[N] = reshape(psi_mat, (chi_l, d, 1))
    return MPS{ComplexF64}(tensors)
end

function run_dmrg(mps_init, mpo, χ, n_sweeps, cutoff)
    state = MPSState(mps_init, mpo; center=1)
    solver = LanczosSolver(4, 14)
    opts   = DMRGOptions(χ, cutoff, d)
    local res
    for sweep in 1:n_sweeps
        dir = isodd(sweep) ? :right : :left
        res = dmrg_sweep(state, solver, opts, dir)
    end
    return state, res.E
end

"""Retained bond spectrum: dims and smallest singular value per bond, read off
the actual MPS representation (gauge-invariant Schmidt values)."""
function bond_report(mps)
    m = deepcopy(mps)
    make_canonical(m, 1)                       # site 1 = center, 2..N right-orth
    dims = Int[]; mins = Float64[]
    # sweep the center rightward, SVD each bond, record singular values
    for i in 1:N-1
        chi_l, di, chi_r = size(m.tensors[i])
        F = svd(reshape(m.tensors[i], (chi_l*di, chi_r)))
        push!(dims, length(F.S)); push!(mins, minimum(F.S))
        k = length(F.S)
        m.tensors[i]   = reshape(F.U, (chi_l, di, k))
        SV = Diagonal(F.S) * F.Vt                  # (k, chi_r)
        cl, ds, cr = size(m.tensors[i+1])
        nxt = SV * reshape(m.tensors[i+1], (cl, ds*cr))   # (k, ds*cr)
        m.tensors[i+1] = reshape(nxt, (k, ds, cr))
    end
    return dims, mins
end

"""Evolve a copy of `mps_init` with 2-site TDVP at step `dt`; return ⟨S^z⟩
profiles AND full state vectors on GRID. Identical TDVP config for every seed."""
function tdvp_on_grid(mps_init, mpo_f, dt)
    state  = MPSState(deepcopy(mps_init), mpo_f; center=1)
    solver = KrylovExponential(krylov_dim, krylov_tol, "real")
    opts   = TDVPOptions(dt, χ, cutoff_tdvp, d)

    profiles = Dict{Float64,Vector{Float64}}()
    vectors  = Dict{Float64,Vector{ComplexF64}}()
    nsteps = round(Int, t_max / dt)
    for step in 1:nsteps
        tdvp_sweep(state, solver, opts, :right)
        tdvp_sweep(state, solver, opts, :left)
        t = step * dt
        for g in GRID
            if isapprox(t, g; atol=1e-9)
                profiles[g] = sz_profile_mps(state.mps)
                v = mps_to_vector(state.mps; d=d); v ./= norm(v)
                vectors[g] = v
            end
        end
    end
    return profiles, vectors, solver, opts
end

infidelity_dist(a, b) = sqrt(max(0.0, 2*(1 - abs(dot(a, b)))))   # ≈ ‖a − e^{iφ}b‖

# ─── setup ──────────────────────────────────────────────────────────────────
println(repeat("=", 78))
println("  Seed-resolved TDVP integration error (N=$N, χ=$χ, h=0, Δi=$Δi→Δf=$Δf)")
println(repeat("=", 78))

mpo_i = build_xxz_mpo(N, J, Δi, h; d=d)
mpo_f = build_xxz_mpo(N, J, Δf, h; d=d)
H_i   = build_xxz_hamiltonian(N, J, Δi, h; d=d)

eig_i = diagonalize(H_i)
ψ_ED  = Vector{ComplexF64}(eig_i.vectors[:, 1]); ψ_ED ./= norm(ψ_ED)
H_f   = build_xxz_hamiltonian(N, J, Δf, h; d=d)
eig_f = diagonalize(H_f)

# ─── seeds ────────────────────────────────────────────────────────────────────
sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels_A = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
state_A, E_DMRG = run_dmrg(product_state(sites, labels_A), mpo_i, χ, n_sweeps_dmrg, cutoff_dmrg)
seedA = state_A.mps                                   # DMRG ground state
seedB = vector_to_mps(ψ_ED, N; d=d)                   # exact, full rank
# exact vector, recompressed at cutoffs that ACTUALLY truncate the small-λ tail
# (the global λ_min is ~7e-11, so 1e-12 is a no-op but 1e-9/1e-7/1e-5 bite).
seedB3 = vector_to_mps(ψ_ED, N; d=d, cutoff=1e-9)
seedB4 = vector_to_mps(ψ_ED, N; d=d, cutoff=1e-7)
seedB5 = vector_to_mps(ψ_ED, N; d=d, cutoff=1e-5)
# round-trip the DMRG state through the SAME raw SVD construction as B:
# isolates representation/gauge from the physical vector.
seedAp = vector_to_mps(mps_to_vector(seedA; d=d), N; d=d)

SEEDS = [("A: DMRG", seedA), ("B: ED→MPS full", seedB),
         ("B3: ED→MPS @1e-9", seedB3), ("B4: ED→MPS @1e-7", seedB4),
         ("B5: ED→MPS @1e-5", seedB5), ("A': DMRG→vec→MPS", seedAp)]

# ─── (1) parameter identity ──────────────────────────────────────────────────
println("\n[1] TDVP config (SAME object construction for every seed):")
println("    KrylovExponential($krylov_dim, $krylov_tol, \"real\")")
println("    TDVPOptions(dt, χ=$χ, cutoff=$cutoff_tdvp, d=$d)   ·   mpo_f shared")

# ─── (2) retained bond spectra ───────────────────────────────────────────────
println("\n[2] Retained bond spectrum (gauge-invariant Schmidt values):")
@printf "    %-22s %-12s %-12s %-s\n" "seed" "global λ_min" "⟨S^z⟩(t=0)" "bond dims"
for (name, s) in SEEDS
    dims, mins = bond_report(s)
    @printf "    %-22s %-12.3e %-12.3e %s\n" name minimum(mins) maximum(abs.(sz_profile_mps(s))) string(dims)
end

# ─── (3) seed-resolved integration error ─────────────────────────────────────
println("\n[3] TDVP vs EXACT evolution of each seed's OWN initial vector:")
println("    sz_vsED   = max_i|⟨S^z⟩_TDVP|                 (the fig2 quantity; ED gs ⟨S^z⟩=0)")
println("    sz_self   = max_i|⟨S^z⟩_TDVP − ⟨S^z⟩_exact(seed)|  (odd-channel integ error)")
println("    integ_tot = max_t ‖ψ_TDVP − e^{-iH_f t}ψ_seed(0)‖  (TOTAL integ error, both parities)")

for (name, s) in SEEDS
    ψ0 = mps_to_vector(s; d=d); ψ0 ./= norm(ψ0)
    # exact ED evolution of THIS seed's own initial vector
    ref_sz  = Dict(g => ed_local_profile(ed_time_evolve(eig_f, ψ0, g), Sz_mat, N; d=d) for g in GRID)
    ref_vec = Dict(g => (v = ed_time_evolve(eig_f, ψ0, g); v ./ norm(v)) for g in GRID)

    println("\n  ── $name ──")
    @printf "    %-8s  %-12s  %-12s  %-12s\n" "dt" "sz_vsED" "sz_self" "integ_tot"
    for dt in DTS
        prof, vecs, _, _ = tdvp_on_grid(s, mpo_f, dt)
        sz_vsED  = maximum(maximum(abs.(prof[g]))            for g in GRID)
        sz_self  = maximum(maximum(abs.(prof[g] .- ref_sz[g])) for g in GRID)
        integ    = maximum(infidelity_dist(vecs[g], ref_vec[g]) for g in GRID)
        @printf "    %-8.4f  %-12.3e  %-12.3e  %-12.3e\n" dt sz_vsED sz_self integ
    end
end

println("\n" * repeat("=", 78))
println("  Read: if integ_tot matches across seeds but sz_self/sz_vsED do not,")
println("  the integration error is the same SIZE and only its parity projection")
println("  differs. If integ_tot itself is smaller for DMRG, the representation")
println("  integrates more accurately. B2 isolates spectrum-regularization from")
println("  parity contamination.")
println(repeat("=", 78))
