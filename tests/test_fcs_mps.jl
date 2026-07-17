# tests/test_fcs_mps.jl — MPS generating-function FCS (Analysis/fcs.jl)
# against exact references:
#   1. Néel product state → P(m = −ℓ/2) = 1 exactly
#   2. random MPS vs brute-force histogram of the full state vector
#      (even AND odd window sizes, off-center window)
#   3. β = 0 purification (embed path) → binomial distribution analytically
#   4. DMRG ground state (N=12, Δ=3.0) vs the exact ED sector ground state
# Run: julia tests/test_fcs_mps.jl

using LinearAlgebra, Printf, Test
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

@testset "MPS staggered FCS" begin
    # ── 1. Néel |↑↓↑↓…⟩: (−1)^i S^z_i = −1/2 on every site ⇒ m ≡ −ℓ/2 ──
    N = 10
    sites_ = [SpinSite(0.5; T = ComplexF64) for _ in 1:N]
    neel = product_state(sites_, [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N])
    for win in (3:6, 2:9)
        mg, P = staggered_fcs(neel, collect(win))
        @test P[1] ≈ 1.0 atol = 1e-12          # m = −ℓ/2 is mgrid[1]
        @test sum(P) ≈ 1.0 atol = 1e-12
    end

    # ── 2. random MPS vs exact histogram of the full vector ──────────────
    mps = random_state(sites_, 8)              # unnormalized, complex
    ψ = mps_to_vector(mps)
    w = abs2.(ψ) ./ sum(abs2.(ψ))
    fs = full_sz_values(N)
    for win in (4:7, 3:7, 2:9)                 # ℓ = 4 (even), 5 (odd), 8
        sitesv = collect(win)
        msv = fs[:, sitesv] * [(-1.0)^i for i in sitesv]
        mg, P = staggered_fcs(mps, sitesv)
        Pex = zeros(length(mg))
        for (k, m) in enumerate(msv)
            Pex[round(Int, m - mg[1]) + 1] += w[k]
        end
        @test maximum(abs.(P .- Pex)) < 1e-12
        @test sum(P) ≈ 1.0 atol = 1e-12
    end

    # ── 3. purification at β = 0: independent spins ⇒ binomial ───────────
    pur = maximally_mixed_purification(12)
    win = 3:10                                  # ℓ = 8
    mg, P = staggered_fcs(pur, collect(win); embed = true)
    Pbin = [binomial(8, Int(m + 4)) / 2^8 for m in mg]
    @test maximum(abs.(P .- Pbin)) < 1e-12

    # ── 4. DMRG GS (N=12, Δ=3.0, staggered h) vs exact ED sector GS ──────
    # A small staggered field lifts the Néel doublet so the GS is unique on
    # both sides (at h=0 DMRG may pick an arbitrary doublet superposition);
    # it also makes P(m) asymmetric — a stricter test of the inversion.
    N12 = 12
    hstag = [0.05 * (-1.0)^i for i in 1:N12]
    mpo = build_xxz_mpo(N12, 1.0, 3.0, hstag)
    s12 = [SpinSite(0.5; T = ComplexF64) for _ in 1:N12]
    mps0 = product_state(s12, [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N12])
    state = MPSState(mps0, mpo; center = 1)
    solver = LanczosSolver(4, 14)
    opts = DMRGOptions(64, 1e-10, 2)
    local E
    for sweep in 1:16
        E = dmrg_sweep(state, solver, opts, isodd(sweep) ? :right : :left).E
    end
    sec = sz_sector(N12, 6)
    E0, ψ0 = ground_state(build_xxz_hamiltonian_sector(sec, 1.0, 3.0, hstag))
    @test abs(E - E0) < 1e-8
    msv, mgE = staggered_values(sec; sites = 3:10)
    P_ed = staggered_fcs(msv, mgE, abs2.(ψ0))
    _, P_mps = staggered_fcs(state.mps, collect(3:10))
    @printf("    DMRG-vs-ED GS FCS max dev: %.2e\n", maximum(abs.(P_mps .- P_ed)))
    @test maximum(abs.(P_mps .- P_ed)) < 1e-6
end

@testset "general-operator FCS" begin
    N = 8
    sites_ = [SpinSite(0.5; T = ComplexF64) for _ in 1:N]
    mps = random_state(sites_, 8)
    ψ = mps_to_vector(mps)
    w = abs2.(ψ) ./ sum(abs2.(ψ))
    fs = full_sz_values(N)
    win = collect(3:6)
    ops_qd = spin_ops(2)

    # 1. uniform S^z window sum (diagonal op) vs brute-force histogram
    mg, P = fcs_distribution(mps, win, Matrix{Float64}(ops_qd[:Z]))
    msv = vec(sum(fs[:, win]; dims = 2))
    Pex = zeros(length(mg))
    for (k, m) in enumerate(msv)
        Pex[round(Int, (m - mg[1]))+1] += w[k]
    end
    @test maximum(abs.(P .- Pex)) < 1e-12

    # 2. transverse channel: A = Σ_{i∈w} S^x_i (NON-diagonal insertions) vs
    #    exact spectral decomposition of the full-space window operator
    Sx = Matrix{ComplexF64}(ops_qd[:X])
    Id = Matrix{ComplexF64}(I, 2, 2)
    M = zeros(ComplexF64, 2^N, 2^N)
    for i in win
        M .+= kron([j == i ? Sx : Id for j in 1:N]...)
    end
    F = eigen(Hermitian(M))
    mgx, Px = fcs_distribution(mps, win, Sx)
    Pex = zeros(length(mgx))
    for (j, λ) in enumerate(F.values)
        Pex[round(Int, λ - mgx[1])+1] += abs2(dot(F.vectors[:, j], ψ)) / sum(abs2.(ψ))
    end
    @test maximum(abs.(Px .- Pex)) < 1e-12

    # 3. per-site operator vector: general path ≡ staggered wrapper
    Szm = Matrix{Float64}(ops_qd[:Z])
    _, Pgen = fcs_distribution(mps, win, [(-1.0)^i * Szm for i in win])
    _, Pstag = staggered_fcs(mps, win)
    @test Pgen == Pstag

    # 4. staggered fcs_summary keeps the ordered_weight alias
    r = fcs_summary(mps, win)
    @test r.ordered_weight == r.extreme_weight == r.P[1] + r.P[end]
end

println("test_fcs_mps: all passed")
