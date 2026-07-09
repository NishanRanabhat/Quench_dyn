# tests/test_finite_T_ed.jl
#
# Validation of the S^z-sector block Hamiltonian and finite-temperature layer.
# Run:  julia --project -e 'include("tests/test_finite_T_ed.jl")'

using LinearAlgebra
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn
using Test

@testset "S^z sector block Hamiltonian" begin
    # sector spectrum must be a subset of the full-space spectrum
    for (N, Δ) in ((6, 3.0), (8, 3.0), (8, 0.8))
        J = 1.0
        h = 0.1 .* randn(N)                       # generic field: no accidental degen structure
        Hfull = build_xxz_hamiltonian(N, J, Δ, h)
        full_vals = sort(eigvals(Hfull))

        # union of all magnetization sectors must recover the full spectrum
        sector_vals = Float64[]
        for n_up in 0:N
            sec = sz_sector(N, n_up)
            Hs = build_xxz_hamiltonian_sector(sec, J, Δ, h)
            append!(sector_vals, eigvals(Hs))
        end
        sort!(sector_vals)
        @test length(sector_vals) == length(full_vals)
        @test maximum(abs.(sector_vals .- full_vals)) < 1e-9
    end
end

@testset "S^z=0 sector: dimension & GS energy" begin
    N, J, Δ = 8, 1.0, 3.0
    sec = sz_sector(N, N ÷ 2)
    @test length(sec) == binomial(N, N ÷ 2)              # 70
    Hs = build_xxz_hamiltonian_sector(sec, J, Δ)          # zero field
    Hfull = build_xxz_hamiltonian(N, J, Δ)
    @test minimum(eigvals(Hs)) ≈ minimum(eigvals(Hfull)) atol=1e-10   # AFM GS lives in S^z=0
end

@testset "thermal machinery" begin
    N, J, Δ = 8, 1.0, 3.0
    sec = sz_sector(N, N ÷ 2)
    Hf = build_xxz_hamiltonian_sector(sec, J, Δ)
    eig = diagonalize(Hf)
    evals = eig.values

    # (a) ρ_diag is a probability vector
    for β in (0.0, 0.5, 2.0, -0.5)
        ρ = thermal_diagonal(eig, β)
        @test all(ρ .>= -1e-14)
        @test sum(ρ) ≈ 1 atol=1e-12
    end

    # (b) energy matching round-trips
    E_target = evals[1] + 0.3 * (sum(evals)/length(evals) - evals[1])  # between GS and mean
    β = effective_beta(evals, E_target)
    @test thermal_energy(evals, β) ≈ E_target atol=1e-9
    @test β > 0                                            # below the mean ⇒ positive T

    # (c) β → ∞ recovers the ground-state correlations
    ρcold = thermal_diagonal(eig, 1e4)
    m_c, C_c = thermal_sz_correlations(sec, ρcold)
    _, ψ0 = ground_state(Hf)
    # GS correlations in the sector basis: ⟨ψ0|S^z_j S^z_k|ψ0⟩ = Σ_c |ψ0[c]|² sz sz
    p0 = abs2.(ψ0)
    m_gs = transpose(sec.sz) * p0
    C_gs = transpose(sec.sz) * (p0 .* sec.sz)
    @test maximum(abs.(C_c .- C_gs)) < 1e-6
    @test maximum(abs.(m_c .- m_gs)) < 1e-6

    # (d) m ≡ 0 in the S^z=0 sector at any temperature
    for β in (0.0, 1.0, 5.0)
        m, _ = thermal_sz_correlations(sec, thermal_diagonal(eig, β))
        @test maximum(abs.(m)) < 1e-12
    end

    # (e) infinite-T diagonal is uniform over the sector
    ρ_inf = thermal_diagonal(eig, 0.0)
    @test maximum(abs.(ρ_inf .- 1/length(sec))) < 1e-12
end

println("all finite-T ED tests passed")
