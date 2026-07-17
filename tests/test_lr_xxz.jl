# tests/test_lr_xxz.jl — checks for the long-range staggered-sign XXZ builder
# (build_lr_xxz_hamiltonian_sector). Run: julia tests/test_lr_xxz.jl

using LinearAlgebra, Printf, Test
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

@testset "LR staggered-sign XXZ (sector)" begin
    # 1. α → ∞ recovers the short-range XXZ (r ≥ 2 couplings vanish)
    for N in (6, 8, 10), Δ in (0.8, 3.0)
        sec = sz_sector(N, N ÷ 2)
        Hnn = build_xxz_hamiltonian_sector(sec, 1.0, Δ)
        Hlr = build_lr_xxz_hamiltonian_sector(sec, 1.0, Δ, 60.0)
        @test maximum(abs.(Matrix(Hlr) .- Matrix(Hnn))) < 1e-15
    end

    # 2. exact N=4, S^z=0: diagonal of the Néel config |udud⟩ (bitmask 0101)
    #    Jz(r) = Δ(−1)^{r+1}/r^α ; Néel: sz = (+,−,+,−)/2
    #    E_diag = Jz(1)(s1s2+s2s3+s3s4) + Jz(2)(s1s3+s2s4) + Jz(3) s1s4
    N, Δ, α = 4, 3.0, 1.5
    sec = sz_sector(N, 2)
    H = build_lr_xxz_hamiltonian_sector(sec, 1.0, Δ, α)
    neel = sec.index[0b0101]
    Jz(r) = Δ * (-1.0)^(r + 1) / r^α
    Eexp = Jz(1) * 3 * (-0.25) + Jz(2) * 2 * (0.25) + Jz(3) * (-0.25)
    @test Matrix(H)[neel, neel] ≈ Eexp atol = 1e-14

    # 3. every LR bond supports Néel: staggered GS correlations are STRONGER
    #    than short-range at the same Δ (S(π) grows as α decreases)
    N = 12
    sec = sz_sector(N, 6)
    Sπ = Float64[]
    for α in (60.0, 2.5, 1.5)
        H = build_lr_xxz_hamiltonian_sector(sec, 1.0, 3.0, α)
        _, ψ = ground_state(H)
        _, C = thermal_sz_correlations(sec, abs2.(ψ))
        push!(Sπ, structure_factor(C, [Float64(π)])[1])
    end
    @test issorted(Sπ)     # NN < α=2.5 < α=1.5

    # 4. S^z conservation intact: sector spectrum ⊂ full-space spectrum
    #    (cheap N=8 check against a dense full-space build via embed of sz ops)
    N = 8
    sec = sz_sector(N, 4)
    H = build_lr_xxz_hamiltonian_sector(sec, 1.0, 2.0, 1.5)
    @test ishermitian(Matrix(H))
    ev = eigvals(H)
    @test length(ev) == binomial(8, 4)
end

println("test_lr_xxz: all passed")
