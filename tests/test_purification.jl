# tests/test_purification.jl
#
# Validate the ancilla-purification finite-T path (Approach A) against exact
# full-Hilbert-space ED thermal averages. This certifies:
#   - the β=0 seed and H⊗I embedding (energy at infinite T = spectrum mean),
#   - the τ↔β convention (β = 2τ) via ⟨H⟩(β) matching ED across a β range,
#   - the thermal S^z one/two-point observables (m, C) and ξ vs ED.
#
# Run:  julia -e 'include("tests/test_purification.jl")'

using LinearAlgebra
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn
using Test

@testset "purification primitives" begin
    N, d = 6, 2
    ψ = maximally_mixed_purification(N; d = d)
    @test length(ψ.tensors) == N
    @test size(ψ.tensors[1]) == (1, d * d, 1)
    @test measure_norm(ψ) ≈ 1 atol = 1e-12          # Tr ρ(β=0) = 1

    # β=0 energy must equal the full-spectrum mean (Tr H / d^N)
    J, Δ = 1.0, 1.5
    Hph = build_xxz_mpo(N, J, Δ, zeros(N); d = d, T = Float64)
    Hpu = purify_mpo(Hph; d = d)
    @test size(Hpu.tensors[2], 3) == d * d           # physical leg enlarged
    @test size(Hpu.tensors[2], 1) == size(Hph.tensors[2], 1)   # bond dim unchanged

    state0 = MPSState(deepcopy(ψ), Hpu; center = 1)
    E0 = measure_energy(state0) / measure_norm(state0.mps)
    evals = eigvals(build_xxz_hamiltonian(N, J, Δ, zeros(N)))
    @test E0 ≈ sum(evals) / length(evals) atol = 1e-10
end

@testset "purification vs full ED thermal" begin
    N, d, J, Δ = 8, 2, 1.0, 1.5
    Hph = build_xxz_mpo(N, J, Δ, zeros(N); d = d, T = Float64)
    Hpu = purify_mpo(Hph; d = d)
    ψ0  = maximally_mixed_purification(N; d = d)
    state = MPSState(ψ0, Hpu; center = 1)

    solver = KrylovExponential(24, 1e-11, "imaginary")
    dt     = 0.02
    opts   = TDVPOptions(dt, 64, 1e-12, d * d)       # χ=64 exact for N=8, local dim 4

    βmax = 2.0
    res = cool_purification!(state, solver, opts; target_beta = βmax, record = true)

    # exact ED reference (full Hilbert space, all sectors)
    eig = diagonalize(build_xxz_hamiltonian(N, J, Δ, zeros(N)))
    szf = full_sz_values(N; d = d)

    # (a) ⟨H⟩(β) trace matches ED across the whole cooling range → τ↔β convention
    devE = maximum(abs(E - thermal_energy(eig.values, β)) for (β, E) in zip(res.βs, res.Es))
    @test devE < 5e-3

    # (b) thermal (m, C) at β = βmax match ED
    m_p, C_p = thermal_sz_from_purification(state; d = d)
    ρ = thermal_diagonal(eig, res.β)
    m_e, C_e = thermal_sz_correlations(szf, ρ)
    @test maximum(abs.(m_p .- m_e)) < 5e-3
    @test maximum(abs.(C_p .- C_e)) < 5e-3

    # (c) staggered correlator and ξ agree
    bulk = collect(2:N-1)
    _, Gp = staggered_correlator(C_p; rmax = N ÷ 2, bulk = bulk)
    _, Ge = staggered_correlator(C_e; rmax = N ÷ 2, bulk = bulk)
    @test maximum(abs.(Gp .- Ge)) < 5e-3
    @test domain_length(0:N÷2, Gp) ≈ domain_length(0:N÷2, Ge) atol = 5e-2

    println("  β reached = $(round(res.β, digits=4)), max χ = $(res.max_chi), " *
            "max |ΔE(β)| = $(round(devE, sigdigits=3))")
end

println("all purification tests passed")
