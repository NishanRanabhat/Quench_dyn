# tests/test_mps_io.jl — MPS JLD2 checkpointing (src/IO/serialization.jl):
# save/load round-trip, snapshot naming, and a micro evolve-save-measure
# pipeline checking that FCS from reloaded snapshots equals FCS measured
# in memory during the evolution. Run: julia tests/test_mps_io.jl

using LinearAlgebra, Printf, Test
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

@testset "MPS JLD2 IO" begin
    tmp = mktempdir()

    # ── round trip: tensors, eltype, metadata ────────────────────────────
    sites = [SpinSite(0.5; T = ComplexF64) for _ in 1:10]
    mps = random_state(sites, 8)
    meta = Dict("t" => 1.5, "N" => 10, "note" => "round-trip")
    f = joinpath(tmp, "roundtrip.jld2")
    save_mps(f, mps; meta = meta)
    mps2, meta2 = load_mps(f)
    @test length(mps2.tensors) == 10
    @test eltype(mps2.tensors[1]) == ComplexF64
    @test all(mps.tensors[i] == mps2.tensors[i] for i in 1:10)
    @test meta2["t"] == 1.5 && meta2["note"] == "round-trip"

    # FCS identical through the round trip
    mg, P1 = staggered_fcs(mps, collect(3:8))
    _, P2  = staggered_fcs(mps2, collect(3:8))
    @test P1 == P2

    # ── snapshot naming: zero-padded, sortable, parseable ────────────────
    @test snapshot_name(0.0) == "mps_t0000.000.jld2"
    @test snapshot_name(12.5) == "mps_t0012.500.jld2"
    @test snapshot_time(snapshot_name(7.25)) == 7.25
    ts = [0.0, 2.0, 10.0, 100.5]
    @test sort(snapshot_name.(ts)) == snapshot_name.(ts)   # lexicographic = time order

    # ── micro pipeline: evolve → save each step → reload → same FCS ──────
    N = 8
    s8 = [SpinSite(0.5; T = ComplexF64) for _ in 1:N]
    mps0 = product_state(s8, [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N])
    st = MPSState(mps0, build_xxz_mpo(N, 1.0, 0.8, zeros(N)); center = 1)
    for sw in 1:12
        dmrg_sweep(st, LanczosSolver(4, 14), DMRGOptions(16, 1e-12, 2),
                   isodd(sw) ? :right : :left)
    end
    padded = pad_mps(st.mps, 16; noise = 1e-7)
    st = MPSState(padded, build_xxz_mpo(N, 1.0, 3.0, zeros(N)); center = 1)
    solver = KrylovExponential(12, 1e-10, "real")
    opts   = TDVPOptions(0.05, 16, 1e-12, 2)
    win = collect(3:6)
    P_mem = Vector{Vector{Float64}}()
    for step in 1:10
        tdvp_sweep_one_site(st, solver, opts, :right)
        tdvp_sweep_one_site(st, solver, opts, :left)
        t = step * 0.05
        save_mps(joinpath(tmp, snapshot_name(t)), st.mps;
                 meta = Dict("t" => t, "N" => N))
        push!(P_mem, staggered_fcs(st.mps, win)[2])
    end
    snaps = sort(filter(startswith("mps_t"), readdir(tmp)))
    @test length(snaps) == 10
    for (k, f) in enumerate(snaps)
        m, meta_k = load_mps(joinpath(tmp, f))
        @test meta_k["t"] ≈ k * 0.05
        @test staggered_fcs(m, win)[2] == P_mem[k]
    end

    rm(tmp; recursive = true)
end

println("test_mps_io: all passed")
