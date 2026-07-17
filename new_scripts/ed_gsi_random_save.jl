# new_scripts/ed_gsi_random_save.jl — random-seeded initial states for the ED
# cache, matching Ho Jang's protocol: ONE Gaussian disorder realization
# (rng seed 1234, base std 1) scaled to magnitudes h = 1e-3, 1e-2, 1e-1 —
# same convention as docs/fcs_n12/make_fcs_seeded_figure.jl (hrand fixed,
# prefactor varied), so branch-selection strength is comparable across h.
# Field in H_i (Δi = 0.8) only; H_f stays clean, so the cached H_f spectra
# are reused unchanged. N = 16 only (the analysis size).
#
# Files data/ed_gsi_N16_delta0.8_rand{h}_seed1234.jls, layout identical to
# the staggered ones (+ :h_profile "gaussian_random", :h_std, :rng_seed, :h).
#
# Run with plain `julia new_scripts/ed_gsi_random_save.jl`.

using LinearAlgebra, Printf, Random, Serialization

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

const N, J, Δi = 16, 1.0, 0.8
const datadir = normpath(joinpath(@__DIR__, "..", "data"))

rng   = MersenneTwister(1234)
hbase = randn(rng, N)                    # THE realization (std-1 base)

git_commit = try
    strip(read(setenv(`git rev-parse HEAD`; dir = @__DIR__), String))
catch
    "unknown"
end

sec = sz_sector(N, N ÷ 2)
for h_std in (1e-3, 1e-2, 1e-1)
    file = joinpath(datadir, "ed_gsi_N$(N)_delta$(Δi)_rand$(h_std)_seed1234.jls")
    if isfile(file)
        @printf("h=%-6g : exists, skipping (%s)\n", h_std, basename(file))
        continue
    end
    h  = h_std .* hbase
    Hi = build_xxz_hamiltonian_sector(sec, J, Δi, h)
    t0 = time()
    Fi = eigen!(Hi, 1:1)
    E0, ψi = Fi.values[1], Fi.vectors[:, 1]
    m_s = sum((-1.0)^j * sec.sz[:, j]' * abs2.(ψi) for j in 1:N) / N
    @printf("h=%-6g : E0/N = %+.10f   m_s = %+.6f   [%5.1f s]\n",
            h_std, E0 / N, m_s, time() - t0)
    serialize(file, Dict{Symbol,Any}(
        :N => N, :J => J, :Delta => Δi, :n_up => sec.n_up, :boundary => "open",
        :git_commit => git_commit, :h_profile => "gaussian_random",
        :h_std => h_std, :rng_seed => 1234, :h => h,
        :energy => E0, :psi => ψi))
end
