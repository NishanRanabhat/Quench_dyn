# new_scripts/ed_gsi_seeded_save.jl — seeded initial states for the ED cache.
#
# Companion to ed_diag_save.jl. The quench protocol used in practice (Ho
# Jang's TDVP runs, the N=12 seeded FCS study) puts a small LONGITUDINAL
# symmetry-breaking seed field in H_i only:
#
#   H_i = XXZ(J, Δi) + Σ_i h_s (−1)^i S^z_i ,   H_f clean (h = 0).
#
# A z-field commutes with S^z_total, so the S^z = 0 sector treatment stays
# valid. Because H_f carries no field, the saved H_f eigensystems
# (data/ed_spectrum_*.jls) are reused unchanged — a seeded quench only needs
# the new initial vector, c0 = V' * ψi. This script therefore saves just the
# ground state of the seeded H_i at each N; it does NOT re-diagonalize
# anything large (lowest eigenpair only: ~2 min at N = 16, seconds below).
#
# Sign convention matches docs/fcs_n12/make_fcs_seeded_figure.jl:
#   h_i = h_s · (−1)^i, i = 1:N  (site 1 gets −h_s), which favors the
#   Néel branch with ⟨S^z_i⟩ ∝ −(−1)^i, i.e. negative staggered
#   magnetization m = Σ_i (−1)^i ⟨S^z_i⟩ < 0.
#
# Strengths: h_s = 0.01 (linear-response regime) and 0.1 (strong seed,
# matching the N=12 study). Random-field profiles were deliberately left
# out (realization-specific, ~10× weaker per unit h at q=π); add here if
# ever needed.
#
# Files data/ed_gsi_N{N}_delta{Δi}_stag{h_s}.jls (Dict{Symbol,Any}):
#   :N, :J, :Delta, :n_up, :boundary, :git_commit,
#   :h_profile => "staggered", :h_s, :h (full per-site field vector),
#   :energy::Float64, :psi::Vector{Float64}
# Basis order = sz_sector(N, N÷2).configs, deterministic, not stored.
#
# Run with plain `julia new_scripts/ed_gsi_seeded_save.jl` (no --project).

using LinearAlgebra, Printf, Serialization

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

const J   = 1.0
const Δi  = 0.8
const hss = (0.01, 0.1)
const Ns  = (8, 10, 12, 14, 16)

const datadir = normpath(joinpath(@__DIR__, "..", "data"))
mkpath(datadir)

git_commit = try
    strip(read(setenv(`git rev-parse HEAD`; dir = @__DIR__), String))
catch
    "unknown"
end

for N in Ns
    sec = sz_sector(N, N ÷ 2)
    @printf("N = %2d  (sector dim %d)\n", N, length(sec))
    for h_s in hss
        h  = [h_s * (-1.0)^i for i in 1:N]
        Hi = build_xxz_hamiltonian_sector(sec, J, Δi, h)
        t0 = time()
        Fi = eigen!(Hi, 1:1)                 # lowest eigenpair only
        E0, ψi = Fi.values[1], Fi.vectors[:, 1]
        # staggered magnetization of the seeded GS — the branch-selection check
        m_s = sum((-1.0)^j * sec.sz[:, j]' * (abs2.(ψi)) for j in 1:N) / N
        @printf("  h_s = %-5g : E0/N = %+.10f   m_s = %+.6f   [%6.1f s]\n",
                h_s, E0 / N, m_s, time() - t0)
        d = Dict{Symbol,Any}(
            :N => N, :J => J, :Delta => Δi, :n_up => sec.n_up,
            :boundary => "open", :git_commit => git_commit,
            :h_profile => "staggered", :h_s => h_s, :h => h,
            :energy => E0, :psi => ψi,
        )
        serialize(joinpath(datadir, "ed_gsi_N$(N)_delta$(Δi)_stag$(h_s).jls"), d)
    end
end

println("\nSeeded initial-state files in $datadir:")
for f in sort(filter(startswith("ed_gsi"), readdir(datadir)))
    @printf("  %-40s %8.2f KB\n", f, filesize(joinpath(datadir, f)) / 1e3)
end
