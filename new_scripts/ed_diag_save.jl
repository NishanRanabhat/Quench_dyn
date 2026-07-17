# new_scripts/ed_diag_save.jl — one-time ED spectrum cache builder.
#
# Full diagonalization of the two final XXZ Hamiltonians of the quench
# protocol, H_f with Δ_f = 1.2 and 3.0 (J = 1, h = 0, open boundaries),
# in the S^z_total = 0 sector for N = 8, 10, 12, 14, 16. The complete
# eigensystem is saved to data/ with Julia's stdlib Serialization, together
# with the ground state of the initial Hamiltonian H_i (Δ_i = 0.8) at each N
# — the starting state of the quench.
#
# Everything downstream is cheap post-processing of these files, with no
# re-diagonalization:
#   thermal state at any β      : weights ∝ exp.(-β .* values)
#   T_eff of a quench           : effective_beta(values, ⟨ψi|H_f|ψi⟩)
#   real-time evolution         : c0 = V' * ψi;  ψ(t) = V * (c0 .* exp.(-im .* values .* t))
#   diagonal ensemble           : abs2.(c0)
#
# File layout (Dict{Symbol,Any}, stdlib `deserialize` to load):
#   spectrum files  ed_spectrum_N{N}_delta{Δf}.jls :
#       :N, :J, :Delta, :n_up, :boundary, :git_commit,
#       :values  :: Vector{Float64}   (ascending)
#       :vectors :: Matrix{Float64}   (columns; H is real symmetric so the
#                                      eigenvectors are kept real — convert
#                                      with ComplexF64.() if an EDEigensystem
#                                      is needed)
#   initial-GS files  ed_gsi_N{N}_delta{Δi}.jls :
#       same metadata plus :energy::Float64, :psi::Vector{Float64}
#
# Basis convention: the coefficient order is sz_sector(N, N÷2).configs —
# deterministic, so the sector is rebuilt on load rather than stored.
#
# Sector dimensions / file sizes (Float64 vectors):
#   N = 8 : 70      ~40 KB        N = 14 : 3432   ~94 MB
#   N = 10: 252     ~500 KB       N = 16 : 12870  ~1.33 GB
#   N = 12: 924     ~7 MB
# eigen! is used in place of eigen/diagonalize to avoid one 1.3 GB copy of H
# at N = 16 (peak RSS ≈ 2.7 GB there).
#
# Run with plain `julia new_scripts/ed_diag_save.jl` (no --project).

using LinearAlgebra, Printf, Serialization

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

const J   = 1.0
const Δi  = 0.8
const Δfs = (1.2, 3.0)
const Ns  = (8, 10, 12, 14, 16)

const datadir = normpath(joinpath(@__DIR__, "..", "data"))
mkpath(datadir)

git_commit = try
    strip(read(setenv(`git rev-parse HEAD`; dir = @__DIR__), String))
catch
    "unknown"
end

metadata(N, Δ, sec) = Dict{Symbol,Any}(
    :N => N, :J => J, :Delta => Δ, :n_up => sec.n_up,
    :boundary => "open", :git_commit => git_commit,
)

for N in Ns
    sec = sz_sector(N, N ÷ 2)
    @printf("N = %2d  (sector dim %d)\n", N, length(sec))

    # ── ground state of H_i(Δi): the quench initial state ────────────────
    Hi = build_xxz_hamiltonian_sector(sec, J, Δi)
    t0 = time()
    Fi = eigen!(Hi, 1:1)                     # lowest eigenpair only
    E0, ψi = Fi.values[1], Fi.vectors[:, 1]
    @printf("  GS(Δ=%.1f)      : E0/N = %+.10f              [%6.1f s]\n",
            Δi, E0 / N, time() - t0)
    d = metadata(N, Δi, sec)
    d[:energy] = E0
    d[:psi]    = ψi
    serialize(joinpath(datadir, "ed_gsi_N$(N)_delta$(Δi).jls"), d)
    Hi = Fi = nothing

    # ── full spectrum of each H_f(Δf) ─────────────────────────────────────
    for Δf in Δfs
        Hf = build_xxz_hamiltonian_sector(sec, J, Δf)
        t0 = time()
        F  = eigen!(Hf)                      # full spectrum, in place
        @printf("  spectrum(Δ=%.1f): E0/N = %+.10f  Emax/N = %+.7f  [%6.1f s]\n",
                Δf, F.values[1] / N, F.values[end] / N, time() - t0)
        d = metadata(N, Δf, sec)
        d[:values]  = F.values
        d[:vectors] = F.vectors
        serialize(joinpath(datadir, "ed_spectrum_N$(N)_delta$(Δf).jls"), d)
        Hf = F = d = nothing
        GC.gc()
    end
end

println("\nAll files written to $datadir:")
for f in sort(readdir(datadir))
    @printf("  %-34s %9.2f MB\n", f, filesize(joinpath(datadir, f)) / 1e6)
end
