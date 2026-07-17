# new_scripts/run_tdvp_1site_checkpoint.jl
#
# Quench dynamics via 1-site TDVP with MPS CHECKPOINTING: the state is SAVED
# to disk (JLD2) at every measurement time instead of measured on the fly.
# All observables — FCS, correlators, profiles, entropies, anything invented
# later — are computed afterwards by new_scripts/measure_fcs_from_saved.jl
# (or any other post-processing script) from the saved snapshots. Evolve
# once, measure forever.
#
# Pipeline (same physics as run_tdvp_1site.jl):
#   1) 2-site DMRG ground state of H(Δi)          →  <run_dir>/gs.jld2
#   2) pad to chi_work (noise seeds new bonds)
#   3) 1-site TDVP under H(Δf); every `save_every` time units the MPS is
#      written to <run_dir>/mps_t<time>.jld2 with full metadata; a light
#      log (t, E, norm) goes to <run_dir>/evolution_log.csv
#
# Output location: see the "WHERE THE MPS SNAPSHOTS ARE SAVED" block below —
# edit save_root (or export QUENCHDYN_RUN_ROOT) for your machine/cluster.
# Details and disk-budget table: docs/mps_checkpointing.md.
#
# Run: julia new_scripts/run_tdvp_1site_checkpoint.jl

using LinearAlgebra, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", string(Sys.CPU_THREADS))))

# ═══ MODEL ═════════════════════════════════════════════════════════════════
N       = 16
J       = 1.0
d       = 2
Delta_i = 0.8
Delta_f = 3.0
h       = zeros(N)          # seed field in H_i ONLY (H_f is built clean below)

# ═══ DMRG (ground state of H_i) ════════════════════════════════════════════
chi_dmrg    = 64
n_sweeps    = 24
cutoff_dmrg = 1e-12

# ═══ PADDING + TDVP (1-site, H_f) ══════════════════════════════════════════
chi_work   = 64             # fixed χ of the evolution (pad target)
pad_noise  = 1e-6
dt         = 0.05
t_max      = 20.0
krylov_dim = 14
krylov_tol = 1e-10

# ═══ CHECKPOINTING ═════════════════════════════════════════════════════════
save_every = 1.0            # time units between saved snapshots

# ═══ WHERE THE MPS SNAPSHOTS ARE SAVED — SET THIS FOR YOUR SYSTEM ══════════
#
# All output of this run (gs.jld2, one mps_t*.jld2 per snapshot,
# evolution_log.csv, and later the post-processing CSVs) goes into
#
#     <save_root>/<tag>/
#
# where <tag> is built from the physics parameters below. CHOOSE save_root
# deliberately — snapshots are LARGE (one snapshot ≈ 16·N·χ²·d bytes; at
# N=100, χ=512 that is ~1.7 GB PER FILE), and they must NOT live in a git
# repository or a small home-quota filesystem.
#
#   • Local workstation:  the default `run_root()` resolves to <repo>/runs
#     (gitignored). Fine for small N; replace with an absolute path on a
#     big disk for anything serious, e.g.
#         save_root = "/media/bigdisk/QuenchDyn/runs"
#
#   • Cluster (Rivanna):  point it at YOUR scratch, e.g.
#         save_root = "/scratch/<your_computing_id>/QuenchDyn/runs"
#     or leave the line as-is and export the environment variable in your
#     job script instead (run_root() honors it):
#         export QUENCHDYN_RUN_ROOT=/scratch/$USER/QuenchDyn/runs
#     Remember Rivanna scratch is PURGED (~90 days idle) — copy the small
#     CSV outputs somewhere permanent; snapshots are regenerable.
#
# The post-processing scripts take the printed run directory as their
# argument, so they work unchanged wherever you point this.
#
save_root = run_root()          # ← EDIT (or set QUENCHDYN_RUN_ROOT)

tag = "xxz_quench_N$(N)_di$(Delta_i)_df$(Delta_f)_chi$(chi_work)" *
      (any(!iszero, h) ? "_h$(maximum(abs.(h)))" : "")
run_dir = joinpath(save_root, tag)
mkpath(run_dir)

git_commit = try
    strip(read(setenv(`git rev-parse HEAD`; dir = @__DIR__), String))
catch; "unknown"; end

params = Dict{String,Any}(
    "N" => N, "J" => J, "d" => d, "Delta_i" => Delta_i, "Delta_f" => Delta_f,
    "h" => h, "chi_dmrg" => chi_dmrg, "chi_work" => chi_work,
    "pad_noise" => pad_noise, "dt" => dt, "t_max" => t_max,
    "save_every" => save_every, "krylov_dim" => krylov_dim,
    "krylov_tol" => krylov_tol, "git_commit" => git_commit,
    "algorithm" => "tdvp_1site (2-site DMRG warmup + pad)",
)

println("run directory: $run_dir")
println("snapshots every $save_every / dt=$dt up to t=$t_max  " *
        "($(floor(Int, t_max / save_every) + 1) files)\n")

# ── 1) DMRG ground state of H_i ────────────────────────────────────────────
mpo_i = build_xxz_mpo(N, J, Delta_i, h; d = d)
sites = [SpinSite(0.5; T = ComplexF64) for _ in 1:N]
mps   = product_state(sites, [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N])
state = MPSState(mps, mpo_i; center = 1)
solver_d = LanczosSolver(4, 14)
opts_d   = DMRGOptions(chi_dmrg, cutoff_dmrg, d)
E_gs = 0.0
for sweep in 1:n_sweeps
    global E_gs = dmrg_sweep(state, solver_d, opts_d,
                             isodd(sweep) ? :right : :left).E
end
@printf("DMRG GS: E = %.10f, var = %.2e\n", E_gs, energy_variance(state))
save_mps(joinpath(run_dir, "gs.jld2"), state.mps;
         meta = merge(params, Dict("what" => "ground state of H_i",
                                   "energy" => E_gs)))

# ── 2) pad → 3) 1-site TDVP under H_f with snapshots ──────────────────────
padded = pad_mps(state.mps, chi_work; noise = pad_noise)
mpo_f  = build_xxz_mpo(N, J, Delta_f, zeros(N); d = d)      # clean H_f
state  = MPSState(padded, mpo_f; center = 1)
solver = KrylovExponential(krylov_dim, krylov_tol, "real")
opts   = TDVPOptions(dt, chi_work, 1e-12, d)

n_steps    = round(Int, t_max / dt)
save_steps = round(Int, save_every / dt)
E0 = measure_energy(state)

logio = open(joinpath(run_dir, "evolution_log.csv"), "w")
println(logio, "t,energy,norm,chi_max")

function checkpoint(t)
    E, nrm = measure_energy(state), real(measure_norm(state.mps))
    χ = maximum(size(A, 3) for A in state.mps.tensors[1:end-1])
    save_mps(joinpath(run_dir, snapshot_name(t)), state.mps;
             meta = merge(params, Dict("what" => "quench snapshot", "t" => t,
                                       "energy" => E, "norm" => nrm)))
    println(logio, "$t,$E,$nrm,$χ"); flush(logio)
    @printf("  t = %7.3f : saved  (E drift %.2e, 1−‖ψ‖² %.2e)\n",
            t, E - E0, 1 - nrm)
end

checkpoint(0.0)
for step in 1:n_steps
    tdvp_sweep_one_site(state, solver, opts, :right)
    tdvp_sweep_one_site(state, solver, opts, :left)
    step % save_steps == 0 && checkpoint(step * dt)
end
close(logio)
println("\ndone. post-process with:\n  julia new_scripts/measure_fcs_from_saved.jl $run_dir")
