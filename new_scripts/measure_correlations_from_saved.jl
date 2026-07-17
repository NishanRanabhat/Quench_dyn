# new_scripts/measure_correlations_from_saved.jl
#
# ═══ GUIDE: correlation-channel observables from saved MPS snapshots ═══════
#
# Companion to measure_fcs_from_saved.jl, same structure: explicit loop over
# the snapshots, library calls only for the individual measurements
# (measure_local_profile, measure_correlation, domain_length). Computes the
# one- and two-point S^z observables the FCS does not cover:
#
#   profile.csv       t, ⟨S^z_i⟩ for every site, staggered mean m_s
#   observables.csv   t, energy & norm (from metadata), m_s, wall density
#                     ρ_w = ⟨1/2 + 2 S^z_i S^z_{i+1}⟩ on bulk bonds
#   correlator.csv    t, bulk staggered G(r), domain length L_int
#
# The G(r) block is the expensive one at large χ — delete it (or shrink
# `refs`) if you only need the cheap rows.
#
# Usage:
#   julia new_scripts/measure_correlations_from_saved.jl <run_dir>
# ═══════════════════════════════════════════════════════════════════════════

# make the QuenchDyn library visible and load it
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn
using LinearAlgebra

# the run directory, from the command line
run_dir = ARGS[1]

# snapshot list (zero-padded names ⇒ sorted = time-ordered)
snaps = sort(filter(startswith("mps_t"), readdir(run_dir)))

# chain length from the first snapshot's metadata
N = load_mps(joinpath(run_dir, snaps[1]))[2]["N"]

# the operator for this whole channel
Sz = Matrix{ComplexF64}(spin_ops(2)[:Z])

# bulk choices: reference sites for G(r), bond range for the wall density
refs  = collect(max(2, N ÷ 4) .+ (0:min(7, N ÷ 2 - 1)))
bonds = (N ÷ 4):(3N ÷ 4)
rmax  = N ÷ 2

# open the three output tables and write their headers
prof = open(joinpath(run_dir, "profile.csv"), "w")
obs  = open(joinpath(run_dir, "observables.csv"), "w")
corr = open(joinpath(run_dir, "correlator.csv"), "w")
println(prof, "t," * join(["sz_$i" for i in 1:N], ",") * ",m_s")
println(obs, "t,energy,norm,m_s,wall_density")
println(corr, "t," * join(["G_r$r" for r in 0:rmax], ",") * ",L_int")

# the loop over snapshots
for f in snaps

    # read the state and metadata; norm divides every expectation value
    # (snapshots may be unnormalized)
    mps, meta = load_mps(joinpath(run_dir, f))
    t   = meta["t"]
    nrm = real(measure_norm(mps))

    # one-point profile ⟨S^z_i⟩ and its staggered average
    sz  = measure_local_profile(mps, Sz) ./ nrm
    m_s = sum((-1.0)^i * sz[i] for i in 1:N) / N
    println(prof, join((t, sz..., m_s), ","))

    # wall density: probability of parallel neighbors, averaged on bulk bonds
    ρw = sum(0.5 + 2 * real(measure_correlation(mps, Sz, b, Sz, b + 1)) / nrm
             for b in bonds) / length(bonds)
    println(obs, join((t, meta["energy"], meta["norm"], m_s, ρw), ","))

    # bulk-averaged staggered correlator G(r) and the domain length
    G = zeros(rmax + 1)
    for r in 0:rmax
        vals = [r == 0 ? real(measure_local_observable(mps, Sz * Sz, i0)) :
                         real(measure_correlation(mps, Sz, i0, Sz, i0 + r))
                for i0 in refs if i0 + r <= N]
        G[r + 1] = (-1.0)^r * sum(vals) / (nrm * length(vals))
    end
    println(corr, join((t, G..., domain_length(0:rmax, G)), ","))

    println("  processed $f (t = $t)")
end

# close the tables
close(prof); close(obs); close(corr)
println("wrote profile.csv, observables.csv, correlator.csv in $run_dir")
