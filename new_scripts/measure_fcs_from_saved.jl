# new_scripts/measure_fcs_from_saved.jl
#
# ═══ GUIDE: full counting statistics from saved MPS snapshots ══════════════
#
# This file is a walkthrough, not a black box. It shows, step by step:
#
#   PART 1 — load ONE saved MPS, build an operator, define a window
#            (subsystem), and compute its FCS with a single call.
#   PART 2 — the explicit for loop over ALL snapshots in a run folder,
#            computing the FCS at every time and saving the results to CSV.
#
# The only library calls are `load_mps` (read a snapshot) and `fcs_summary`
# (measure the distribution); everything else is ordinary Julia that you
# control and can edit. See new_scripts/README.md for how the snapshots got
# on disk in the first place.
#
# Usage:
#   julia new_scripts/measure_fcs_from_saved.jl <run_dir>
# ═══════════════════════════════════════════════════════════════════════════

# make the QuenchDyn library visible (src/ sits one level above this folder)
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# load the library: provides load_mps, fcs_summary, spin_ops, ...
using QuenchDyn

# the run directory written by run_tdvp_1site_checkpoint.jl,
# taken from the command line
run_dir = ARGS[1]


# ═══ PART 1: FCS of a single saved MPS ═════════════════════════════════════

# pick one snapshot file — the zero-padded name encodes the time (t = 5 here)
file = joinpath(run_dir, "mps_t0005.000.jld2")

# load it: `mps` is the state, `meta` is a Dict with everything the run
# script recorded — meta["t"], meta["N"], meta["Delta_f"], meta["energy"], ...
mps, meta = load_mps(file)

# chain length, read from the snapshot's own metadata
N = meta["N"]

# ── the window (subsystem): the ℓ contiguous sites over which the FCS is
#    taken. Here: ℓ = 8 sites centered in the chain (buffered from the edges).
ℓ   = 8
win = collect(((N - ℓ) ÷ 2 + 1):((N - ℓ) ÷ 2 + ℓ))

# ── the operator: the FCS is the distribution of  A = Σ_{i ∈ win} O_i .
#    Build one matrix per window site. Here O_i = (−1)^i S^z_i, so A is the
#    STAGGERED magnetization of the window; the (−1)^i uses the ABSOLUTE
#    site index i (site 1 of the chain is odd), matching the ED convention.
#    For a uniform observable use e.g.  ops = [Sz for i in win]  — or the
#    transverse channel:  ops = [Matrix(spin_ops(2)[:X]) for i in win].
Sz  = Matrix(spin_ops(2)[:Z])
ops = [(-1.0)^i * Sz for i in win]

# ── the measurement: one call. Returns a NamedTuple with
#      r.m         the possible values of A (here −4, −3, …, +4)
#      r.P         the exact probability of each value (sums to 1)
#      r.extreme_weight   P(A = min) + P(A = max)  — for the staggered
#                         operator this is the "ordered weight": the
#                         probability that the window is perfectly Néel
#      r.mean, r.variance, r.kurtosis   (kurtosis: 3 = Gaussian, 1 = bimodal)
#    (for a finite-T purification snapshot add: embed = true)
r = fcs_summary(mps, win, ops)

# look at it
println("t = $(meta["t"]),  window = $win")
println("  m       : ", r.m)
println("  P(m)    : ", round.(r.P; digits = 4))
println("  ordered weight = $(round(r.extreme_weight; digits = 4)), ",
        "kurtosis = $(round(r.kurtosis; digits = 3))")


# ═══ PART 2: loop over every snapshot in the folder, save FCS to CSV ═══════

# list the snapshot files; the zero-padded names make an alphabetical sort
# identical to time ordering
snaps = sort(filter(startswith("mps_t"), readdir(run_dir)))

# open the output table next to the data (one row per time)
open(joinpath(run_dir, "fcs_l$ℓ.csv"), "w") do io

    # column names: time, one P column per value of A, then the summaries
    println(io, "t," * join(["P(m=$m)" for m in r.m], ",") *
                ",extreme_weight,mean,variance,kurtosis")

    # the loop: load each snapshot, measure, write one row
    for f in snaps

        # read the state and its metadata from disk
        mps, meta = load_mps(joinpath(run_dir, f))

        # same window, same operator, same single call as in PART 1
        r = fcs_summary(mps, win, ops)

        # append: time, the full distribution, and the summary numbers
        println(io, join((meta["t"], r.P..., r.extreme_weight,
                          r.mean, r.variance, r.kurtosis), ","))
    end
end

println("wrote $(joinpath(run_dir, "fcs_l$ℓ.csv"))  ($(length(snaps)) times)")

# To scan several window sizes, wrap PART 2 in `for ℓ in (4, 8, 12) ... end`
# (rebuild `win` and `ops` inside the loop, and name the file per ℓ).
