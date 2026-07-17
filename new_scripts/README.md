# new_scripts/ — checkpointed quench runs and post-processing

The workflow in this folder separates the **expensive evolution** from the
**cheap measurements**: the TDVP run saves the full MPS to disk at every
measurement time, and every observable — FCS, correlators, anything invented
next month — is computed later from those files. Evolve once, measure
forever.

```
run_tdvp_1site_checkpoint.jl        evolution → saves MPS snapshots
measure_fcs_from_saved.jl           guide: FCS from the snapshots
measure_correlations_from_saved.jl  guide: profile / walls / G(r) from them
ed_diag_save.jl                     (ED side) full-spectrum cache builder
ed_gsi_seeded_save.jl               (ED side) staggered-seed initial states
ed_gsi_random_save.jl               (ED side) random-seed initial states
```

## run_tdvp_1site_checkpoint.jl

The evolution driver. Three stages, same physics as `scripts/run_tdvp_1site.jl`:

1. **DMRG** (2-site) finds the ground state of the pre-quench Hamiltonian
   H(Δᵢ) — this is the initial state of the quench. It is saved to
   `gs.jld2`.
2. **Padding** enlarges every bond to `chi_work` with tiny noise. 1-site
   TDVP cannot grow bonds, so the working manifold must be allocated up
   front; the noise (~1e-6) seeds the new directions so the quench can
   rotate into them.
3. **1-site TDVP** evolves under the post-quench H(Δ_f). It is
   bond-preserving and exactly unitary on the fixed-χ manifold, so energy
   and norm are conserved to Krylov tolerance (~1e-12 over hundreds of
   steps). Every `save_every` time units the full MPS is written to disk;
   nothing else is measured during the run.

All knobs (N, Δᵢ → Δ_f, seed field h in H_i, χ, dt, t_max, save_every) are
plain variables at the top of the script.

## Where and how the data are saved

The save location is set by the `save_root` line in the run script (a large
comment there explains the choices). Default: `run_root()` — which is the
gitignored `<repo>/runs/` locally, or `$QUENCHDYN_RUN_ROOT` if that
environment variable is set (on Rivanna export it to
`/scratch/<your_id>/QuenchDyn/runs`; scratch is purged after ~90 idle days,
so rsync the small CSV outputs somewhere permanent).

One run = one directory, named by its parameters:

```
<save_root>/xxz_quench_N16_di0.8_df3.0_chi64/
    gs.jld2                DMRG ground state of H_i (the t = 0⁻ state)
    mps_t0000.000.jld2     MPS snapshot at t = 0 (just after the quench)
    mps_t0001.000.jld2     ... one file per save_every ...
    mps_t0020.000.jld2
    evolution_log.csv      light in-run log: t, energy, norm, χ
    fcs_l8.csv             ← written later by the measurement scripts
    profile.csv  observables.csv  correlator.csv
```

Snapshot filenames are zero-padded (`snapshot_name(t)`), so an alphabetical
sort of the directory listing is automatically a time-ordered list.

Each `.jld2` file holds exactly two objects:

- `"tensors"` — the MPS as a plain `Vector{Array{ComplexF64,3}}`
  (left-bond × physical × right-bond). No custom types on disk: any Julia
  session with JLD2 can read these files, with or without this repository.
- `"meta"` — a `Dict{String,Any}` with the time `"t"`, every model and
  algorithm parameter (`"N"`, `"Delta_i"`, `"Delta_f"`, `"h"`, `"dt"`,
  `"chi_work"`, ...), the measured `"energy"` and `"norm"` at save time, and
  the `"git_commit"` of the code that produced it. A snapshot is therefore
  self-describing — post-processing scripts read N and t from the file, not
  from filenames or assumptions.

Disk cost: one snapshot ≈ 16 · N · χ² · d bytes. N=16, χ=64 → ~2 MB;
N=100, χ=512 → ~1.7 GB per file. Pick `save_every` with this in mind.

## Post-processing architecture

The measurement scripts are deliberately written as **guides**: every step
is a commented, editable line — load one MPS, build the operator, define the
window, call one measurement function — followed by the explicit `for` loop
that repeats it over all snapshots and saves CSVs. The library
(`src/Analysis`, `src/IO`) provides exactly three kinds of calls and hides
only the tensor contractions:

```julia
mps, meta = load_mps(file)              # read one snapshot (src/IO)
r = fcs_summary(mps, window, ops)       # FCS of Σ_{i∈window} O_i (src/Analysis/fcs.jl)
measure_correlation(mps, Sz, i, Sz, j)  # two-point functions (src/TensorOps)
```

`fcs_summary` returns `(m, P, extreme_weight, mean, variance, kurtosis)`:
the exact distribution of the windowed sum, computed by the generating-
function method (single-site phase insertions + exact discrete Fourier
inversion — Ranabhat & Collura, SciPost Phys. 12, 126 (2022)). The operator
is whatever you build in the script: `[(-1.0)^i * Sz for i in win]` gives
the staggered magnetization, a single `Sz` the uniform one, `spin_ops(2)[:X]`
the transverse channel. For finite-temperature purification snapshots pass
`embed = true`.

Because the snapshots persist, a new observable never costs a re-run:
copy a measurement script, change the operator/loop body, point it at the
same run directory.

## ED cache scripts (`ed_*.jl`)

Independent of the MPS pipeline: they build the exact-diagonalization
reference cache in `data/` (full S^z=0-sector eigensystems of H_f for
N = 8–16 and the clean/seeded initial states) used by `docs/ed_analysis/`
and by the FCS benchmarks in `tests/`. Run once; everything downstream
(thermal states, real-time evolution, distributions) is post-processing of
those files — the same evolve-once philosophy at the ED level.
