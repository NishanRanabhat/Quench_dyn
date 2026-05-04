# Parameter Reference

Cheat-sheet for choosing DMRG, TDVP, and Krylov parameters. Read this
*after* `algorithms_primer.md` (which explains *why* each algorithm has
the parameters it does) and `diagnostics_guide.md` (which explains how
to react when the parameters you chose turn out wrong).

This document answers a single question per parameter: **what does it
control, and how do I pick it?**

---

## 1. DMRG parameters

### `chi_max_dmrg` — maximum bond dimension

| | |
|---|---|
| **Controls** | Rank kept at each bond after SVD. Bounds entanglement: `S ≤ log χ_max`. |
| **How to pick** | Scan. Run DMRG at `χ ∈ {32, 64, 128}`. Pick the smallest where `\|E(χ) − E(2χ)\| / N < 10⁻⁶`. |
| **Typical values for XXZ N=32** | gapped phase 32–64; near criticality 128–256 |
| **Cost** | `O(N · χ³ · D · d)` per sweep. Doubling χ → ~8× wall time. |
| **If wrong** | See `diagnostics_guide.md` §2.1 |

### `cutoff_dmrg` — SVD truncation threshold

| | |
|---|---|
| **Controls** | Singular values below this fraction of `‖S‖` are discarded. Acts as a soft bond-dim limit. |
| **How to pick** | `1e-10` for production, `1e-8` for exploration. Tighter than `1e-12` is wasted in double precision. |
| **Interaction with χ_max** | Whichever is more restrictive wins. |

### `n_sweeps_dmrg` — number of DMRG sweeps

| | |
|---|---|
| **Controls** | One left + one right pass per sweep. |
| **How to pick** | Watch per-sweep energy. Stop when `\|ΔE\|/N < 10⁻¹⁰`. 10–30 sweeps usually enough. |
| **If still drifting at 30 sweeps** | χ_max is too small, not sweeps too few. |

### `LanczosSolver(krylov_dim, max_iter)` — local eigensolver

| | |
|---|---|
| **`krylov_dim`** | Krylov subspace per Lanczos restart. Default 4. We only need the lowest eigenvalue, so small is fine. |
| **`max_iter`** | Lanczos restarts per local update. Default 14. Increase if local energies are noisy sweep-to-sweep. |
| **Tuning** | `(4, 14)` for gapped non-degenerate ground states. `(10, 30)` near gap closings or in degenerate sectors. |

---

## 2. TDVP parameters

### `chi_max_tdvp` — bond dimension ceiling during evolution

| | |
|---|---|
| **Controls** | Maximum bond dimension during time evolution. The single most important parameter for quench dynamics. |
| **Why it matters** | Entanglement grows linearly post-quench (Calabrese-Cardy). Bond dim required to faithfully represent the state grows exponentially in t. Eventually any finite cap saturates. |
| **How to pick** | Scan. For each `χ ∈ {2·χ_dmrg, 4·χ_dmrg, 8·χ_dmrg}`, run a short TDVP trajectory. Pick the smallest χ where observables agree with the next-larger-χ run within tolerance through `t_max`. |
| **Lower bound** | Must be ≥ `chi_max_dmrg` so the pre-quench ground state is representable without immediate truncation. |
| **Typical for XXZ N=32 quench, t ~ 5/J** | 200–800 |
| **Cost** | `O(N · χ³ · D · d)` per sweep, doubled per TDVP step (right + left). Larger χ → linear scaling in number-of-Krylov-applications too. |

Use `scripts/scan_tdvp_chi.jl` to do the χ-scan.

### `cutoff_tdvp` — SVD truncation during evolution

| | |
|---|---|
| **Controls** | Same meaning as `cutoff_dmrg`. |
| **How to pick** | `1e-10` for production, `1e-8` for exploration. Tighter than `1e-12` hits floating-point noise. |
| **Interaction with χ_max_tdvp** | If `cutoff_tdvp` is loose enough that it bites before `χ_max` does, `max_chi` will plateau below the cap. Tighten the cutoff to push `max_chi` toward the cap. |

### `dt` — time step

| | |
|---|---|
| **Controls** | Time per TDVP step. |
| **Error budget** | Trotter splitting error `O(dt²)` per step, `O(dt)` accumulated. The Krylov approximation of `exp(−iH_eff·dt)` adds its own error scaling with `dt^krylov_dim`. |
| **How to pick** | `dt ∈ [0.01, 0.05] · (1/J)` for XXZ at J=1. Verify by halving `dt` and checking observables move by less than your accuracy target. |
| **Relationship with `krylov_dim`** | Doubling `krylov_dim` lets you double `dt` at the same accuracy. Net cost is roughly a wash. |

### `t_max` — total evolution time

| | |
|---|---|
| **Controls** | When to stop. |
| **Bounded by** | `chi_max_tdvp`. Don't aim for a `t_max` past the saturation point — those data points are noise. |
| **How to find your reliable window** | Look at `1 − ‖ψ‖²(t)` and `max_chi(t)` from a single run. The time at which `max_chi` saturates marks the start of degradation. |

### `KrylovExponential(krylov_dim, krylov_tol, evol_type)`

| | |
|---|---|
| **`krylov_dim`** | Subspace size for `exp(−iH_eff·dt)` approximation. Default 30. Scripts use 14–30. Past ~40, modified Gram-Schmidt loses orthogonality and accuracy *decreases*. |
| **`krylov_tol`** | Convergence threshold for the Krylov expansion. Default 1e-12. |
| **`evol_type`** | `"real"` for unitary `exp(−iH·t)` (quench dynamics) or `"imaginary"` for `exp(−H·t)` (cooling / projection). |

---

## 3. Picking parameters — workflow

The conservative way:

1. **Pick `chi_max_dmrg`.** Run DMRG at `χ ∈ {32, 64, 128}`. Plateau in
   `E(χ)` tells you what's enough.
2. **Pick `n_sweeps_dmrg`.** Watch per-sweep energy. Stop when it
   flatlines.
3. **Verify with `var(H)`.** Should be `1e-10` or smaller. If not, see
   `diagnostics_guide.md` §2.1.
4. **Pick `chi_max_tdvp`.** Run `scripts/scan_tdvp_chi.jl` over a
   small χ ladder for short `t_max`. Find the smallest χ where
   observables agree with the next-larger χ within tolerance.
5. **Pick `dt`.** Halve it; check observables. If they move less than
   your tolerance, you're fine.
6. **Verify the production run.** `1 − ‖ψ‖²(t_max)` ≪ tolerance and
   `max_chi(t)` not flatlined for the last 20% of trajectory? Trust the
   run. If not, iterate.

---

## 4. Quick reference table

For an XXZ quench at N=32, J=1:

| Parameter | Conservative | Production |
|---|---|---|
| `chi_max_dmrg` | 64 | 128–256 |
| `cutoff_dmrg` | `1e-8` | `1e-10` |
| `n_sweeps_dmrg` | 20 | 30–50 |
| `lanczos_krylov_dim` | 4 | 6–10 |
| `lanczos_max_iter` | 14 | 20–30 |
| `chi_max_tdvp` | 128 | 256–800 |
| `cutoff_tdvp` | `1e-8` | `1e-10` |
| `dt` | 0.05 | 0.01–0.02 |
| `krylov_dim_exp` | 14 | 20–30 |
| `krylov_tol_exp` | `1e-8` | `1e-12` |

These are starting points. Tune from them per the workflow in §3.

---

## 5. Cross-references

- **Why each algorithm has the parameters it does:** `algorithms_primer.md`
- **What to do when diagnostics show a parameter is wrong:** `diagnostics_guide.md`
- **API and code structure:** `architecture.md`
- **Quick start:** `quickstart.md`
