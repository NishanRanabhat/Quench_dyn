# Diagnostics Guide

A training manual for reading the output of QuenchDyn runs and acting
on what you see. Read `algorithms_primer.md` first if you don't yet
know what bond dimension is or why 2-site TDVP can't conserve norm.

This document answers four questions:

1. **What does each diagnostic measure?**
2. **What do healthy vs unhealthy values look like?**
3. **If unhealthy, what should I change?**
4. **How do I read a CSV from start to finish?**

The diagnostics QuenchDyn produces:

- `norm_sq` (a.k.a. `1 − ‖ψ‖²`) — how much weight has been lost to truncation
- `energy` and `dE = E(t) − E(0)` — energy drift
- `max_chi(t)` — the largest bond dimension currently used
- `max_trunc` — relative norm² shed in the worst bond of the latest sweep
- `total_trunc` — the same, summed over all bonds in the sweep
- `var(H) = ⟨H²⟩ − ⟨H⟩²` — only meaningful for DMRG ground states
- (planned) `S(t)` — half-chain entanglement entropy

We discuss each in turn, then give two diagnostic decision trees (one
for DMRG, one for TDVP), then walk through three real CSV examples.

---

## 1. The diagnostics, one by one

### 1.1 `1 − ‖ψ‖²`

**What it measures.** The MPS norm squared, `⟨ψ|ψ⟩`, computed by
contracting the MPS with itself. We log `1 − ‖ψ‖²` so that a perfectly
normalized state shows zero.

**Why TDVP-2 leaks it.** Each SVD truncation discards singular weight,
and the discarded weight comes directly out of `‖ψ‖²`. A perfectly
unitary algorithm would log values flat at floating-point noise
(±1e-14). 2-site TDVP under truncation cannot.

**Healthy.** Either floating-point noise (~1e-13, no truncation) or a
small monotonic positive drift (truncation is happening but staying
under control). What "small" means depends on your accuracy target —
see the table in §1 of `algorithms_primer.md` for the rough
correspondence between `1 − ‖ψ‖²` and worst-case observable error.
Order of magnitude: `1e-6` is paper-quality for local observables;
`1e-3` is exploratory only.

**Unhealthy.**

- **Sudden onset of growth** in the middle of a trajectory. Cross-
  reference `max_chi(t)`: this typically marks the moment the bond
  dimension saturated. Past that point, the state is straining
  against its budget. This is the textbook 2-site TDVP failure mode.
- **Growth with no visible saturation**, while `max_chi(t) <<
  chi_max`. Means `cutoff_tdvp` is biting before χ_max. Tighten the
  cutoff if you want less drift; loosen it if you can tolerate the
  drift in exchange for a smaller χ.
- **Negative `1 − ‖ψ‖²`** at floating-point levels (e.g., −1e-13) is
  fine — it's just round-off. **Negative growing** is a bug; tell us.

**How to act.** See decision tree §2.2 below.

---

### 1.2 `energy` and `dE`

**What it measures.** Expectation `⟨ψ(t)|H|ψ(t)⟩` under the *post-quench*
Hamiltonian. For exact unitary evolution this is conserved exactly. For
2-site TDVP, drift comes from the same SVD truncation that leaks norm.

**Healthy.** `dE` floating-point noise (~1e-12 relative to E) until
truncation kicks in; small drift after.

**Unhealthy.**

- **Constant nonzero `dE`** appearing immediately at t > 0 with no
  growth. This is a Krylov-approximation artifact: the local
  exponential `exp(−i H_eff dt)` is replaced by a Krylov projection,
  which can introduce a small constant offset. Increasing `krylov_dim`
  fixes it. We saw this in the validate_tdvp.jl test (4.4e-5 offset
  with N=6).

- **Growing `dE(t)`** that tracks `1 − ‖ψ‖²(t)`. Same root cause as
  norm drift — SVD truncation. Same fix.

- **`dE(t)` growing while `1 − ‖ψ‖²` is flat.** Unusual. Likely a
  numerical issue in the energy computation (e.g., an MPS that has
  drifted out of canonical form). Recompute `make_canonical` before
  measuring.

**How to act.** Energy and norm drift have the same root cause in
QuenchDyn: SVD truncation. Fix one and the other follows.

---

### 1.3 `max_chi(t)`

**What it measures.** The largest bond dimension across all bonds of
the MPS at time t.

**Healthy.** During a quench, expect `max_chi(t)` to grow roughly
linearly in time at first, then *saturate* at some value. If saturation
occurs *below* your `chi_max_tdvp` cap, the algorithm is not constrained
by your χ budget — the state is genuinely well-represented.

**Unhealthy.**

- **`max_chi(t) = chi_max_tdvp`** is the smoking gun. Once the
  measured max bond dim equals your cap, every SVD step is doing
  forced truncation, and from that moment `max_trunc` will grow.
- **`max_chi` plateauing well below the cap, but truncation still
  active.** This is the cutoff biting before χ_max. Tighten `cutoff_tdvp`
  if you want fewer truncations; this typically just makes χ rise to
  hit the cap a bit later. Often acceptable.

**How to act.** If `max_chi` saturates and you need more time
reliability, increase `chi_max_tdvp`. There is no other fix.

---

### 1.4 `max_trunc` and `total_trunc`

**What they measure.** For each SVD truncation step:

  discarded_weight = (Σ all σ²) − (Σ kept σ²)
                   ─────────────────────────────
                          Σ all σ²

This is the *relative* norm² shed at that bond. `max_trunc` is the
worst over all bonds in a sweep; `total_trunc` is the sum.

**Why relative.** The kept singular values may have a wildly different
overall scale than the cutoff_tdvp threshold suggests; normalizing by
the total spectrum makes the metric meaningful regardless of the state
being normalized at that moment.

**Healthy.** Floating-point noise (~1e-13 to 1e-15) when no real
truncation happens. The cumulative `1 − ‖ψ‖²(t)` is the time integral
of `total_trunc`, modulo signs.

**Unhealthy.** `max_trunc > 1e-10` and rising means truncation is
biting; expect norm and energy drift to grow accordingly.

**How to act.** Increase `chi_max_tdvp`, or tighten `cutoff_tdvp` if
the cap isn't binding.

---

### 1.5 `var(H) = ⟨H²⟩ − ⟨H⟩²`

**What it measures.** Variance of H in the state |ψ⟩. Exactly zero at
any eigenstate of H. So it tells you precisely how far your DMRG
ground-state estimate is from being a true eigenstate.

**When to use.** Only meaningful for **DMRG-converged states**. During
a quench, the state is *deliberately* not an eigenstate of H_post-quench;
`var(H)` will be O(N) right after the quench and stay there. Don't log
it during TDVP.

**Healthy.** For a converged DMRG ground state of a gapped 1D chain at
modest χ: `var(H) ~ 1e-10` to `1e-12` (absolute). For our XXZ N=24 χ=64
benchmark, we saw `var(H) = 2.3e-12` — excellent.

**Unhealthy.** `var(H) > 1e-6` after many sweeps usually means one of:

- DMRG stuck in a local minimum. Restart with a different initial MPS,
  or use a chi-warmup if you don't already.
- χ is too small for this Hamiltonian. Increase `chi_max_dmrg`.
- The Lanczos solver isn't converging the local eigenproblem. Increase
  `lanczos_max_iter` or `lanczos_krylov_dim`.

**How to act.** See decision tree §2.1.

---

### 1.6 `S(t)` — half-chain entanglement entropy *(planned)*

**What it would measure.** S = −Σ σ² log σ², computed from the
Schmidt spectrum across the half-chain bond. The single-most-physical
diagnostic for quench dynamics.

**Healthy.** Linear growth post-quench (Calabrese-Cardy), saturating
once `chi_max_tdvp` becomes the binding constraint. The slope of the
linear growth is set by physics (the quasiparticle velocity); the
saturation level is set by `log(chi_max_tdvp)`.

**Why it's worth adding.** `max_chi(t)` is a discrete proxy. `S(t)`
gives you the continuous picture: when does S(t) cross `log(chi_max)`?
That's the moment you start needing more bond dim than your budget
allows.

We don't currently log it; ~20 LOC to add to `run_tdvp.jl` plus a
helper in `TensorOps/measurements.jl`.

---

## 2. Troubleshooting decision trees

### 2.1 DMRG won't converge

You ran DMRG, the energy isn't where you expected it to be, or it's
still drifting after many sweeps. Work down this list in order:

| Symptom | Diagnose by | Fix |
|---|---|---|
| Energy drifting between sweeps | look at last ~5 sweep energies | run more sweeps |
| Energy plateaued but still high | compute `var(H)` | if large → next row |
| Plateaued, `var(H) < 1e-10` | nothing — you're done | (this is convergence) |
| Plateaued, `var(H) > 1e-6` | likely local minimum or χ-bound | try larger `chi_max_dmrg`; if no help, restart from a different initial MPS (random instead of Néel, or vice versa) |
| Plateaued, `max_trunc` ~ floating-point noise | cutoff and χ are not binding; you found the χ-bound minimum | increase `chi_max_dmrg` |
| Plateaued, `max_trunc > 1e-10` | truncation is biting | tighten `cutoff_dmrg` to 1e-12 or increase `chi_max_dmrg` |
| Local energies look noisy sweep to sweep | Lanczos isn't converging | increase `lanczos_max_iter` (try 30) or `lanczos_krylov_dim` (try 10) |
| Energy *increases* between sweeps | numerical issue (canonical form, roundoff) | tell us — this should not happen for 2-site DMRG |

### 2.2 TDVP norm or energy drifting too fast

You ran TDVP, `1 − ‖ψ‖²(t)` or `dE(t)` exceeds your tolerance before
your target `t_max`. Diagnose:

| First, look at | If… | Then |
|---|---|---|
| `max_chi(t)` | saturates at `chi_max_tdvp` before drift gets bad | **increase `chi_max_tdvp`** (most common case) |
| `max_chi(t)` | stays well below `chi_max_tdvp` but `max_trunc > 1e-10` | `cutoff_tdvp` is biting; tighten it (1e-10 → 1e-12) |
| `max_chi(t)` | stays low, `max_trunc` ~ floating-point noise, but `dE` is constant from t > 0 | **Krylov artifact**; increase `krylov_dim` (14 → 30) |
| `max_chi(t)` | stays low, all `max_trunc` ~ floating-point, but `dE` *grows* | recompute energy after `make_canonical`; if it persists, tell us |
| `dE` stable, `1 − ‖ψ‖²` only slightly above floating-point | you're fine; this is the Krylov approximation residual, not a bug | move on |

The most common verdict: `chi_max_tdvp` is too small for your `t_max`.
You can either (a) raise it, (b) accept the drift and shorten your
analysis window to `t_reliable`, or (c) declare a tolerance on
`1 − ‖ψ‖²` (say, 1e-4) and report only times where the bound holds.

### 2.3 TDVP results disagree across χ values

You ran a χ-scan and the runs disagree. This is the *good* outcome —
you have an explicit measurement of the χ-dependence of your
observable. To use it:

| Pattern | Meaning |
|---|---|
| `M_stag(t)` agrees across all χ | χ-converged in this window. Trust the answer. |
| Curves agree until t* then diverge | t_reliable ≈ t* for the smallest χ in the disagreement. Trust each curve up to its own t_reliable. |
| Curves diverge from t = 0 | smallest χ is too small even at t=0; increase the floor of the ladder |
| Reference (largest) χ shows visible drift in `1 − ‖ψ‖²` | even the reference is under-converged; add a larger rung to the ladder |

The right reference is whichever χ shows `1 − ‖ψ‖²` near floating-point
noise out to `t_max`. If no rung satisfies that, you don't have a
trustworthy answer at `t_max` and need to push χ higher (or accept a
smaller window).

### 2.4 Krylov solver complaints

You see `krylov_dim` warnings, residuals not shrinking, or noisy
energies sweep to sweep:

| What's happening | Fix |
|---|---|
| `krylov_dim` too small for the local 2-site space | increase `krylov_dim` (14 is default; 30 for hard cases) |
| `krylov_tol` too tight for `dt` | loosen `krylov_tol` to 1e-8, or decrease `dt` |
| Loss of orthogonality in Lanczos basis | reduce `krylov_dim` (paradoxically — past ~40, the modified Gram-Schmidt loses orthogonality and *less* is more) |

---

## 3. Reading the CSV — three worked examples

The CSV columns produced by `run_tdvp.jl`:

```
step, t, energy, dE, norm_sq, one_minus_norm_sq, m_stag,
max_chi, max_trunc, total_trunc
```

### Example A: a clean run (truncation never bites)

From `tests/benchmark_tdvp_drift.csv`, N=20 χ_tdvp=128 cutoff=1e-8,
truncated to a few rows:

```
step    t      1−‖ψ‖²       χ_max   max_trunc
   0   0.00   −4.2e-15      64    0.000e+00
  10   0.50   −4.5e-14      77    4.4e-16
  20   1.00   −9.7e-14      99    1.1e-15
```

What to read:
- `1 − ‖ψ‖²` is *negative* and bouncing around 1e-14: this is round-
  off, not real drift. The state is fully unitary so far.
- `χ_max` is climbing: 64 → 77 → 99. The state is becoming entangled
  but is still well below the cap of 128.
- `max_trunc` is at floating-point noise (1e-15 to 1e-16). No real
  truncation is happening yet.

Verdict: this run is clean through t=1.0. Trust the observables.

### Example B: the saturation cliff

Same CSV, later rows:

```
step    t      1−‖ψ‖²       χ_max   max_trunc
  30   1.50   −4.8e-14     128    1.8e-15    ← χ saturates
  35   1.75   +2.4e-13     128    2.1e-14
  40   2.00   +2.9e-12     128    1.7e-13
  50   2.50   +8.4e-11     128    3.3e-12
  60   3.00   +1.4e-09     128    4.3e-11
```

What to read:
- At t=1.50, `χ_max` first equals `chi_max_tdvp = 128`.
- One step later (t=1.75) `max_trunc` jumps from 1e-15 to 2e-14: the
  first real truncation.
- `1 − ‖ψ‖²` *flips sign and starts growing*. From t=1.75 onward,
  every step is bleeding norm.
- The growth is roughly geometric: 2.9e-12 → 1.4e-9, factor 500 over
  Δt=1. Consistent with linear-in-t entanglement growth +
  fixed-budget truncation.

Verdict: trust observables to about t=1.5 at this χ. Beyond, expect
visible χ-dependent corrections. If your physics target is t=3,
double `chi_max_tdvp`.

### Example C: a converged production run

From `scripts/scan_results/tdvp_chi128.csv`, N=24 χ_tdvp=128
cutoff=1e-10, t_max=3:

```
step    t      dE          1−‖ψ‖²       χ_max   max_trunc
  10   0.50   −3.6e-12    −1.4e-13     128    2.2e-16
  30   1.50   −7.3e-12    −7.9e-14     128    8.0e-15
  40   2.00   +8.7e-11    +9.3e-12     128    3.1e-13
  60   3.00   +4.2e-08    +3.8e-09     128    7.7e-11
```

`χ_max` is at the cap from very early (t=0.5), but `cutoff_tdvp = 1e-10`
keeps `max_trunc` near floating-point noise until t≈2.0. Final drift at
t=3 is `1 − ‖ψ‖² ~ 4e-9`: **paper-quality**.

Why is this so much cleaner than Example B at the same `χ_max = 128`?
Because the cutoff is tighter (1e-10 here vs 1e-8 there) and the state
at this Δ_f has slightly less entanglement growth. Same algorithm, same
χ — different physics, dramatically different drift.

The lesson: **`chi_max_tdvp` and `cutoff_tdvp` interact**. For
production-quality runs, set `cutoff_tdvp = 1e-10` and let `chi_max_tdvp`
do the work.

---

## 4. First-run checklist

After a TDVP run finishes, before you trust any observable:

1. **Did DMRG converge?** Check `var(H)` from the post-DMRG print. Should
   be `1e-10` or smaller.
2. **What's `1 − ‖ψ‖²(t_max)`?** If above your accuracy target (1e-4 for
   paper-quality observables), the trajectory is unreliable past some
   earlier time.
3. **When did `max_chi` saturate?** That moment marks the start of
   truncation-driven drift. If your physics target is past saturation,
   you need a χ-scan to know how much you can trust.
4. **What's `max_trunc(t_max)`?** Above 1e-8 means significant per-step
   norm bleeding.
5. **Is `dE(t)` consistent with the truncation story?** `dE` should
   roughly track `1 − ‖ψ‖²` — both go up together, and both stay near
   floating-point until truncation begins.

If steps 1-5 all pass at your tolerance, the run is trustworthy. If any
fail, see the decision trees above.

---

## 5. When the diagnostics tell you to stop

There is no shame in declaring `t_reliable < t_max` and reporting
results only up to `t_reliable`. This is *exactly* what diagnostic
infrastructure is for. Pushing χ higher to extend `t_reliable` has
exponentially diminishing returns once entanglement saturates the bond
budget.

A scan over χ — `scripts/scan_tdvp_chi.jl` — gives you direct evidence
of where each χ becomes unreliable. Use the script. Plot the four
quantities (`M_stag`, `1 − ‖ψ‖²`, `max_chi`, `max_trunc`) for each χ
on the same axes; the time at which observables stop overlaying with
the next-larger-χ run is `t_reliable` for the smaller χ.
