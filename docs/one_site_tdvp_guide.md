# One-site TDVP, padding, and the χ-scan: a guide

This document explains the quench protocol used by
`scripts/run_tdvp_1site.jl` and `scripts/scan_tdvp_chi_1site.jl`. It is
written for someone comfortable with quantum mechanics and numerical
methods but new to tensor networks.

Read order: top to bottom. Each section builds on the previous one.

---

## 1. What we are actually computing

We want to simulate a **quench**: prepare the ground state of one
Hamiltonian, then suddenly change the Hamiltonian and watch the state
evolve in real time. Concretely, for our XXZ chain:

```
H(Δ) = J Σᵢ (Sxᵢ Sxᵢ₊₁ + Syᵢ Syᵢ₊₁) + Δ Σᵢ Szᵢ Szᵢ₊₁ + Σᵢ hᵢ Szᵢ
```

The protocol is:

1. Find the ground state of `H(Δᵢ)` → call it `|ψ₀⟩`.
2. Time-evolve under `H(Δf)`: `|ψ(t)⟩ = exp(-i H(Δf) t) |ψ₀⟩`.
3. Measure observables like `⟨Sz_i⟩(t)`, `M_stag(t)`, etc.

Doing this naively requires storing `|ψ⟩` as a `2^N`-dimensional vector,
which is intractable beyond `N ≈ 20`. Tensor network methods get
around this by **compressing** the state into a structure called a
Matrix Product State (MPS), parameterised by a small "bond dimension"
`χ`. The cost scales like `N · χ³` rather than `2^N`. The catch:
states with too much entanglement cannot be represented at any finite
`χ` — but during real-time dynamics of an initially weakly entangled
state, entanglement grows linearly in time, so we have a time window
where MPS works.

You don't need to understand MPS internals for this guide. All you need
to know is:

- An MPS at bond dimension `χ` lives in a curved manifold inside the
  full Hilbert space.
- The manifold gets bigger as `χ` grows. At `χ = 2^(N/2)` for spin-1/2,
  the MPS manifold IS the full Hilbert space (no compression).
- Time evolution within this manifold needs an algorithm that respects
  the manifold's geometry. That algorithm is **TDVP** (Time-Dependent
  Variational Principle).

---

## 2. The drift problem (why we needed to change the algorithm)

You noticed earlier that energy and norm were drifting in the
quench runs. This is not a code bug — it is a structural feature of
the **2-site TDVP** algorithm that we were using. Here is what was
happening.

A 2-site TDVP sweep does this at each bond:

1. Contract two neighbouring MPS tensors into one big block (4 indices).
2. Apply `exp(-i H_eff dt/2)` to that block (a small Krylov problem).
3. **Singular Value Decomposition (SVD)** of the result; keep only the
   top `χ_max` singular values; discard the rest.
4. Move to the next bond and repeat.

Step 3 is the truncation. It is mathematically necessary — the evolved
block has higher rank than the original, and you must throw something
away to keep the MPS bond bounded. But the discarded weight is exactly
what destroys unitarity:

- `‖ψ‖²` is no longer exactly 1 (norm drift).
- `⟨H⟩` is no longer conserved (energy drift).
- The errors accumulate over time.

Concretely, if the singular value spectrum of the evolved 2-site block
is `σ₁ ≥ σ₂ ≥ ...`, truncating to `χ_max` discards a weight
`Σ_{k > χ_max} σ_k²`. For early times this is tiny (low entanglement,
spectrum decays fast). As entanglement grows during the quench, the
spectrum gets fatter, more weight sits below `χ_max`, and drift gets
worse. The error you reported is exactly this mechanism saturating.

**This is why the size of the drift IS the algorithm's truncation
error** when you run 2-site TDVP. You and your collaborator were
correctly identifying a real problem.

---

## 3. Enter 1-site TDVP: same equations, different splitting

There is a second TDVP algorithm that updates one MPS tensor at a
time, with a "back-evolve the bond" correction in between. It is
exactly the same TDVP equations, just split across one site instead
of two. The key difference:

| Step | 2-site TDVP | 1-site TDVP |
|---|---|---|
| Decomposition | SVD (truncates) | QR or LQ (does not truncate) |
| Bond dimension | Can grow (up to `χ_max`) | **Fixed** at whatever it started at |
| Norm conservation | Approximate (truncation) | **Exact** (up to Krylov error) |
| Energy conservation | Approximate (truncation) | **Exact** (up to Krylov error) |
| Where it lives | A growing manifold (capped at `χ_max`) | A fixed-`χ` manifold |

The mathematical reason: 1-site TDVP is the *exact* projection of the
Schrödinger equation onto the tangent space of the fixed-`χ` manifold.
The projection is unitary. There is no SVD anywhere in the algorithm,
so there is nothing to truncate, so there is no discarded weight, so
norm and energy are conserved exactly.

In code (`src/Algorithms/tdvp.jl`):

- `tdvp_sweep` (2-site) calls `_svd_truncate` and reports `max_trunc`,
  `total_trunc`. Bond dim grows up to `chi_max`.
- `tdvp_sweep_one_site` calls only `qr` / `lq`. There is no `chi_max`,
  no `cutoff`; those fields of `TDVPOptions` are silently ignored.

You can verify on a small test: at `N=12, χ_work=64` (= the full
Hilbert manifold, where 1-site TDVP becomes equivalent to exact
diagonalisation), the drift is at the Krylov floor `~1e-9`, not the
truncation floor.

---

## 4. The catch: 1-site cannot grow `χ`

Here is the price you pay. If you start with a low-`χ` MPS (say a
product state, `χ=1`) and run 1-site TDVP, the algorithm cannot
expand `χ` — by construction it lives in a fixed-`χ` manifold. So
1-site TDVP is useless from a product-state start: you would be
forever stuck at `χ=1`.

Similarly, your DMRG ground state typically has `χ_DMRG` chosen to
accurately represent the (weakly-entangled) ground state. The
*post-quench dynamics* generally needs a larger `χ_work` because
entanglement grows. If you go straight from `DMRG (χ=χ_DMRG)` into
1-site TDVP, you are stuck at `χ_DMRG` for the whole evolution —
much too small to represent the post-quench state.

You need a way to lift a `χ_DMRG` MPS into a larger `χ_work` MPS
without changing what state it represents. That operation is **MPS
padding**.

---

## 5. Padding: lifting an MPS into a bigger manifold

The padding utility (`src/TensorOps/padding.jl`, function `pad_mps`)
does exactly this. Conceptually, for each MPS tensor it:

1. Creates a new, larger tensor with extra "slots" along the bond
   indices.
2. Copies the original tensor data into the top-left-front block of
   the new tensor.
3. Fills the new (extra) entries with small random Gaussian noise of
   magnitude `ε` (default `ε = 1e-6`).
4. Re-canonicalises the resulting MPS (SVD-based) and renormalises to
   `‖ψ‖² = 1`.

In code (`src/TensorOps/padding.jl:90-91`):

```julia
B = T.(noise) .* randn(T, chi_l_new, d, chi_r_new)
B[1:chi_l_old, :, 1:chi_r_old] .= A
```

Whole new tensor is filled with `ε · randn`, then the original block
is overwritten in place.

### 5.1 What state does the padded MPS represent?

Heuristically:

```
|ψ_pad⟩ ≈ |ψ_orig⟩ + ε · |φ⟩
```

where `|φ⟩` lives in the "new" directions that didn't exist in the
original MPS. After renormalisation:

- Overlap: `|⟨ψ_orig | ψ_pad⟩|² ≈ 1 - O(ε²) ≈ 1 - 10⁻¹²` for `ε=1e-6`.
- Energy: `⟨ψ_pad | H | ψ_pad⟩ ≈ E_orig + O(ε²)` (about `1e-12` shift).
- Local observables: shifted by `O(ε²)` too.

**So padding does not dilute the state to any meaningful precision.**
You can — and the script does — check this directly:

```
DMRG energy   = -4.4633110742
⟨ψ_pad|H_pre|ψ_pad⟩ = -4.4633110742
|ΔE|          = 5.24e-14   (well below ε² = 1e-12)
```

That is from the N=12 scan further below.

### 5.2 Why noise and not zeros?

This is the most subtle point in the whole protocol, and it is what
makes the protocol *work* rather than fail silently. Here is the
reasoning.

If you padded with zeros instead of noise, the new bond directions
would carry literally zero amplitude. The state vector would be
exactly `|ψ_orig⟩` lifted into a larger MPS structure, with the new
directions sitting at exactly 0.

Then you run 1-site TDVP. The algorithm projects the Schrödinger
equation onto the *tangent space* of the current MPS. But the tangent
space at directions where amplitude is zero is also zero — there is
no infinitesimal perturbation that uses those directions while
keeping the state normalised. Concretely: the local effective
Hamiltonian `H_eff` couples to the new directions, but the projector
onto those directions vanishes because there is no amplitude there.

The net effect: dynamics cannot leak into the new directions. They
stay zero forever. Your beautiful enlarged manifold goes unused, and
1-site TDVP behaves as if it were still at `χ_DMRG`. Quench-driven
entanglement growth is silently capped.

With Gaussian noise at the new directions:

- The amplitudes there start at `O(ε)`.
- `H_eff` now has a non-zero handle to push real weight into them.
- Quench-driven correlations *can* grow into the enlarged manifold.

So the padding noise is not a perturbation we tolerate — it is the
**seed amplitude that makes the larger manifold usable**.

### 5.3 The noise-amplitude trade-off

From `padding.jl`'s docstring:

- `ε = 1e-14`: padded amplitudes at numerical zero → as bad as
  zeros, dynamics cannot use the new directions.
- `ε = 1e-6` (default): observable shift `≈ ε² = 1e-12`, invisible to
  any measurement, but enough seed amplitude for dynamics.
- `ε = 1e-3`: observable shift visible in the 6th decimal of `⟨Sz⟩`.
  Too much.

`1e-6` is the standard compromise. Do not change it unless you have a
specific reason.

### 5.4 Per-bond padding rules

`pad_mps` does not blindly pad every bond to `χ_work`. It respects
the **natural ceiling** at each bond: bond `i` (separating `i` sites
on the left from `N-i` on the right) cannot have Schmidt rank higher
than `min(d^i, d^(N-i))`. There is no point padding a bond past this
ceiling — those directions are mathematically empty.

So for `N=12, d=2`, the bond profile after padding to `χ_work = 64`
might look like:

```
bonds: [2, 4, 8, 16, 32, 64, 32, 16, 8, 4, 2]
```

The boundary bonds stay tiny; only the middle bonds grow to `χ_work`.
Implementation in `padding.jl:47-50`.

---

## 6. The full 1-site protocol

Putting it together, the protocol used by
`scripts/run_tdvp_1site.jl`:

```
  Start: product state (Néel) at χ=1
      │
      ▼
  STEP 1: 2-site DMRG at χ_dmrg
      Grows χ from 1 → χ_dmrg via SVD.
      Converges to ground state of H(Δᵢ).
      Output: an MPS at χ = χ_dmrg with E ≈ E_gs.
      │
      ▼
  STEP 2: pad_mps(state, χ_work; noise=1e-6)
      Lifts the χ_dmrg MPS into the χ_work manifold.
      Adds O(1e-6) noise in the new bond directions.
      Sanity check: ⟨ψ_pad|H_pre|ψ_pad⟩ should match E_gs to ~1e-12.
      │
      ▼
  STEP 3: Switch MPO to H(Δf), rebuild environments at χ_work
      The pre-quench Hamiltonian is replaced.
      State is unchanged; only the operator changes.
      │
      ▼
  STEP 4: 1-site TDVP loop
      For each timestep:
          tdvp_sweep_one_site(:right)   # dt/2 forward at each site
          tdvp_sweep_one_site(:left)    # dt/2 forward (other direction)
      χ stays fixed at χ_work throughout.
      Norm conserved to ~Krylov tol.
      Energy conserved to ~Krylov tol.
```

Step 1 uses 2-site DMRG because 1-site DMRG also cannot grow `χ`. The
χ-growth phase needs SVD; refinement and evolution do not.

---

## 7. How do you know your `χ_work` is big enough?

This is the central methodological question for 1-site TDVP. In the
2-site protocol the algorithm itself reports a truncation error per
step, and you can read off "I am discarding `1e-5` weight per sweep"
or whatever. In the 1-site protocol, **the algorithm has nothing to
report** — it does not truncate, so there is no discarded weight to
log. The price for exact unitarity is the loss of an automatic
in-algorithm error signal.

So how do you decide if `χ_work` is adequate? Two complementary
diagnostics:

### 7.1 The `χ`-scan (the gold standard)

Run the protocol at several `χ_work` values (e.g. `[64, 96, 128]`),
compute the same physical observable trajectory each time, and check
that it stabilises as `χ` grows. If `M_stag(t; χ=128)` agrees with
`M_stag(t; χ=96)` to high precision, you are converged. If they
disagree, `χ` is too small.

This is implemented in `scripts/scan_tdvp_chi_1site.jl`. The script
runs DMRG once, then loops over `chi_ladder`, padding fresh and
evolving each time. Output: one CSV per `χ` plus a cross-`χ`
convergence table.

### 7.2 The Schmidt-spectrum tail (per-run diagnostic)

Even without comparing across `χ` runs, you can ask of a single
trajectory: "is my MPS bumping the manifold ceiling?" Compute the
SVD at every bond (outside the algorithm — purely diagnostic) and
look at the **smallest singular value** at the most entangled bond.

- If that smallest σ is `~1e-9` or below: there is room. The MPS
  effectively uses fewer than `χ_work` Schmidt values; dynamics has
  not yet saturated the manifold.
- If it is `~1e-2`: the manifold is saturated. The "smallest" σ is
  carrying real weight, which means singular values above it are
  even fatter, and at the next time step you might want to discard
  some of them — except you can't, you are at `χ_work`. Grow `χ`.

The script logs this as `σ_tail_min` at every measurement step.

### 7.3 Why energy/norm drift do NOT tell you about `χ_work`

This is a common confusion worth stating explicitly. In 1-site TDVP,
energy and norm are conserved *by construction on a fixed manifold*.
So if you see drift at the `Krylov tol` level (`~1e-9`), that drift
is **Krylov error**, not manifold-truncation error. Cranking up
`χ_work` will not fix it; tightening `krylov_tol` and increasing
`krylov_dim` will. Conversely: if your `χ_work` is way too small,
energy and norm will still be conserved beautifully — the algorithm
faithfully evolves a *wrong* state in the wrong manifold. Drift is
silent on manifold adequacy.

This is the opposite of the 2-site case, where drift = truncation
error. Make sure your collaborator understands this inversion. The
diagnostic you can no longer use is energy/norm drift; the
diagnostic you must learn to use is the `χ`-scan + Schmidt tail.

---

## 8. Reading scan output: a worked example

This is the actual output from a quick scan we ran:

```
Model:  XXZ chain, N=12, J=1.0, Δᵢ=0.5 → Δf=3.0
Field:  hᵢ = 0.1 · (-1)ⁱ  (staggered)
DMRG:   χ_dmrg=16, converged with var(H) = 9.2e-08
χ ladder: [16, 32, 64]    (χ=64 = full manifold for N=12)
Evolution: dt=0.05, t_max=1.0
```

Per-run summary at `t = t_max = 1.0`:

```
χ    M_stag       |ΔM vs χ=64|    σ_tail_min    max|ΔE|     max|1-‖ψ‖²|
16   -0.294546    9.0e-7          2.4e-4        1.04e-8     1.22e-9
32   -0.294546    2.5e-9          2.3e-6        7.97e-9     9.36e-10
64   -0.294546    0 (reference)   5.5e-9        2.02e-9     2.37e-10
```

**How to read this row by row:**

- `M_stag` is the staggered magnetisation. To six decimal places, all
  three runs agree — the physics is converged at `χ=16` already for
  this small system, short evolution.
- `|ΔM vs χ=64|` is the discrepancy from the reference (the full
  manifold run). Drops by ~2 orders of magnitude per doubling of `χ`.
  This is fast convergence; the dynamics is not particularly demanding
  at these parameters.
- `σ_tail_min` is the manifold-adequacy signal. At `χ=16` the smallest
  σ at the worst bond is `2.4e-4` — there is real weight at the
  manifold ceiling. By `χ=32` it has dropped to `2.3e-6`. By `χ=64`
  it is `5.5e-9` — full headroom.
- `max|ΔE|` is the energy drift over the whole evolution. *Decreases*
  with `χ`: this is Krylov error improving as the local effective
  space gets larger.
- `max|1-‖ψ‖²|` is the norm drift. Same Krylov-error story.

**Reading across, what the table is telling you:**

1. The protocol works: drift is uniformly at the Krylov floor
   (`~10⁻⁹`), no truncation contribution.
2. The `χ`-ladder convergence is monotone and rapid.
3. The Schmidt-tail diagnostic is consistent with the cross-`χ`
   diagnostic: both signal that `χ=64` has full headroom and
   anything smaller is borderline (`χ=16`) or comfortable (`χ=32`).
4. For *this* problem at *this* time, `χ=32` would be a safe
   production choice. For larger `N` or longer `t`, you would need
   to redo the scan.

### 8.1 Red flags to watch for in a scan

- **Non-monotone `|ΔM|`**: if `|M(χ=128) - M(χ=96)|` is larger than
  `|M(χ=96) - M(χ=64)|`, your top of ladder is not converged. Add a
  larger `χ`.
- **`σ_tail_min` growing with `t`** rather than oscillating: the
  state is being squeezed into the manifold ceiling and you are
  losing physics. Stop trusting the run beyond the time where
  `σ_tail_min` exceeds, say, `1e-4`.
- **Drift *increasing* with `χ`**: Krylov subspace dim is too small
  for the larger local space. Bump `krylov_dim` (15 → 20 → 30) and
  tighten `krylov_tol`.

---

## 9. Practical recipe

For a new physics problem:

1. **Estimate the ground-state `χ_DMRG`** with a single
   `run_dmrg_2site.jl` run. Watch the energy converge with sweeps
   and the truncation error fall to `~ cutoff`. A good `χ_DMRG`
   has trailing-sweep energy stable to ~10⁻¹⁰ and `total_trunc <
   1e-8`.
2. **Estimate `χ_work` you will likely need** with a
   `scan_tdvp_chi_1site.jl` run on the smallest interesting `N`
   and shortest `t_max` that makes physical sense. Start with
   `chi_ladder = [chi_DMRG, 1.5 · chi_DMRG, 2 · chi_DMRG]`.
3. **Read the scan summary.** If observables agree across the
   ladder *and* `σ_tail_min` is below `1e-6` at `t_max`, you are
   converged. Use the smallest converged `χ` for production.
4. **If the scan shows the top of the ladder is not converged**:
   expand the ladder upward, or stop trusting the run beyond the
   time where the lower `χ` started to deviate.
5. **For production**, run `run_tdvp_1site.jl` with the converged
   `χ_work`. Still log Schmidt tail per step so you can spot any
   late-time saturation.

Parameter defaults that work for this XXZ project:

| Parameter | Where set | Typical |
|---|---|---|
| `chi_dmrg` | DMRG ground state | 64 |
| `chi_work` | TDVP working manifold | 128 |
| `pad_noise` | padding amplitude | 1e-6 |
| `dt` | TDVP time step | 0.05 |
| `krylov_dim` | Krylov subspace dim | 14 |
| `krylov_tol` | Krylov convergence | 1e-8 |

---

## 10. When to still use 2-site

The 1-site protocol is the right default for *production quench
dynamics*, but the 2-site routines are not obsolete. Use 2-site:

- **For DMRG ground-state preparation.** This is non-negotiable —
  1-site DMRG cannot grow `χ` from a product state. Always 2-site
  for χ-growth, then optionally 1-site for refinement.
- **For exploratory TDVP runs** where you do not yet know what
  `χ_work` you need. 2-site grows `χ` adaptively up to `χ_max`,
  and the truncation error tells you when you are running into
  trouble. Useful for "is this problem even feasible?" scans.
- **For benchmarking.** Comparing 1-site and 2-site results at the
  same `χ` is a useful cross-check: in regimes where 2-site
  truncation is negligible, the two should agree.

Both pipelines live side-by-side in `scripts/`:

```
run_dmrg_2site.jl
run_dmrg_1site.jl     # = 2-site warmup + 1-site refinement
run_tdvp_2site.jl
run_tdvp_1site.jl     # = 2-site DMRG + pad + 1-site TDVP
scan_tdvp_chi.jl      # 2-site χ-scan
scan_tdvp_chi_1site.jl # 1-site χ-scan
```

---

## 11. Summary in one paragraph

The 2-site TDVP algorithm we started with had to truncate via SVD at
every step, and the discarded weight showed up as drift in energy and
norm — the drift you noticed was real and was telling you how much
truncation error the algorithm was accumulating. The 1-site TDVP
algorithm uses QR/LQ instead of SVD, lives in a fixed-`χ` manifold,
and is exactly unitary there — no drift, no truncation. The price is
that 1-site cannot grow `χ`, so we prepare the ground state with
2-site DMRG and then **pad** the MPS up to the working `χ`, filling
the new bond directions with `1e-6` random noise so that 1-site TDVP
has something to push amplitude into. To check that `χ_work` is big
enough we run a **`χ`-scan** and watch observables converge as `χ`
grows; we also log the Schmidt-spectrum tail at every measurement
step as a per-run manifold-adequacy signal. Energy and norm drift in
1-site TDVP measure only Krylov error, not manifold truncation — they
no longer serve the diagnostic role they played in 2-site, and this
is the most important conceptual switch to internalise.
