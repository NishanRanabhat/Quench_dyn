# Finite-temperature states via ancilla purification (imaginary-time TDVP)

A protocol for generating the thermal state ρ(β) = e^{−βH}/Z of a 1D chain as a
matrix-product state and measuring thermal observables from it. Written to be
implementable in any MPS/TDVP codebase; a validated Julia reference lives in
this repo (see **Reference implementation** at the end).

## The idea

Represent the mixed state ρ(β) as a **purification**: a pure state |ψ_β⟩ on an
enlarged Hilbert space physical ⊗ ancilla, such that tracing out the ancilla
returns ρ:

    ρ(β) = Tr_anc |ψ_β⟩⟨ψ_β| .

The trick that keeps the code simple: **merge the physical and ancilla index of
each site into a single leg of dimension d² = 4** (for spin-½). Then |ψ_β⟩ is an
*ordinary* MPS with local dimension 4, and the *ordinary* MPS TDVP machinery
evolves it — no special "MPDO" contraction routines are needed. The ancilla is
carried along as a spectator.

Convention (matters for every embedding below): merged index

    k = (s − 1)·d + a ,   s = physical (outer),  a = ancilla (inner).

With this ordering, "act with operator O on the physical leg, identity on the
ancilla" is exactly the Kronecker product `kron(O, I_d)`.

## The three steps

**1. Infinite-temperature seed (β = 0).**
Product state, bond dimension 1, each site the maximally-entangled physical–
ancilla pair

    |site⟩ = (1/√d) Σ_s |s⟩_phys |s⟩_anc      ⇒   tensor[1, (s−1)d+s, 1] = 1/√d.

Tracing the ancilla gives ρ = (I/d)^{⊗N} = maximally mixed = infinite T. Its
norm ⟨ψ|ψ⟩ = Tr ρ = 1.

**2. Cool by imaginary-time TDVP under H ⊗ I_anc.**
Build the Hamiltonian MPO on the merged space by tensoring an identity onto the
ancilla of every physical MPO block:

    W'[l, r, :, :] = kron( W[l, r, :, :], I_d )     (MPO bond dimension unchanged).

Evolve |ψ⟩ in imaginary time with the standard 2-site TDVP, but replacing the
real-time local propagator exp(−i·Δt·H_eff) by the **imaginary** one
exp(−Δt·H_eff). Accumulate imaginary time τ. Because tracing the ancilla gives
ρ ∝ e^{−2τH}, the inverse temperature is

    β = 2τ .

Cool until the desired β (or until ⟨H⟩ reaches a target energy — see below).
Start from the bond-dimension-1 seed and let the 2-site SVD grow χ.

**3. Thermal observables.**
For any physical single-site operator O,

    ⟨O_i⟩_β = ⟨ψ_β| (O ⊗ I_anc)_i |ψ_β⟩ / ⟨ψ_β|ψ_β⟩ ,

i.e. measure `kron(O, I_d)` on the merged-index MPS with the ordinary MPS
measurement routine, and divide by the norm ⟨ψ|ψ⟩ = Tr ρ. Two-point functions
⟨O_i O_j⟩_β are the same with the two operators embedded and inserted. (The
⊗ I_anc automatically performs the ancilla trace.)

## Cooling to a target energy (for a quench's effective temperature)

For quench physics you usually want the thermal state at the energy the quench
deposits, E_target = ⟨ψ_0|H_f|ψ_0⟩ (ψ_0 = initial state, H_f = post-quench
Hamiltonian). E_target is a *physical* (d-dimensional) energy, and it equals the
purified ⟨H⟩ = Tr(ρ H) directly (the ⊗ I_anc adds no energy). So: cool while
monitoring ⟨H⟩ = ⟨ψ|H⊗I|ψ⟩ / ⟨ψ|ψ⟩, and **stop at the round-trip where ⟨H⟩
first drops to E_target**. The β reached is the effective inverse temperature;
T_eff = 1/β. (⟨H⟩ decreases monotonically from the infinite-T mean toward the
ground state, so the crossing is unique.)

## Observables

Because the purification is an *ordinary* MPS with the ancilla merged into the
local leg, **every thermal observable is measured with the existing MPS
measurement routines — no ancilla-aware ("MPDO") measurement code is needed.**
There are only two rules:

1. **Embed** the physical operator onto the merged leg: `embed_physical(O)` =
   `kron(O, I_d)`. The ⊗ I_anc performs the ancilla trace automatically.
2. **Normalize** by the trace `Tr ρ = ⟨ψ|ψ⟩ = measure_norm(state.mps)` — thermal
   expectations are ratios.

So for any physical single-site operators `O`, `P`:

```julia
⟨O_i⟩_β     = measure_local_observable(mps, embed_physical(O), i) / measure_norm(mps)
⟨O_i P_j⟩_β = measure_correlation(mps, embed_physical(O), i,
                                       embed_physical(P), j) / measure_norm(mps)
```

### What can be measured

- **Energy density** ⟨H⟩_β / N — `measure_energy(state) / measure_norm(state.mps)`
  (the purified MPO already carries H ⊗ I). This is the cooling diagnostic and
  the quantity used for energy → temperature matching.

- **Longitudinal (CDW / Néel) channel** ⟨S^z_i S^z_j⟩_β — from
  `thermal_sz_from_purification(state)` → `(m, C)`. Feed the matrix `C` to the
  coarsening observables:
  - `staggered_correlator(C)` → `G(r)` → `domain_length` = **staggered
    correlation length ξ(T)** (the coarsening ceiling);
  - `structure_factor(C, qs)` → **S(q)**; the q = π peak is the CDW response;
  - `staggered_magnetization_sq(C)` = S(π)/N = **⟨m_s²⟩** (order parameter²).

- **Local magnetization** ⟨S^z_i⟩_β — the CDW one-point profile (≡ 0 at h = 0 by
  symmetry, nonzero with a field). Returned as `m` above, or
  `measure_local_observable(mps, embed_physical(Sz), i) / measure_norm(mps)`.

- **Transverse (XY) channel** ⟨S^+_i S^-_j⟩_β and ⟨S^x_i S^x_j + S^y_i S^y_j⟩_β —
  physically distinct from the longitudinal channel in XXZ (different decay /
  exponent), so measure it if the physics cares about both. Same machinery:
  ```julia
  Sp = embed_physical(spin_ops(d)[:Sp]; d=d);  Sm = embed_physical(spin_ops(d)[:Sm]; d=d)
  ⟨S⁺_i S⁻_j⟩_β = measure_correlation(mps, Sp, i, Sm, j) / measure_norm(mps)
  ```

- **Connected correlators** Cᶜ_ij = ⟨S^z_i S^z_j⟩ − ⟨S^z_i⟩⟨S^z_j⟩ —
  `connected_matrix(C, m)`; the seed-independent / parity-even signal (Cᶜ = C
  whenever m ≡ 0, e.g. at h = 0).

- **Any physical operator** — S^x, S^y, bond operators, currents, … : all work
  through `embed_physical` + normalize; no new observable functions required.

- **Thermodynamics** (from the U(β) trajectory the run script logs):
  - internal energy U(β) = ⟨H⟩_β (recorded at each measurement step);
  - specific heat C_v(T) — either finite-difference dU/dT along the trajectory
    (robust), or C_v = β²·Var(H) with Var(H) = ⟨H²⟩_β − ⟨H⟩_β². **Caveat:**
    `energy_variance` assumes a unit-norm state, so for the purification either
    normalize ⟨H²⟩ and ⟨H⟩ by Tr ρ or rely on the per-step center
    renormalization (⟨ψ|ψ⟩ ≈ 1);
  - entropy S(T) / free energy — by thermodynamic integration of C_v(T), or from
    ln Z tracked through the norm decay during cooling.

- **Operator-space (bond) entanglement** — the bond entropy of the purified MPS
  is the operator entanglement of ρ^{1/2} (it sets χ; see the scaling notes),
  a *different* quantity from pure-state entanglement — interpret accordingly.

`run_finite_T.jl` logs the energy density, ξ, and ⟨m_s²⟩ at each measurement
step; add any of the above inside its measurement block with the two-rule
pattern.

## Practical knobs and their meaning

| knob | role | guidance |
|------|------|----------|
| `dt` | imaginary-time step per TDVP round-trip | error is O(dt²); dt ≈ 0.05–0.1 is a good balance. Smaller near a target if you need β precise. |
| `chi_max` | bond-dimension cap of the purification | this is the accuracy/cost knob. **Converge ξ (and any observable) in χ** before trusting it — cold, near-critical states need larger χ. |
| `cutoff` | SVD singular-value discard threshold | 1e-9–1e-10. |
| `krylov` dim / tol | local matrix-exponential accuracy | 8 / 1e-9 suffices for cooling; larger is wasted cost. |

## Cost and scaling notes (important when scaling up)

- **χ, not β, drives cost.** Unlike *real*-time evolution (where entanglement
  grows ballistically with t), imaginary-time cooling entanglement **saturates**
  for gapped systems and grows only **logarithmically at criticality**. So χ
  *peaks near the critical point* (here Δ → 1⁺) and is modest away from it. No
  ancilla "disentangler" is needed for pure cooling (it only helps real-time
  finite-T dynamics).
- **Always check χ-convergence** of the reported quantity, especially for cold
  (β large) and near-critical (Δ near 1) states, where χ is largest.
- **Speed-up for scaling:** grow χ with a few 2-site sweeps, then switch to
  **1-site** TDVP (bond-preserving, no SVD — much cheaper per sweep) to cool the
  rest of the way at fixed χ. Mirror the DMRG "2-site warmup + 1-site refine"
  pattern.
- **Numerical range:** e^{−2τH} over/underflows; renormalize the state
  periodically (rescale the orthogonality-center tensor to unit norm). All
  observables are ratios, so this is free of physical consequence. Renormalize
  more frequently (per sweep, or per site) if you hit overflow at large N·β.
- **Ensemble:** the standard purification samples the **full** canonical
  ensemble (all magnetization sectors). A quench conserving S^z_total selects
  the S^z=0 sector; the two agree for local observables at matched energy
  density and converge as N→∞ (O(1/N) at small N). If you need the sector
  exactly, use a U(1)-symmetric purification (ancilla carries opposite charge).

## Reusing / saving states (optional)

Cooling is the expensive step, and one run passes through every intermediate
temperature — so if you want a whole temperature series, measure (and if you
like, save) the state at intervals inside the step loop. The reference script
`run_finite_T.jl` logs observables to a CSV at each measurement (like the TDVP
scripts) and does **not** serialize the state by default. To reuse a state
later, serialize its purification tensors (local dim d²) plus metadata (N, J, Δ,
β, and the merged-index convention); reload and measure any physical operator O
as `embed_physical(O)` divided by ⟨ψ|ψ⟩. Julia's `Serialization` works with zero
dependencies; prefer HDF5/JLD2 for cross-codebase portability.

## Reference implementation (this repo)

- `src/Builders/purification.jl` — `maximally_mixed_purification`, `purify_mpo`,
  `embed_physical`.
- `src/Algorithms/purification_tdvp.jl` — `thermal_sz_from_purification` → (m, C);
  `cool_purification!` (a single-call helper that cools to a target β *or*
  energy — handy for matching a quench's deposited energy).
- `scripts/run_finite_T.jl` — the **pipeline**: an explicit imaginary-time
  step loop, structured exactly like `run_tdvp_2site.jl` (set `beta_max`,
  measure every `measure_every` steps, CSV log).
- `tests/test_purification.jl` — validation against exact ED (⟨H⟩(β) to ~1e-8,
  plus (m, C) and ξ).

The reference reuses the repo's existing MPS 2-site TDVP unchanged; the only
finite-T-specific code is the three small embeddings above and the cooling loop.
