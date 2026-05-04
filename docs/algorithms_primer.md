# Algorithms Primer

For a physicist new to matrix product states. We assume you know the
Hamiltonian we are simulating and what observables you want; we explain
*why* the numerical machinery is shaped the way it is and *what* it
promises (and does not promise) about the answers it gives you.

The running example throughout is the 1D XXZ chain that QuenchDyn implements:

  H = J Σᵢ (Sˣᵢ Sˣᵢ₊₁ + Sʸᵢ Sʸᵢ₊₁) + Δ Σᵢ Sᶻᵢ Sᶻᵢ₊₁ + Σᵢ hᵢ Sᶻᵢ

You should leave this document understanding (i) what bond dimension is
and why we have to truncate, (ii) what DMRG and 2-site TDVP each
*promise*, and (iii) why 2-site TDVP **necessarily** breaks norm and
energy conservation under truncation.

---

## 1. MPS in 60 seconds

A quantum state on N sites with local Hilbert dimension d lives in a
d^N-dimensional Hilbert space. For N=32 spin-1/2's, that's 2³² ≈ 4
billion amplitudes — infeasible. We need a compact representation.

The key empirical fact: physical states (ground states of local
Hamiltonians, low-energy states, states reachable by physical time
evolution from a product state) have **bounded bipartite entanglement**.
Cut the chain in two halves; the entanglement entropy across the cut is
typically O(1) for gapped systems, O(log L) for critical systems, and
grows linearly in time during a quench. It almost never approaches the
maximum O(L) that a generic Hilbert-space vector would have.

A **matrix product state** (MPS) exploits this. Each site i gets a
rank-3 tensor `A[i]` of shape `(χ_left, d, χ_right)`:

```
  A[i]:    χ_left ──┐
                    │── d  (physical leg)
                    │
                    └── χ_right
```

The full state is the contraction of all the tensors along their bond
legs:

  |ψ⟩ = Σ_{s₁...s_N}  A[1]^{s₁} · A[2]^{s₂} · ... · A[N]^{s_N}  |s₁...s_N⟩

The dimensions χ_left and χ_right of the bond legs are the **bond
dimensions**. They control the expressive power of the state: across any
cut between site i and i+1, the Schmidt rank is at most χ, and therefore

  S(cut) ≤ log(χ).

A product state has χ=1 everywhere. A maximally entangled state needs
χ = d^{N/2}. For our XXZ chain at N=32, χ between 64 and 256 typically
suffices for good accuracy.

**Truncation.** When an algorithm produces an MPS bond bigger than our
budget χ_max, we SVD-truncate: keep the largest χ_max singular values,
discard the rest. This is a *projection*, and it does not preserve the
norm of the state — `‖ψ_truncated‖² = ‖ψ‖² − (sum of squared discarded
singular values)`. The discarded weight is the truncation error. This
is the entire reason the `1−‖ψ‖²` diagnostic exists.

**Cost** of MPS arithmetic scales as O(N · χ³ · poly(d, D)). Doubling
χ multiplies wall time by ~8.

---

## 2. MPO in 60 seconds

A **matrix product operator** (MPO) represents an operator (typically
the Hamiltonian) as a chain of rank-4 tensors `W[i]` of shape
`(D_left, D_right, d, d)`. The two physical legs are bra and ket.

For local Hamiltonians, the MPO bond dimension D is small and *N-
independent*:

| Operator | D |
|---|---|
| Transverse-field Ising | 3 |
| XXZ chain | 5 |
| Long-range power-law (compressed) | O(log N) |

For QuenchDyn's XXZ MPO, the five bond states are: `idle`, `S⁺ placed`,
`S⁻ placed`, `Sᶻ placed`, `coupling done`. Each MPO transition emits one
operator and the FSM walks left to right through the chain.

Once H is in MPO form, the energy ⟨ψ|H|ψ⟩ is a clean tensor sandwich
with cost O(N · χ² · D · d²). The same machinery gives you ⟨ψ|H²|ψ⟩
(used by `energy_variance`) at cost O(N · χ³ · D² · d²).

---

## 3. DMRG: ground state by local optimization

The density-matrix renormalization group (DMRG) finds the ground state
of H within the family of MPS at fixed χ. It is **variational**:

  E_DMRG(χ) = min_{|ψ⟩ ∈ MPS_χ}  ⟨ψ|H|ψ⟩  ≥  E_exact

The minimum bonds the true ground-state energy from above. The bound
tightens monotonically as χ grows. This is the cleanest convergence
property of any tensor-network algorithm.

### How a sweep works

Two-site DMRG iterates over neighbouring pairs (site i, site i+1) and
locally minimizes the energy:

```
   site:   1   2   3   4   ...   N
           ┌───┬───┬───┬───┐ ... ┌───┐
           │ A │ A │ A │ A │     │ A │
           └─┬─┴─┬─┴─┬─┴─┬─┘     └─┬─┘
             s   s   s   s         s

   Step (right sweep, at sites i=2,3):
     1. Contract A[2] · A[3] → θ                       (rank-4)
     2. Build H_eff_2site = ⟨env_left| W[2] W[3] |env_right⟩
     3. Lanczos:  find the lowest eigenvector of H_eff θ → θ_new
     4. SVD θ_new = U · diag(S) · V*,  truncate to ≤ χ
     5. A[2] ← U,   A[3] ← diag(S) · V*
     6. Update env_left to include site 2; orthogonality center moves to 3
```

The **environment tensors** `env_left` and `env_right` summarize the
fixed parts of the chain to the left and right of the active pair. They
are updated as the sweep moves through the chain, so building H_eff
locally costs only O(χ² · D · d² · χ_active) — much cheaper than
contracting the full 32-site MPO each time.

The **orthogonality center** is the one site of the MPS that is not
left- or right-orthogonal. Keeping the center on the active pair makes
the SVD truncation optimal for the global state (this is a theorem;
just trust it).

### What DMRG promises

- Per-sweep energy decreases monotonically (modulo round-off).
- E_DMRG(χ) → E_exact as χ → ∞.
- E_DMRG(χ) ≥ E_exact for any χ.

### What DMRG does *not* promise

- Convergence to the *global* minimum. DMRG can get stuck in local
  minima — most often when the true ground state has structure
  orthogonal to your initial guess. Symptoms: energy plateaus at a
  value too high to be the true ground state, or `var(H)` stays large
  even after many sweeps.

  Escape hatches: random initial state, multiple restarts, larger χ
  during early sweeps (a chi-warmup), explicit symmetry sectors
  (we don't currently use these).

- Anything about excited states. DMRG only finds the lowest-energy
  state in whatever sector it lands in.

### How to know it converged

Two complementary signals:

1. **Per-sweep ΔE plateaus** below your tolerance. Necessary but not
   sufficient — DMRG sometimes stalls in a local minimum where ΔE is
   small but the state is wrong.
2. **Energy variance** `var(H) = ⟨H²⟩ − ⟨H⟩²` is small. This is
   exactly zero at any eigenstate of H, so non-zero `var(H)` is a
   precise measure of *how far* you are from being an eigenstate. This
   is the gold standard. We compute it in `energy_variance(state)`.

A converged DMRG ground state for a gapped system at modest χ should
give `var(H) / N ~ 1e-10` or smaller, depending on the cutoff and χ.

---

## 4. 2-site TDVP: time evolution by tangent-space projection

The time-dependent variational principle (TDVP) solves the
Schrödinger equation while keeping the state on the manifold of
fixed-χ MPS. The trick is to project the right-hand side:

  i d|ψ⟩/dt  =  P_T(ψ) H |ψ⟩

where `P_T(ψ)` is the orthogonal projector onto the tangent space of
the χ-MPS manifold at the point |ψ⟩. The "tangent space" is the
linearized space of small variations of the MPS tensors that keep
the state on the manifold — concretely, infinitesimal perturbations
to each `A[i]` that preserve the bond dimensions.

### Why project?

Without projection, `−i H |ψ⟩` typically points *off* the manifold
(it adds entanglement that χ can't represent). Naive integration would
fall off. Projection brings the gradient back onto the manifold so the
ODE has a well-defined flow there.

The error of the projection is bounded by the component of `H|ψ⟩`
orthogonal to the tangent space. If the state is well-represented at
χ, this is small. If the state is straining against its bond budget
(e.g., halfway through a quench, where entanglement is growing fast),
the projection error becomes the dominant systematic.

### One-site versus two-site TDVP

There are two flavors:

- **1-site TDVP** updates one tensor `A[i]` per local step. The
  resulting evolution is **exactly unitary** within the χ-manifold —
  norm and energy are conserved to floating-point. But the bond
  dimension is *frozen* at its initial value. For a quench problem
  where entanglement grows in time, 1-site TDVP cannot follow the
  state and the projection error blows up. It is essentially useless
  for quench dynamics.

- **2-site TDVP** updates a pair `(A[i], A[i+1])` per local step. The
  pair lives in a larger space than each tensor individually, so the
  algorithm can grow the bond dimension on the fly via SVD up to
  χ_max. This is the standard for quench problems.

QuenchDyn implements the 2-site variant.

### One full step of 2-site TDVP (right sweep)

For each bond (site i, site i+1) from left to right:

1. Form the 2-site tensor θ = A[i] · A[i+1]
2. Evolve θ under the **2-site effective Hamiltonian** for time dt/2:
       θ ← exp(−i H_eff_2site · dt/2) θ
   (Krylov approximation; H_eff_2site is Hermitian, so this step is
   exactly unitary in floating point.)
3. SVD θ = U · S · V*, **truncate** to χ_max singular values.
4. A[i] ← U, A[i+1] ← S · V*.
5. If not at the right edge: evolve A[i+1] under the **1-site
   effective Hamiltonian** for time *minus* dt/2:
       A[i+1] ← exp(+i H_eff_1site · dt/2) A[i+1]
   This backward step is the dt-splitting symmetry: it cancels the
   linear-in-dt contribution that the forward step over-counted.

A full TDVP "step" in our scripts is one right sweep + one left
sweep (the mirror image). The combined left+right structure makes
the Trotter error O(dt²) per step, O(dt) accumulated.

### Three sources of error in 2-site TDVP

These are *distinct* and require *distinct* fixes. Internalize this:

| Source | Scales as | Fix |
|---|---|---|
| Trotter splitting (forward θ + backward 1-site) | O(dt²) per step | Decrease `dt` |
| Tangent-space projection (P_T H ≠ H) | depends on χ | Increase `chi_max_tdvp` |
| SVD truncation | sum of discarded sing. weights | Increase `chi_max_tdvp` *or* tighten `cutoff_tdvp` |

The first two cause the state direction to drift away from the exact
evolution. **The third is the only one that breaks unitarity.**

### Why 2-site TDVP is not unitary under truncation

The forward and backward Krylov-exponential steps are exactly unitary
(`exp(±iAt)` for Hermitian A). The Krylov approximation error is in
direction, not magnitude — the output has the same norm as the input
in floating point.

The SVD step is what changes the norm. Decomposing a matrix as
U · S · V* and discarding the smallest k singular values gives a
matrix whose Frobenius norm is `sqrt(sum(σ_kept²))`, smaller than
the original `sqrt(sum(σ_all²))` by exactly the discarded weight.
There is no way to repair this without throwing away the truncation.

So the question is not *whether* norm and energy drift — they will,
once truncation kicks in — but *when truncation kicks in* and *how
big* the drift is when it does. Those are the central questions
diagnostics answer.

---

## 5. What to expect during a quench

Here is the picture, all together:

1. You prepare the ground state of H_initial via DMRG. At small χ this
   is fast and converges nicely.
2. You quench: replace H_initial by H_final (with a different Δ in
   our case). The ground-state MPS is no longer an eigenstate of
   H_final, and TDVP starts evolving it.
3. Entanglement grows roughly linearly in time post-quench
   (Calabrese-Cardy 2005). The bond dimension required to represent
   the state therefore grows *exponentially* in time.
4. Eventually `max_chi(t)` reaches `chi_max_tdvp`. From that moment,
   every SVD step truncates, and `1 − ‖ψ(t)‖²` starts to climb.
5. The drift growth is itself exponential: at each step the discarded
   weight grows as the entanglement budget falls further behind reality.
6. Past some `t_reliable(χ)`, the trajectory is no longer trustworthy.
   You can either accept the drift (with explicit error bars from a
   χ-scan) or push χ higher.

Single-site TDVP avoids the unitarity problem by giving up on growing
entanglement. For quench problems, that's the wrong tradeoff. Two-site
TDVP with explicit drift diagnostics is the standard tool.

What you actually want to know — what is `t_reliable(χ)` for *your*
problem, and is it bigger than your target `t_max`? — is exactly what
the diagnostic CSV from a single run, plus a small χ-scan, will tell
you. See `diagnostics_guide.md` next.

---

## Further reading

- **Schollwöck (2011)** "The density-matrix renormalization group in the
  age of matrix product states", *Annals of Physics* 326. Comprehensive
  review of MPS, MPO, DMRG. Start here for any deeper question.
- **Haegeman et al. (2011)** *Phys. Rev. Lett.* **107**, 070601 — original
  TDVP paper.
- **Haegeman et al. (2016)** *Phys. Rev. B* **94**, 165116 — 2-site
  TDVP, which is what we implement.
- **Paeckel et al. (2019)** *Annals of Physics* **411**, 167998 —
  comparative review of all major time-evolution methods. The
  practical recipes for 2-site TDVP came largely from this paper.
- **Calabrese & Cardy (2005)** *J. Stat. Mech.* P04010 — the linear
  growth of entanglement after a global quench.
