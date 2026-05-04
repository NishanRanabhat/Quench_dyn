# Algorithms: DMRG and TDVP

This document covers the two sweep-based algorithms in QuenchDyn: DMRG for ground states and TDVP for real-time evolution.

Both are **two-site** algorithms, meaning they optimize/evolve two adjacent MPS tensors at once. This allows the bond dimension to grow adaptively during the calculation.

## Common concepts

### Sweep direction

Both algorithms sweep back and forth across the chain:

```
Right sweep:  site 1-2, 2-3, 3-4, ..., (N-1)-N
Left sweep:   site N-(N-1), (N-1)-(N-2), ..., 2-1
```

One full update = one right sweep + one left sweep. For DMRG, each sweep refines the ground state. For TDVP, each sweep advances the state by one time step `dt`.

Convention in the code: odd sweeps go right, even sweeps go left. This is handled by the caller:

```julia
dir = isodd(sweep) ? :right : :left
```

### Effective Hamiltonian

At each bond (i, i+1), the algorithm constructs a local effective Hamiltonian by sandwiching the two MPO tensors between left and right environments:

```
L_env --- W_i --- W_{i+1} --- R_env
  |        |         |          |
  |       bra       bra         |
  |        |         |          |
  |       ket       ket         |
  |        |         |          |
L_env --- W_i --- W_{i+1} --- R_env
```

The effective Hamiltonian acts on the two-site tensor `theta[chi_L, d, d, chi_R]` as a linear map (matrix-vector product). It is never formed explicitly as a matrix -- only the action `H_eff * theta` is computed via tensor contractions.

### SVD truncation

After solving or evolving at each bond, the two-site tensor is split back into two single-site tensors via SVD with truncation:

```
theta[chi_L, d, d, chi_R]  -->  reshape to (chi_L*d, d*chi_R)
                            -->  SVD: U * S * V
                            -->  truncate to chi_max singular values
                            -->  discard values below cutoff
```

The `cutoff` is applied to the **normalized** singular value spectrum: `s_i / ||S||`. This makes it a relative threshold independent of the overall state norm.

### Environment updates

As the orthogonality center moves, environments are updated incrementally. When sweeping right, each step adds one site to the left environment. When sweeping left, each step adds one site to the right environment. This avoids rebuilding the full environment from scratch at each bond.

## DMRG

**Purpose**: Find the ground state (lowest eigenstate) of the Hamiltonian.

**File**: `Algorithms/dmrg.jl`

### How it works

At each bond during a sweep:

1. **Contract** the two adjacent MPS tensors into a two-site tensor `theta`
2. **Build** the effective Hamiltonian from left environment, two MPO tensors, and right environment
3. **Solve** the eigenvalue problem `H_eff |theta> = E |theta>` using Lanczos to find the lowest eigenvalue and eigenvector
4. **Split** the optimized `theta` via truncated SVD back into two MPS tensors
5. **Update** the environment for the next step

### Lanczos eigensolver

The `LanczosSolver` finds the lowest eigenvalue of the effective Hamiltonian iteratively:

```julia
solver = LanczosSolver(krylov_dim, max_iter)
```

**Parameters:**

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| `krylov_dim` | 3-6 | Number of Lanczos vectors per restart. Small is fine for ground state (we only need the lowest eigenvalue). Larger values improve convergence per restart but cost more memory and compute. |
| `max_iter` | 50-200 | Maximum number of Lanczos restarts. Each restart refines the eigenvector estimate. The solver converges when the eigenvalue changes by less than `tol * max(1, |E|)` between restarts. |

The Lanczos algorithm:
1. Start from the current `theta` as initial guess
2. Build a Krylov subspace by repeatedly applying `H_eff`
3. Diagonalize the small `krylov_dim x krylov_dim` tridiagonal matrix
4. Extract the lowest eigenvalue/eigenvector
5. If not converged, restart with the current best eigenvector

### Parameter tuning guidelines

| Parameter | Start with | Increase if... |
|-----------|-----------|----------------|
| `chi_max` | 64-128 | Energy hasn't converged (compare successive sweeps) |
| `n_sweeps` | 20-30 | Energy is still decreasing at the end |
| `cutoff` | 1e-10 | Truncation error is significant (check via `truncation_error()`) |
| `krylov_dim` | 4 | Rarely needs increasing for ground state |
| `max_iter` | 100 | Lanczos isn't converging (you'll see noisy energies) |

**Convergence check**: The energy should stabilize across sweeps. If it oscillates or keeps decreasing after many sweeps, increase `chi_max` or `n_sweeps`.

### Usage

```julia
state = MPSState(mps, mpo; center=1)
solver = LanczosSolver(4, 100)
opts = DMRGOptions(128, 1e-10, d)
n_sweeps = 30

for sweep in 1:n_sweeps
    dir = isodd(sweep) ? :right : :left
    res = dmrg_sweep(state, solver, opts, dir)
    # res = (E, max_trunc, total_trunc, max_chi)
    E = res.E
end
```

Note: `dmrg_sweep` prints the energy at each bond optimization (via the internal `println`). The returned value is the energy from the final bond of the sweep.

## TDVP (Time-Dependent Variational Principle)

**Purpose**: Time-evolve a state under a Hamiltonian while keeping it in the MPS manifold.

**File**: `Algorithms/tdvp.jl`

### How it works

Two-site TDVP uses a second-order Trotter decomposition. At each bond during a **right sweep**:

1. **Contract** two adjacent tensors into `theta`
2. **Forward evolve**: Apply `exp(-i H_eff dt/2)` to `theta` using Krylov exponential
3. **Split** via truncated SVD: `theta -> U * S * V`. Left tensor = `U`, right tensor = `S * V`
4. **Backward evolve** (bond correction): Apply `exp(+i H_eff dt/2)` to the right tensor with a **one-site** effective Hamiltonian. This corrects the Trotter error at the bond.
5. **Update** left environment

The left sweep does the mirror operation. Together, one right + one left sweep gives one full time step of `dt`.

### Why the backward evolution?

The backward (negative time) evolution on the bond tensor is essential for:
- **Second-order accuracy**: Without it, the error is O(dt) instead of O(dt^2)
- **Norm conservation**: The forward and backward evolutions together preserve unitarity
- **Consistent gauge**: It maintains the proper canonical form as the center moves

### Krylov exponential solver

The `KrylovExponential` computes `exp(-i H dt) |v>` without forming the full matrix exponential:

```julia
solver = KrylovExponential(krylov_dim, tol, evol_type)
```

**Parameters:**

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| `krylov_dim` | 20-40 | Size of the Krylov subspace. Larger = more accurate per step. The matrix exponential is computed in this small subspace. |
| `tol` | 1e-12 | Convergence tolerance. The solver checks element-wise convergence of the evolved vector between successive Krylov dimensions. |
| `evol_type` | `"real"` | `"real"` for unitary evolution `e^{-iHt}` (physical dynamics). `"imaginary"` for `e^{-Ht}` (ground state cooling). |

The Krylov method:
1. Build a Krylov subspace `{v, Hv, H^2v, ..., H^k v}` via Lanczos
2. Orthogonalize to get a tridiagonal matrix `T_k`
3. Compute `exp(-i T_k dt)` (small k x k matrix)
4. Project back to the full space
5. Check convergence by comparing with the previous Krylov dimension

### Parameter tuning guidelines

| Parameter | Start with | Increase if... |
|-----------|-----------|----------------|
| `dt` | 0.05 | Norm drift is large or observables are noisy (reduce dt) |
| `chi_max_tdvp` | 128-256 | Entanglement grows beyond current chi (check bond entropy) |
| `cutoff_tdvp` | 1e-10 | Truncation is too aggressive |
| `krylov_dim` | 30 | Krylov exponential isn't converging (rare for dt < 0.1) |
| `krylov_tol` | 1e-12 | Usually fine at this value |

**Key tradeoffs:**
- Smaller `dt` = more accurate per step, but more steps needed = slower
- Larger `chi_max` = less truncation error, but each step costs more
- The dt and chi_max tradeoff is problem-dependent. For weakly entangled systems, larger dt with moderate chi works. For highly entangled dynamics (near critical points), small dt and large chi are needed.

### Conservation laws

For real-time evolution (`evol_type = "real"`):
- **Energy** should be exactly conserved (it's the expectation value of the Hamiltonian you're evolving under). Drift indicates insufficient chi or dt too large.
- **Norm** should stay at 1.0. Small drift is expected from SVD truncation. If norm drifts significantly, reduce cutoff or increase chi.

Always monitor both as sanity checks.

### Usage

```julia
state = MPSState(mps, mpo; center=1)
solver = KrylovExponential(30, 1e-12, "real")
opts = TDVPOptions(0.05, 256, 1e-10, d)

for step in 1:n_steps
    # one full time step = right sweep + left sweep
    tdvp_sweep(state, solver, opts, :right)
    tdvp_sweep(state, solver, opts, :left)
    # state.mps is now at time t = step * dt
end
```

## Combining DMRG and TDVP for quench dynamics

The typical quench protocol:

```julia
# 1) DMRG: find ground state at initial parameters
mpo_i = build_xxz_mpo(N, J, Delta_i, h)
state = MPSState(mps, mpo_i; center=1)
[run DMRG sweeps]

# 2) Quench: swap the MPO
mpo_f = build_xxz_mpo(N, J, Delta_f, h)
state = MPSState(state.mps, mpo_f; center=1)
# This rebuilds environments for the new Hamiltonian
# The MPS is unchanged (still the ground state of H_i)

# 3) TDVP: evolve under H_f
[run TDVP steps]
```

The key point: when you create the new `MPSState`, the MPS is re-canonicalized and new environments are built for the post-quench Hamiltonian. The MPS itself is not modified beyond canonicalization.
