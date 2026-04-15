# MPO Construction

This document explains how Matrix Product Operators are built in QuenchDyn.

There are two MPO builders:

1. **`build_xxz_mpo`** (`Builders/xxz_builder.jl`) — Purpose-built for the XXZ model with site-dependent fields. This is the primary builder for this project.

2. **`build_mpo`** (`Builders/mpobuilder.jl`) — Generic FSM-based builder for uniform Hamiltonians. Useful for translation-invariant models but cannot handle site-dependent fields. See the [known issue](#known-issue-fsm-builder-operator-ordering) section.

## The XXZ MPO

### Hamiltonian

```
H = J Sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1})
  + Delta Sum_i Sz_i Sz_{i+1}
  + Sum_i h_i Sz_i
```

Using the identity `Sx Sx + Sy Sy = (1/2)(S+ S- + S- S+)`, the coupling terms become:

```
(J/2) S+_i S-_{i+1}  +  (J/2) S-_i S+_{i+1}  +  Delta Sz_i Sz_{i+1}
```

### W-matrix structure

The MPO uses bond dimension chi = 5 with an **upper-triangular** convention. Each bulk tensor `W[left_bond, right_bond, bra, ket]` has the structure:

```
     col 1    col 2       col 3       col 4      col 5
     -----    -----       -----       -----      -----
row 1:  I      (J/2)*S+   (J/2)*S-   Delta*Sz   h_i*Sz
row 2:  0        0          0          0          S-
row 3:  0        0          0          0          S+
row 4:  0        0          0          0          Sz
row 5:  0        0          0          0          I
```

The 5 bond states represent the FSM (Finite State Machine) that walks left-to-right:

| Bond index | Meaning |
|------------|---------|
| 1 | Left idle: no coupling started yet (propagating identity) |
| 2 | S+ placed: left site emitted S+, waiting for S- on right |
| 3 | S- placed: left site emitted S-, waiting for S+ on right |
| 4 | Sz placed: left site emitted Sz, waiting for Sz on right |
| 5 | Right idle: coupling completed (propagating identity) |

### How to read the W-matrix

**First row** (row 1): What the left site can *emit*.
- `W[1,1] = I`: keep idle (no coupling yet)
- `W[1,2] = (J/2)*S+`: start a `S+_i S-_{i+1}` coupling by placing S+ on this site
- `W[1,3] = (J/2)*S-`: start a `S-_i S+_{i+1}` coupling by placing S- on this site
- `W[1,4] = Delta*Sz`: start a `Sz_i Sz_{i+1}` coupling by placing Sz on this site
- `W[1,5] = h_i*Sz`: on-site field (goes directly from left-idle to right-idle)

**Last column** (col 5): What the right site *absorbs*.
- `W[2,5] = S-`: absorb the S+ from the left by placing S- here
- `W[3,5] = S+`: absorb the S- from the left by placing S+ here
- `W[4,5] = Sz`: absorb the Sz from the left by placing Sz here
- `W[5,5] = I`: keep idle (coupling already completed)

### Why upper-triangular?

The upper-triangular convention means the FSM flows naturally from left to right:

```
State 1 (not coupled) --emit left op--> States 2,3,4 (waiting) --absorb right op--> State 5 (done)
```

This matches the physical picture: operator A goes on the left site, operator B goes on the right site, and the W-matrix is read left-to-right just like the FSM transitions.

For a coupling `A_i * B_{i+1}`:
- A (with coupling constant) goes in the **first row** (left site emission)
- B goes in the **last column** (right site absorption)

See `docs/mpo_convention_fix.md` for a detailed comparison of upper-triangular vs lower-triangular conventions and why upper-triangular avoids operator ordering pitfalls.

### Boundary conditions

The bulk W is `(chi, chi, d, d)`. Boundaries are sliced to enforce the FSM start/end:

**Left boundary** (site 1): Take the first row of the bulk W
```julia
tensors[1] = reshape(W_1[1, :, :, :], (1, chi, d, d))
```
This starts the FSM in state 1 (left-idle).

**Right boundary** (site N): Take the last column of the bulk W
```julia
tensors[N] = reshape(W_N[:, chi, :, :], (chi, 1, d, d))
```
This terminates the FSM in state 5 (right-idle).

### Site-dependent fields

Unlike the generic `build_mpo` which replicates a single bulk tensor, `build_xxz_mpo` creates a separate tensor for each site. The only site-dependent element is `W[1,5,:,:] = h_i * Sz`, but the per-site construction allows straightforward extension to other site-dependent parameters.

### Usage

```julia
# site-dependent field
h = [0.1, -0.2, 0.15, -0.05, 0.3, -0.1]
mpo = build_xxz_mpo(6, 1.0, 2.0, h)

# uniform field
mpo = build_xxz_mpo(20, 1.0, 2.0, 0.5)

# zero field
mpo = build_xxz_mpo(20, 1.0, 2.0)
```

### Verifying correctness

For small systems, convert the MPO to a dense matrix and compare with ED:

```julia
H_mpo = mpo_to_matrix(mpo)        # from TensorOps/measurements.jl
H_ed  = build_xxz_hamiltonian(N, J, Delta, h)  # from ED/hamiltonian.jl
maximum(abs.(Matrix(H_ed) .- real.(H_mpo)))     # should be < 1e-14
```

## The generic FSM builder

`build_mpo(fsm; N, d, T)` constructs an MPO from an FSM specification. The FSM is built by composing channel objects:

```julia
channels = [
    FiniteRangeCoupling(:Sp, :Sm, 1, J/2),   # S+_i S-_{i+1}
    FiniteRangeCoupling(:Sm, :Sp, 1, J/2),   # S-_i S+_{i+1}
    FiniteRangeCoupling(:Z, :Z, 1, Delta),    # Sz_i Sz_{i+1}
    Field(:Z, h),                              # h * Sz_i
]
fsm = build_FSM(channels)
mpo = build_mpo(fsm; N=20)
```

Available channel types:
- `FiniteRangeCoupling(op1, op2, dx, weight)` — coupling at distance dx
- `ExpChannelCoupling(op1, op2, amplitude, decay)` — exponentially decaying coupling
- `PowerLawCoupling(op1, op2, J, alpha, bondH, N)` — power-law coupling (approximated by sum of exponentials)
- `Field(op, weight)` — uniform on-site field

**Limitations:**
- Creates a single bulk tensor replicated for all sites (no site-dependent parameters)
- Uses lower-triangular convention (see known issue below)

## Known issue: FSM builder operator ordering

The generic `build_mpo` uses a lower-triangular convention where `L = bulk[chi,:,:,:]` (last row) and `R = bulk[:,1,:,:]` (first column). Combined with the FSM path construction, a `FiniteRangeCoupling(:Y, :Z, 1, J)` produces the Hamiltonian `J * Z_i * Y_{i+1}` instead of the intended `J * Y_i * Z_{i+1}`.

This bug is **invisible** for symmetric operators (Heisenberg, XXZ, where both ops are the same or S+/S- always appear in conjugate pairs), but produces **wrong results** for asymmetric couplings (e.g., Dzyaloshinskii-Moriya interaction).

For the full analysis and proposed fix, see `docs/mpo_convention_fix.md`.

**For the XXZ model, always use `build_xxz_mpo` which is correct by construction.**
