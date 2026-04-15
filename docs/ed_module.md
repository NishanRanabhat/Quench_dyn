# Exact Diagonalization Module

The ED module provides brute-force exact methods for benchmarking MPS results. It works in the full `d^N`-dimensional Hilbert space, so it is limited to small systems (N ~ 12-14 for spin-1/2).

The module is split into four files, each independent and composable:

```
ED/
  operators.jl     # Kronecker embedding primitives (model-agnostic)
  hamiltonian.jl   # XXZ Hamiltonian construction (uses operators.jl)
  eigensolver.jl   # Diagonalization (uses LinearAlgebra)
  observables.jl   # Expectations, time evolution (uses operators.jl)
```

## Basis convention

The computational basis uses the convention:

- **Site 1 is the most significant index** (leftmost in tensor product)
- For spin-1/2: index 1 = up (+1/2), index 2 = down (-1/2)
- Basis state index: `idx = 1 + (s1-1)*d^(N-1) + (s2-1)*d^(N-2) + ... + (sN-1)`

This matches the MPS convention (after the `mps_to_vector` permutation fix) and `spin_ops()` where `Sz = diag(+1/2, -1/2)`.

Example for N=3, d=2:
```
Index 1: |up up up>       = |1,1,1>
Index 2: |up up down>     = |1,1,2>
Index 3: |up down up>     = |1,2,1>
Index 4: |up down down>   = |1,2,2>
Index 5: |down up up>     = |2,1,1>
Index 6: |down up down>   = |2,1,2>
Index 7: |down down up>   = |2,2,1>
Index 8: |down down down> = |2,2,2>
```

## Operators (`operators.jl`)

Model-agnostic building blocks for embedding local operators into the full Hilbert space via Kronecker products.

### `embed_operator(op, site, N; d=2)`

Embed a single-site operator:

```julia
Sz = Float64.([0.5 0; 0 -0.5])
Sz_full = embed_operator(Sz, 3, 6)   # Sz on site 3 of a 6-site chain
# Result: I (x) I (x) Sz (x) I (x) I (x) I   (64 x 64 matrix)
```

### `embed_two_site(op1, op2, site1, site2, N; d=2)`

Embed a two-site operator product:

```julia
Sp = Float64.([0 1; 0 0])
Sm = Float64.([0 0; 1 0])
SpSm = embed_two_site(Sp, Sm, 2, 4, 6)   # S+_2 * S-_4
```

Sites need not be adjacent. The result is the full Kronecker product with identities on all other sites.

### `embed_multi_site(ops, sites, N; d=2)`

Generalization to arbitrary multi-site operators:

```julia
ops = [Sz, Sz, Sz]
sites = [1, 3, 5]
O = embed_multi_site(ops, sites, 6)   # Sz_1 * Sz_3 * Sz_5
```

### `basis_state(config, N; d=2)`

Construct a computational basis state vector:

```julia
psi = basis_state([1, 2, 1, 2], 4)   # |up down up down> for N=4
```

`config` is a length-N vector of local state indices (1-based).

### `neel_state(N; start_up=true, d=2)`

Convenience for the Neel state:

```julia
psi_neel = neel_state(6)               # |up down up down up down>
psi_neel2 = neel_state(6; start_up=false)  # |down up down up down up>
```

## Hamiltonian (`hamiltonian.jl`)

### `build_xxz_hamiltonian(N, J, Delta, h; d=2)`

Builds the full `d^N x d^N` Hamiltonian matrix:

```julia
h = [0.1, -0.2, 0.15, -0.05, 0.3, -0.1]
H = build_xxz_hamiltonian(6, 1.0, 2.0, h)
# Returns Hermitian{Float64} matrix (64 x 64)
```

Internally uses `(J/2)(S+ S- + S- S+)` for the XY coupling, which keeps the matrix real-valued. The Hermitian wrapper ensures eigenvalues are real.

Convenience methods:

```julia
H = build_xxz_hamiltonian(6, 1.0, 2.0, 0.5)       # uniform field h=0.5
H = build_xxz_hamiltonian(6, 1.0, 2.0)             # zero field
```

**Scaling**: The matrix has `d^N x d^N` elements. For spin-1/2:
| N | Matrix size | Memory |
|---|------------|--------|
| 8 | 256 x 256 | 0.5 MB |
| 10 | 1024 x 1024 | 8 MB |
| 12 | 4096 x 4096 | 128 MB |
| 14 | 16384 x 16384 | 2 GB |
| 16 | 65536 x 65536 | 32 GB |

Practical limit is around N = 14-16 depending on available memory.

## Eigensolver (`eigensolver.jl`)

### `diagonalize(H)`

Full diagonalization via Julia's `eigen()`. Returns an `EDEigensystem`:

```julia
eig = diagonalize(H)
# eig.values  : Vector{Float64} of all eigenvalues (ascending)
# eig.vectors : Matrix{ComplexF64} where eig.vectors[:, k] is the k-th eigenstate
```

### `ground_state(eig)` / `ground_state(H)`

Extract the ground state:

```julia
E_gs, psi_gs = ground_state(eig)
# or directly from H:
E_gs, psi_gs = ground_state(H)
```

### `low_energy_states(eig, n)`

Get the lowest `n` states:

```julia
energies, states = low_energy_states(eig, 5)
# energies : Vector{Float64} of length 5
# states   : Matrix where states[:, k] is the k-th eigenstate
```

## Observables (`observables.jl`)

All functions work with plain state vectors and operator matrices. They are decoupled from the eigensolver and Hamiltonian builder.

### Expectation values

```julia
# generic: <psi|O|psi>
val = ed_expectation(psi, O_matrix)

# single-site: <psi|O_site|psi>
sz_3 = ed_local_expectation(psi, Sz, 3, N)

# two-site correlation: <psi|O1_i O2_j|psi>
c = ed_correlation(psi, Sz, 1, Sz, 4, N)
```

### Profiles and correlation matrices

```julia
# <O_i> for all sites
sz_profile = ed_local_profile(psi, Sz, N)    # Vector{Float64} of length N

# <O1_i O2_j> for all pairs
C = ed_correlation_matrix(psi, Sz, Sz, N)    # N x N Float64 matrix
# Diagonal: C[i,i] = <(Sz)^2_i>
# Off-diagonal: C[i,j] = <Sz_i Sz_j>
```

### Time evolution

Exact time evolution via the eigenbasis:

```julia
# single time point
psi_t = ed_time_evolve(eig, psi0, t)

# or from Hamiltonian directly (diagonalizes each time -- slow)
psi_t = ed_time_evolve(H, psi0, t)
```

The eigenbasis method is efficient for multiple time points since diagonalization is done once.

### Evolve and measure

Apply a measurement function at each time step:

```julia
times = collect(0.0:0.1:10.0)
sz_profiles = ed_evolve_and_measure(eig, psi0, times) do psi
    ed_local_profile(psi, Sz, N)
end
# sz_profiles[k] = <Sz_i>(t_k) for all sites i
```

The `do` block syntax makes it easy to measure arbitrary observables. The function receives the evolved state and returns whatever you want to track.

## Typical benchmarking workflow

### Compare DMRG ground state with ED

```julia
# ED
H = build_xxz_hamiltonian(N, J, Delta, h)
E_ed, psi_ed = ground_state(H)

# DMRG
mpo = build_xxz_mpo(N, J, Delta, h)
[run DMRG]
E_dmrg = dmrg_sweep(state, solver, opts, dir)

# compare
println("Energy difference: ", abs(E_ed - E_dmrg))   # should be < 1e-8
```

### Compare TDVP time evolution with ED

```julia
# ED evolution
eig = diagonalize(H)
psi_ed_t = ed_time_evolve(eig, psi0, t)
sz_ed = ed_local_profile(psi_ed_t, Sz, N)

# TDVP evolution
[run TDVP sweeps]
sz_tdvp = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]

# compare local observables
max_err = maximum(abs.(sz_ed .- sz_tdvp))

# compare full state overlap (fidelity)
psi_tdvp = mps_to_vector(state.mps)
fidelity = abs(dot(psi_ed_t, psi_tdvp))^2    # should be close to 1
```

Both comparisons are implemented in the test scripts `tests/validate_mpo_and_gs.jl` and `tests/validate_tdvp.jl`.
