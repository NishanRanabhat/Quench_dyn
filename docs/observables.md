# Observables and Measurements

This document covers all measurement functions available in QuenchDyn, for both MPS-based and ED-based calculations.

## MPS measurements (`TensorOps/measurements.jl`)

These functions work directly on MPS objects. They compute expectation values by contracting the MPS network without converting to a full state vector, so they scale with bond dimension, not system size.

### `measure_energy(state::MPSState) -> Float64`

Compute `<psi|H|psi>` by contracting the full MPS-MPO-MPS sandwich from left to right.

```julia
E = measure_energy(state)
```

The MPS should be in canonical form (this is guaranteed if `state` was created via `MPSState(mps, mpo; center=...)` or has been through DMRG/TDVP sweeps).

### `measure_norm(mps::MPS) -> Float64`

Compute `<psi|psi>` by contracting the MPS with itself (transfer matrix method, no MPO).

```julia
nrm = measure_norm(state.mps)
```

Should return 1.0 for a properly normalized state. Deviations indicate truncation effects.

Note: takes an `MPS` object, not an `MPSState`.

### `measure_local_observable(mps::MPS, op::Matrix, site::Int) -> Complex`

Compute `<psi|O_site|psi>` for a single-site operator at a given site.

```julia
ops = spin_ops(d)
Sz = Matrix{ComplexF64}(ops[:Z])    # must be Matrix, not Diagonal

sz_value = real(measure_local_observable(state.mps, Sz, 3))
```

**Important**: The operator must be a `Matrix`, not a `Diagonal`. Use `Matrix{ComplexF64}(ops[:Z])` to convert.

**How it works**: Contracts transfer matrices from left, inserts the operator at the target site, then continues contracting to the right. Cost is O(N * chi^2 * d) per site.

### `measure_correlation(mps::MPS, op_L::Matrix, site_L::Int, op_R::Matrix, site_R::Int) -> Complex`

Compute `<psi|O_L(site_L) * O_R(site_R)|psi>` for two operators at different sites.

```julia
c = real(measure_correlation(state.mps, Sz, 1, Sz, 5))
```

Requires `site_L < site_R`. For same-site "correlations", use `measure_local_observable` with the product operator.

**How it works**: Contracts from left, inserts `op_L` at `site_L`, propagates the overlap through intermediate sites, inserts `op_R` at `site_R`, then contracts to the right.

### Computing common observables

**Sz profile** (magnetization at each site):
```julia
sz_profile = [real(measure_local_observable(state.mps, Sz, i)) for i in 1:N]
```

**Staggered magnetization** (Neel order parameter):
```julia
m_stag = sum((-1.0)^i * sz_profile[i] for i in 1:N) / N
```

**Sz-Sz correlations from site 1**:
```julia
for j in 2:N
    c = real(measure_correlation(state.mps, Sz, 1, Sz, j))
    println("C(1,$j) = $c")
end
# For j=1 (auto-correlation):
c11 = real(measure_local_observable(state.mps, Sz * Sz, 1))
```

**Bond entanglement entropy**:
```julia
mps_copy = MPS{eltype(state.mps.tensors[1])}(copy.(state.mps.tensors))
for bond in 1:N-1
    make_canonical(mps_copy, bond)
    A = mps_copy.tensors[bond]
    chi_l, d_loc, chi_r = size(A)
    F = svd(reshape(A, chi_l * d_loc, chi_r))
    S_ent = entropy(F.S)
    println("Bond $bond-$(bond+1): S = $S_ent")
end
```

Note: We copy the MPS because `make_canonical` modifies it in-place. The entropy is computed from the singular values of the reshaped center tensor.

## Full-space conversion utilities

These are in `TensorOps/measurements.jl` and are only practical for small systems.

### `mps_to_vector(mps::MPS; d=2) -> Vector{ComplexF64}`

Contract the MPS into a full `d^N` state vector.

```julia
psi = mps_to_vector(state.mps)
```

The returned vector uses the convention that site 1 is the most significant index, matching the ED basis ordering. Internally, the left-to-right contraction produces site 1 as least significant (Julia column-major), so a `permutedims` is applied to correct the ordering.

**Use case**: Computing fidelity with an ED state:
```julia
psi_mps = mps_to_vector(state.mps)
fidelity = abs(dot(psi_ed, psi_mps))^2
```

### `mpo_to_matrix(mpo::MPO; d=2) -> Matrix{ComplexF64}`

Contract the MPO into a full `d^N x d^N` operator matrix.

```julia
H_dense = mpo_to_matrix(mpo)
```

Uses the same basis convention (site 1 = most significant). Useful for verifying MPO construction against ED Hamiltonians.

## ED measurements (`ED/observables.jl`)

These work on plain state vectors in the full `d^N` Hilbert space. They use Kronecker embedding internally, so they scale as O(d^N) per measurement.

### `ed_expectation(psi, O) -> Float64`

Generic expectation value `<psi|O|psi>` for any operator matrix `O` in the full Hilbert space.

```julia
val = ed_expectation(psi, H)    # energy
```

### `ed_local_expectation(psi, op, site, N; d=2) -> Float64`

Single-site expectation `<psi|O_site|psi>`. Embeds the operator via Kronecker products internally.

```julia
sz_3 = ed_local_expectation(psi, Sz, 3, N)
```

### `ed_correlation(psi, op1, site1, op2, site2, N; d=2) -> Float64`

Two-site correlation `<psi|O1_i O2_j|psi>`.

```julia
c = ed_correlation(psi, Sz, 1, Sz, 4, N)
```

### `ed_local_profile(psi, op, N; d=2) -> Vector{Float64}`

Measure a local operator at every site. Returns a length-N vector.

```julia
sz_profile = ed_local_profile(psi, Sz, N)
```

### `ed_correlation_matrix(psi, op1, op2, N; d=2) -> Matrix{Float64}`

Full N x N correlation matrix `C[i,j] = <psi|O1_i O2_j|psi>`.

```julia
C = ed_correlation_matrix(psi, Sz, Sz, N)
```

Diagonal elements use the product operator: `C[i,i] = <psi|(O1*O2)_i|psi>`.

### `ed_time_evolve(eig, psi0, t) -> Vector{ComplexF64}`

Exact time evolution `|psi(t)> = exp(-i H t) |psi0>` using a pre-computed eigensystem.

```julia
eig = diagonalize(H)
psi_t = ed_time_evolve(eig, psi0, 1.5)
```

Efficient for multiple time points: diagonalization is done once, then each time point is just phase rotation in the eigenbasis.

### `ed_evolve_and_measure(eig, psi0, times, measure_fn) -> Vector`

Evolve and apply a measurement function at each time:

```julia
times = collect(0.0:0.1:5.0)

# measure Sz profile at each time
profiles = ed_evolve_and_measure(eig, psi0, times) do psi
    ed_local_profile(psi, Sz, N)
end

# measure staggered magnetization at each time
m_stag = ed_evolve_and_measure(eig, psi0, times) do psi
    profile = ed_local_profile(psi, Sz, N)
    sum((-1.0)^i * profile[i] for i in 1:N) / N
end
```

## Comparing MPS and ED measurements

For benchmarking, it's useful to compare the same observable computed both ways.

**Local observable comparison**:
```julia
# ED
sz_ed = ed_local_profile(psi_ed, Sz_real, N)

# MPS
Sz_complex = Matrix{ComplexF64}(ops[:Z])
sz_mps = [real(measure_local_observable(state.mps, Sz_complex, i)) for i in 1:N]

# compare
max_diff = maximum(abs.(sz_ed .- sz_mps))
```

Note: ED functions typically use real-valued operators (`Float64`), while MPS functions require `ComplexF64` matrices. Make sure to use the appropriate type for each.

**State overlap**:
```julia
psi_mps = mps_to_vector(state.mps)
fidelity = abs(dot(psi_ed, psi_mps))^2
infidelity = 1.0 - fidelity
```

## Utility functions (`TensorOps/decomposition.jl`)

### `entropy(S::Vector) -> Float64`

Von Neumann entanglement entropy from singular values:

```
S_ent = -sum_i |s_i|^2 log(|s_i|^2)
```

where the singular values are normalized: `s_i -> s_i / ||S||`. Zero singular values are excluded to avoid `log(0)`.

```julia
F = svd(matrix)
S_ent = entropy(F.S)
```

### `truncation_error(S::Vector, chi::Int) -> Float64`

Weight of the discarded singular values:

```
err = sum_{i > chi} |s_i / ||S|||^2
```

```julia
err = truncation_error(F.S, 64)    # error from truncating to chi=64
```

## `spin_ops(d)` reference

Returns a Dict of spin operators for total spin S = (d-1)/2:

| Key | Operator | Spin-1/2 matrix |
|-----|----------|----------------|
| `:X` | Sx | `[0 1/2; 1/2 0]` |
| `:Y` | Sy | `[0 -i/2; i/2 0]` |
| `:Z` | Sz | `[1/2 0; 0 -1/2]` |
| `:Sp` | S+ | `[0 1; 0 0]` |
| `:Sm` | S- | `[0 0; 1 0]` |
| `:I` | Identity | `[1 0; 0 1]` |

Convention: index 1 = highest m-value (spin up), index 2 = lowest (spin down). This is consistent with `Sz = diag(+1/2, -1/2)`.

Note: `Sz` is returned as `Diagonal`. For `measure_local_observable` and `measure_correlation`, convert to `Matrix{ComplexF64}` first.
