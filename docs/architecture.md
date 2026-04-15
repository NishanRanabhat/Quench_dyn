# Architecture Overview

This document explains how the codebase is organized, the key data types, and how they connect.

## Module structure

Everything lives under `src/QuenchDyn.jl`, which includes files from five directories:

```
src/
  QuenchDyn.jl           # module definition, includes, exports
  Core/
    types.jl             # MPS, MPO, Environment, options structs
    site.jl              # SpinSite, SuperSite, spin_ops()
    fsm.jl               # FSM channel types for generic MPO builder
    states.jl            # MPSState bundle
  TensorOps/
    decomposition.jl     # SVD truncation, entropy, truncation_error
    canonicalization.jl   # make_canonical, orthogonality checks
    environment.jl       # environment contraction and update
    measurements.jl      # energy, norm, local observables, correlations
  Algorithms/
    solvers.jl           # LanczosSolver, KrylovExponential, effective Hamiltonians
    dmrg.jl              # two-site DMRG sweep
    tdvp.jl              # two-site TDVP sweep
  Builders/
    mpobuilder.jl        # generic FSM-based MPO builder (uniform systems)
    mpsbuilder.jl        # product_state, random_state
    xxz_builder.jl       # site-dependent XXZ MPO builder
  ED/
    operators.jl         # Kronecker embedding primitives
    hamiltonian.jl       # XXZ Hamiltonian construction
    eigensolver.jl       # full diagonalization, EDEigensystem
    observables.jl       # expectations, time evolution, profiles
```

## Design principles

1. **Modularity**: Every operation is an independent function. There are no black-box pipelines. A user composes their workflow by calling individual functions.

2. **Separation of concerns**: Hamiltonian building, state preparation, algorithms, and measurements are in separate files. Each can be used independently.

3. **No hidden state**: Functions take explicit arguments and return explicit results. The only mutable state is `MPSState`, which bundles MPS + MPO + environments for sweep algorithms.

## Key data types

### Tensor networks

```julia
struct MPS{T} <: TensorNetwork{T}
    tensors::Vector{Array{T,3}}    # tensors[i] has shape (chi_left, d, chi_right)
end

struct MPO{T} <: TensorNetwork{T}
    tensors::Vector{Array{T,4}}    # tensors[i] has shape (chi_left, chi_right, d_bra, d_ket)
end

struct Environment{T}
    tensors::Vector{Union{Array{T,3}, Nothing}}
    # tensors[i] = left env between sites i-1 and i
    # tensors[N+1] = boundary tensor (1,1,1)
end
```

### MPS tensor index convention

Each MPS tensor `A[i]` has three indices:

```
        d (physical)
        |
chi_L --A-- chi_R
```

- Index 1: left bond (chi_left)
- Index 2: physical (d = local Hilbert space dimension)
- Index 3: right bond (chi_right)

Boundary tensors have chi=1 on the open end: `A[1]` has shape `(1, d, chi)`, `A[N]` has shape `(chi, d, 1)`.

### MPO tensor index convention

Each MPO tensor `W[i]` has four indices:

```
chi_L --W-- chi_R
        |
       d_bra
        |
       d_ket
```

- Index 1: left MPO bond
- Index 2: right MPO bond
- Index 3: bra (acts on the ket of the upper/conjugate MPS)
- Index 4: ket (acts on the ket of the lower MPS)

### MPSState

The workhorse for DMRG and TDVP. Bundles the MPS, MPO, environment, and orthogonality center:

```julia
mutable struct MPSState{Tmps, Tmpo, Tenv}
    mps::MPS{Tmps}
    mpo::MPO{Tmpo}
    environment::Environment{Tenv}
    center::Int
end
```

The constructor `MPSState(mps, mpo; center=1)`:
1. Canonicalizes the MPS to the given center (in-place)
2. Builds left and right environments around the center
3. Type-promotes the environment: `Tenv = promote_type(Tmps, Tmpo)`

### Options structs

```julia
struct DMRGOptions
    chi_max::Int         # maximum bond dimension for this sweep
    cutoff::Float64      # SVD truncation cutoff
    local_dim::Int       # local Hilbert space dimension (d)
end

struct TDVPOptions
    dt::Float64          # time step
    chi_max::Int         # maximum bond dimension
    cutoff::Float64      # SVD truncation cutoff
    local_dim::Int       # local Hilbert space dimension (d)
end

struct SweepSchedule
    maxdims::Vector{Int}       # per-sweep chi values
    cutoffs::Vector{Float64}   # per-sweep cutoff values
    n_sweeps::Int
end
```

`SweepSchedule(chi_max, n_sweeps)` ramps chi linearly from `chi_min` to `chi_max` over the first half of sweeps, then holds. Cutoff tightens similarly.

### Solvers

```julia
struct LanczosSolver
    krylov_dim::Int      # Krylov subspace size (typically 3-6)
    max_iter::Int        # maximum Lanczos restarts (typically 100)
end

struct KrylovExponential
    krylov_dim::Int      # Krylov subspace size (typically 20-40)
    tol::Float64         # convergence tolerance (typically 1e-12)
    evol_type::String    # "real" for e^{-iHt}, "imaginary" for e^{-Ht}
end
```

### Spin sites

```julia
struct SpinSite{T} <: AbstractSite{T}
    dim::Int                # 2S+1
    ops::Dict               # :X, :Y, :Z operators as matrices
    spectra::Dict           # precomputed eigenvalues/eigenvectors per axis
end
```

Created via `SpinSite(S; T=ComplexF64)` where `S` is the spin quantum number (0.5 for spin-1/2).

Eigenvalue convention: sorted **descending**, so for spin-1/2:
- `(:Z, 1)` = spin-up (m = +1/2)
- `(:Z, 2)` = spin-down (m = -1/2)

This matches `spin_ops(d)` where `Sz = diag(+1/2, -1/2)`.

### ED types

```julia
struct EDEigensystem
    values::Vector{Float64}           # eigenvalues (ascending)
    vectors::Matrix{ComplexF64}       # eigenvectors as columns
end
```

## Data flow

### Ground state search (DMRG)

```
build_xxz_mpo(N, J, Delta, h)  -->  MPO
product_state(sites, labels)    -->  MPS
MPSState(mps, mpo; center=1)   -->  MPSState (canonicalized + environments built)

for sweep in 1:n_sweeps
    dmrg_sweep(state, solver, opts, direction)  -->  energy
end

measure_local_observable(state.mps, Sz, i)  -->  <Sz_i>
```

### Quench dynamics (DMRG + TDVP)

```
# 1) ground state at initial parameters
build_xxz_mpo(N, J, Delta_i, h)  -->  MPO_i
[DMRG as above]

# 2) switch to post-quench Hamiltonian
build_xxz_mpo(N, J, Delta_f, h)  -->  MPO_f
MPSState(state.mps, MPO_f; center=1)  -->  new MPSState

# 3) time evolution
for step in 1:n_steps
    tdvp_sweep(state, solver, opts, :right)
    tdvp_sweep(state, solver, opts, :left)
    [measurements]
end
```

### ED benchmarking

```
build_xxz_hamiltonian(N, J, Delta, h)  -->  Hermitian matrix
diagonalize(H)                          -->  EDEigensystem
ground_state(eig)                       -->  (energy, vector)
ed_time_evolve(eig, psi0, t)           -->  psi(t)
ed_local_profile(psi, Sz, N)           -->  [<Sz_1>, <Sz_2>, ...]
```

## Environment storage layout

The `Environment` holds `N+1` tensors:

```
env.tensors[N+1] = boundary (1,1,1)   -- used as both L and R boundary
env.tensors[1..center-1] = left environments
env.tensors[center] = nothing
env.tensors[center+1..N] = right environments
```

During sweeps, environments are updated incrementally as the orthogonality center moves.

## Type system

The code uses Julia's parametric type system for element types:
- MPS is typically `MPS{ComplexF64}` (quantum states are complex)
- MPO is typically `MPO{Float64}` (XXZ Hamiltonian is real)
- Environments are `Environment{ComplexF64}` (promoted from MPS and MPO types)

The promotion happens automatically in `MPSState`:
```julia
Tenv = promote_type(Tmps, Tmpo)
```
