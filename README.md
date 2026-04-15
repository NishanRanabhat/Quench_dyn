# QuenchDyn

Tensor network simulation of quench dynamics in the 1D XXZ spin chain with site-dependent longitudinal field.

## Physics

The target Hamiltonian is the XXZ model with open boundary conditions:

```
H = J Sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1})
  + Delta Sum_i Sz_i Sz_{i+1}
  + Sum_i h_i Sz_i
```

The quench protocol starts from the ground state at high `Delta/J` (Ising-ordered phase) and suddenly switches to low `Delta/J` (XY-dominated phase). This is a **reverse quench**: instead of watching order melt, we observe the scale of order building after the quench.

## Methods

- **DMRG** (Density Matrix Renormalization Group): Two-site algorithm for finding ground states as Matrix Product States
- **TDVP** (Time-Dependent Variational Principle): Two-site algorithm for real-time evolution within the MPS manifold
- **Exact Diagonalization**: Full Hilbert space methods for benchmarking (small systems, N ~ 12-14)

## Project structure

```
src/
  QuenchDyn.jl           # module entry point
  Core/                  # MPS/MPO types, spin operators, state bundles
  TensorOps/             # SVD, canonicalization, environments, measurements
  Algorithms/            # DMRG sweeps, TDVP sweeps, Lanczos/Krylov solvers
  Builders/              # MPO and MPS construction (XXZ + generic FSM)
  ED/                    # exact diagonalization (operators, Hamiltonian, eigensolver, observables)
scripts/
  run_dmrg.jl            # ground state search with all parameters documented
  run_tdvp.jl            # full quench protocol (DMRG ground state + TDVP evolution)
tests/
  validate_mpo_and_gs.jl # MPO vs ED Hamiltonian + DMRG vs ED ground state
  validate_tdvp.jl       # TDVP vs ED time evolution (local observables + fidelity)
docs/
  quickstart.md          # get running in 5 minutes
  architecture.md        # code structure, types, data flow
  algorithms.md          # DMRG and TDVP details, parameter tuning
  mpo_construction.md    # MPO W-matrix structure and conventions
  ed_module.md           # exact diagonalization usage
  observables.md         # all measurement functions
  mpo_convention_fix.md  # documented FSM operator ordering issue
```

## Quick start

```bash
# run validation tests
julia tests/validate_mpo_and_gs.jl
julia tests/validate_tdvp.jl

# run a ground state search
julia scripts/run_dmrg.jl

# run a quench dynamics simulation
julia scripts/run_tdvp.jl
```

Edit the parameter blocks at the top of each script to configure your simulation.

## Dependencies

- Julia 1.9+
- `LinearAlgebra` (stdlib)
- `TensorOperations`

## Documentation

See the `docs/` folder for detailed documentation:

- **[Quickstart](docs/quickstart.md)** -- install, validate, run your first simulation
- **[Architecture](docs/architecture.md)** -- module structure, key types, data flow
- **[Algorithms](docs/algorithms.md)** -- DMRG/TDVP sweep logic, parameter tuning
- **[MPO Construction](docs/mpo_construction.md)** -- W-matrix structure, upper-triangular convention
- **[ED Module](docs/ed_module.md)** -- exact diagonalization for benchmarking
- **[Observables](docs/observables.md)** -- all measurement functions (MPS and ED)
