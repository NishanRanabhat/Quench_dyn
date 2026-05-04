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
  scan_tdvp_chi.jl       # χ-scan for TDVP, writes one CSV per χ + cross-χ summary
tests/
  validate_mpo_and_gs.jl # MPO vs ED Hamiltonian + DMRG vs ED ground state
  validate_tdvp.jl       # TDVP vs ED time evolution (local observables + fidelity)
  benchmark_tdvp_drift.jl# single-run drift diagnostic, writes CSV
docs/
  algorithms_primer.md   # *start here if new to MPS*: pedagogical primer
  diagnostics_guide.md   # how to read CSV diagnostics + troubleshooting trees
  parameter_reference.md # cheat sheet for picking χ, dt, cutoffs, etc.
  quickstart.md          # get running in 5 minutes
  architecture.md        # code structure, types, data flow
  algorithms.md          # DMRG/TDVP code-level reference (sweep mechanics)
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

Reading order if you are new to tensor networks:

1. **[Algorithms primer](docs/algorithms_primer.md)** — what bond dimension is, how DMRG and 2-site TDVP work, why TDVP-2 cannot conserve norm under truncation. Read this first.
2. **[Quickstart](docs/quickstart.md)** — install, validate, run your first simulation.
3. **[Diagnostics guide](docs/diagnostics_guide.md)** — what every column in `quench_log.csv` means, troubleshooting decision trees, three worked CSV examples. Read this *after* a run finishes and you don't yet know if you can trust it.
4. **[Parameter reference](docs/parameter_reference.md)** — cheat sheet for picking χ, dt, cutoffs, sweeps. Reach for this when tuning.

Reference documentation (developer / maintainer):

- **[Architecture](docs/architecture.md)** — module structure, key types, data flow
- **[Algorithms (code-level)](docs/algorithms.md)** — sweep mechanics, environment construction
- **[MPO construction](docs/mpo_construction.md)** — W-matrix structure, upper-triangular convention
- **[ED module](docs/ed_module.md)** — exact diagonalization for benchmarking
- **[Observables](docs/observables.md)** — all measurement functions (MPS and ED)
