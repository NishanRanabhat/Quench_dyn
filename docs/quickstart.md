# Quickstart

Get up and running in 5 minutes.

## Prerequisites

- Julia 1.9 or later
- Required packages:

```julia
using Pkg
Pkg.add("LinearAlgebra")
Pkg.add("TensorOperations")
```

## Project structure

```
Quench_dyn/
  src/
    QuenchDyn.jl         # module entry point
    Core/                # types, spin operators, states
    TensorOps/           # SVD, canonicalization, environments, measurements
    Algorithms/          # DMRG, TDVP, Lanczos/Krylov solvers
    Builders/            # MPO and MPS construction
    ED/                  # exact diagonalization (benchmarking)
  scripts/
    run_dmrg.jl          # ground state search
    run_tdvp.jl          # quench dynamics
  tests/
    validate_mpo_and_gs.jl   # MPO correctness + DMRG vs ED
    validate_tdvp.jl         # TDVP vs ED time evolution
  docs/
```

## Loading the module

From any script:

```julia
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn
```

Or from the REPL (standing in the project root):

```julia
push!(LOAD_PATH, joinpath(pwd(), "src"))
using QuenchDyn
```

## Run the validation tests

These confirm everything works on your machine.

```bash
# from the project root
julia tests/validate_mpo_and_gs.jl    # ~10 seconds
julia tests/validate_tdvp.jl          # ~30 seconds
```

Expected output: `ALL TESTS PASSED` and `TDVP VALIDATION PASSED`.

## Minimal DMRG example

Find the ground state of a 10-site XXZ chain:

```julia
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using QuenchDyn, LinearAlgebra

N = 10; J = 1.0; Delta = 2.0; d = 2
h = zeros(N)

# build MPO
mpo = build_xxz_mpo(N, J, Delta, h)

# initial MPS: Neel state |up down up down ...>
sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps = product_state(sites, labels)

# set up DMRG
state = MPSState(mps, mpo; center=1)
solver = LanczosSolver(4, 100)
schedule = SweepSchedule(64, 20; cutoff_final=1e-10)

# run sweeps
E = 0.0
for sweep in 1:schedule.n_sweeps
    opts = DMRGOptions(schedule.maxdims[sweep], schedule.cutoffs[sweep], d)
    dir = isodd(sweep) ? :right : :left
    E = dmrg_sweep(state, solver, opts, dir)
end

println("Ground state energy: $E")
```

## Minimal TDVP quench example

Quench from Delta=3 to Delta=0.5:

```julia
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using QuenchDyn, LinearAlgebra

N = 10; J = 1.0; d = 2; h = zeros(N)

# 1) ground state at Delta_i = 3.0
mpo_i = build_xxz_mpo(N, J, 3.0, h)
sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mps = product_state(sites, labels)

state = MPSState(mps, mpo_i; center=1)
solver = LanczosSolver(4, 100)
schedule = SweepSchedule(64, 20; cutoff_final=1e-10)

for sweep in 1:schedule.n_sweeps
    opts = DMRGOptions(schedule.maxdims[sweep], schedule.cutoffs[sweep], d)
    dir = isodd(sweep) ? :right : :left
    dmrg_sweep(state, solver, opts, dir)
end

# 2) quench: switch MPO to Delta_f = 0.5
mpo_f = build_xxz_mpo(N, J, 0.5, h)
state = MPSState(state.mps, mpo_f; center=1)

solver_tdvp = KrylovExponential(30, 1e-12, "real")
tdvp_opts = TDVPOptions(0.05, 128, 1e-10, d)

ops = spin_ops(d)
Sz = Matrix{ComplexF64}(ops[:Z])

# 3) evolve
for step in 1:100
    tdvp_sweep(state, solver_tdvp, tdvp_opts, :right)
    tdvp_sweep(state, solver_tdvp, tdvp_opts, :left)
end

# measure final Sz profile
for i in 1:N
    sz = real(measure_local_observable(state.mps, Sz, i))
    println("Site $i: <Sz> = $sz")
end
```

## Full run scripts

For production runs with all parameters documented:

```bash
julia scripts/run_dmrg.jl     # ground state with measurements
julia scripts/run_tdvp.jl     # full quench protocol
```

Edit the parameter blocks at the top of each script to configure your run.

## Next steps

- [Architecture overview](architecture.md) — understand the code structure
- [Algorithms](algorithms.md) — DMRG/TDVP details and parameter tuning
- [MPO construction](mpo_construction.md) — how MPOs are built
- [ED module](ed_module.md) — exact diagonalization for benchmarking
- [Observables](observables.md) — all measurement functions
