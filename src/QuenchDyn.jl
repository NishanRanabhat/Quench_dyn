module QuenchDyn

using LinearAlgebra
using TensorOperations

# ── Core types and operators ───────────────────────────────────────────────
include("Core/types.jl")
include("Core/site.jl")
include("Core/fsm.jl")
include("Core/states.jl")

# ── Tensor operations ─────────────────────────────────────────────────────
include("TensorOps/decomposition.jl")
include("TensorOps/canonicalization.jl")
include("TensorOps/environment.jl")
include("TensorOps/measurements.jl")

# ── Algorithms ────────────────────────────────────────────────────────────
include("Algorithms/solvers.jl")
include("Algorithms/dmrg.jl")
include("Algorithms/tdvp.jl")

# ── Builders ──────────────────────────────────────────────────────────────
include("Builders/mpobuilder.jl")
include("Builders/mpsbuilder.jl")
include("Builders/xxz_builder.jl")

# ── Exact Diagonalization ─────────────────────────────────────────────
include("ED/operators.jl")
include("ED/hamiltonian.jl")
include("ED/eigensolver.jl")
include("ED/observables.jl")

# ── Exports ───────────────────────────────────────────────────────────────

# Types
export TensorNetwork, MPS, MPO, MPDO, Environment
export DMRGOptions, TDVPOptions
export MPSState
export AbstractSite, SpinSite, SuperSite
export Channel, Spin
export FiniteRangeCoupling, ExpChannelCoupling, PowerLawCoupling, Field
export LanczosSolver, KrylovExponential

# Core functions
export spin_ops

# Builders
export build_mpo, build_FSM
export build_xxz_mpo
export product_state, random_state

# Algorithms
export dmrg_sweep, tdvp_sweep

# Measurements
export measure_energy, measure_norm, energy_variance
export measure_local_observable, measure_correlation

# TensorOps
export make_canonical, is_orthogonal
export entropy, truncation_error
export mps_to_vector, mpo_to_matrix

# ED: operators
export embed_operator, embed_two_site, embed_multi_site
export basis_state, neel_state

# ED: hamiltonian
export build_xxz_hamiltonian

# ED: eigensolver
export EDEigensystem, diagonalize, ground_state, low_energy_states

# ED: observables
export ed_expectation, ed_local_expectation, ed_correlation
export ed_local_profile, ed_correlation_matrix
export ed_time_evolve, ed_evolve_and_measure

end # module QuenchDyn
