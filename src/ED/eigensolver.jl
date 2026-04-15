# ED/eigensolver.jl
#
# Diagonalization routines. Separated from Hamiltonian construction so that
# the same eigensystem can be reused for observables and time evolution.

"""
    EDEigensystem

Container for the full eigensystem of a Hamiltonian.
Fields:
- `values`  : sorted eigenvalues  (length D)
- `vectors` : eigenvectors as columns of a D×D matrix
               vectors[:, k] is the k-th eigenstate
"""
struct EDEigensystem
    values::Vector{Float64}
    vectors::Matrix{ComplexF64}
end

"""
    diagonalize(H::Hermitian)

Full diagonalization of H. Returns an `EDEigensystem` with eigenvalues
sorted in ascending order.
"""
function diagonalize(H::Hermitian)
    F = eigen(H)
    return EDEigensystem(F.values, ComplexF64.(F.vectors))
end

"""
    ground_state(eig::EDEigensystem)

Extract the ground state from a pre-computed eigensystem.
Returns `(energy, state_vector)`.
"""
function ground_state(eig::EDEigensystem)
    return eig.values[1], eig.vectors[:, 1]
end

"""
    ground_state(H::Hermitian)

Diagonalize and return `(energy, state_vector)` for the ground state.
Convenience wrapper when you only need the ground state.
"""
function ground_state(H::Hermitian)
    return ground_state(diagonalize(H))
end

"""
    low_energy_states(eig::EDEigensystem, n::Int)

Return the lowest `n` eigenvalues and corresponding eigenvectors.
Returns `(energies::Vector, states::Matrix)` where `states[:, k]`
is the k-th eigenstate.
"""
function low_energy_states(eig::EDEigensystem, n::Int)
    @assert 1 <= n <= length(eig.values) "n=$n out of range"
    return eig.values[1:n], eig.vectors[:, 1:n]
end
