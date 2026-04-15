# ED/observables.jl
#
# Expectation values, time evolution, and correlation measurements
# on exact states. All functions work with plain state vectors and
# operator matrices — no coupling to the eigensolver or Hamiltonian builder.

# ── Expectation values ────────────────────────────────────────────────────

"""
    ed_expectation(psi::AbstractVector, O::AbstractMatrix)

Compute ⟨ψ|O|ψ⟩. Returns a real number (takes the real part, appropriate
for Hermitian observables).
"""
function ed_expectation(psi::AbstractVector, O::AbstractMatrix)
    return real(dot(psi, O * psi))
end

"""
    ed_local_expectation(psi::AbstractVector, op::AbstractMatrix,
                         site::Int, N::Int; d::Int=2)

Compute ⟨ψ|O_site|ψ⟩ where O is a local operator embedded at `site`.
"""
function ed_local_expectation(psi::AbstractVector, op::AbstractMatrix,
                              site::Int, N::Int; d::Int = 2)
    O_full = embed_operator(op, site, N; d = d)
    return ed_expectation(psi, O_full)
end

"""
    ed_correlation(psi::AbstractVector, op1::AbstractMatrix, site1::Int,
                   op2::AbstractMatrix, site2::Int, N::Int; d::Int=2)

Compute ⟨ψ|O1_{site1} O2_{site2}|ψ⟩.
"""
function ed_correlation(psi::AbstractVector, op1::AbstractMatrix, site1::Int,
                        op2::AbstractMatrix, site2::Int, N::Int; d::Int = 2)
    O_full = embed_two_site(op1, op2, site1, site2, N; d = d)
    return ed_expectation(psi, O_full)
end

# ── Profiles and bulk measurements ────────────────────────────────────────

"""
    ed_local_profile(psi::AbstractVector, op::AbstractMatrix, N::Int; d::Int=2)

Compute ⟨ψ|O_i|ψ⟩ for every site i = 1, ..., N.
Returns a Vector{Float64} of length N.
"""
function ed_local_profile(psi::AbstractVector, op::AbstractMatrix, N::Int;
                          d::Int = 2)
    return [ed_local_expectation(psi, op, i, N; d = d) for i in 1:N]
end

"""
    ed_correlation_matrix(psi::AbstractVector, op1::AbstractMatrix,
                          op2::AbstractMatrix, N::Int; d::Int=2)

Compute the N×N matrix C[i,j] = ⟨ψ|O1_i O2_j|ψ⟩ for all pairs.
Diagonal elements C[i,i] = ⟨ψ|(O1·O2)_i|ψ⟩.
"""
function ed_correlation_matrix(psi::AbstractVector, op1::AbstractMatrix,
                               op2::AbstractMatrix, N::Int; d::Int = 2)
    C = zeros(Float64, N, N)
    op_diag = op1 * op2
    for i in 1:N
        C[i, i] = ed_local_expectation(psi, op_diag, i, N; d = d)
        for j in i+1:N
            c_ij = ed_correlation(psi, op1, i, op2, j, N; d = d)
            C[i, j] = c_ij
            C[j, i] = c_ij
        end
    end
    return C
end

# ── Time evolution ────────────────────────────────────────────────────────

"""
    ed_time_evolve(eig::EDEigensystem, psi0::AbstractVector, t::Float64)

Exact time evolution:  |ψ(t)⟩ = exp(-i H t) |ψ₀⟩

Uses a pre-computed eigensystem to avoid repeated diagonalizations.
Expand ψ₀ in the eigenbasis, apply phase factors, transform back.
"""
function ed_time_evolve(eig::EDEigensystem, psi0::AbstractVector, t::Float64)
    # coefficients in eigenbasis
    c = eig.vectors' * psi0
    # apply time-evolution phases
    c_t = c .* exp.(-im .* eig.values .* t)
    # transform back
    return eig.vectors * c_t
end

"""
    ed_time_evolve(H::Hermitian, psi0::AbstractVector, t::Float64)

Exact time evolution by diagonalizing H on the fly.
Use the eigensystem variant if evolving to multiple times.
"""
function ed_time_evolve(H::Hermitian, psi0::AbstractVector, t::Float64)
    return ed_time_evolve(diagonalize(H), psi0, t)
end

"""
    ed_evolve_and_measure(eig::EDEigensystem, psi0::AbstractVector,
                          times::AbstractVector{Float64},
                          measure_fn::Function)

Evolve |ψ₀⟩ to each time in `times` and apply `measure_fn(psi_t)` at
each step. Returns a Vector of whatever `measure_fn` returns.

`measure_fn` should accept a state vector and return the desired observable(s).

# Example
```julia
# measure Sz profile at each time
results = ed_evolve_and_measure(eig, psi0, 0.0:0.1:10.0) do psi
    ed_local_profile(psi, Sz, N)
end
```
"""
function ed_evolve_and_measure(eig::EDEigensystem, psi0::AbstractVector,
                               times::AbstractVector{Float64},
                               measure_fn::Function)
    # pre-compute eigenbasis coefficients once
    c = eig.vectors' * psi0
    results = Vector{Any}(undef, length(times))
    for (k, t) in enumerate(times)
        c_t = c .* exp.(-im .* eig.values .* t)
        psi_t = eig.vectors * c_t
        results[k] = measure_fn(psi_t)
    end
    return results
end
