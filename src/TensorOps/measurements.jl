# TensorOps/measurements.jl

"""
    measure_energy(state::MPSState)

Compute ⟨ψ|H|ψ⟩ for an MPSState by contracting the full MPS-MPO-MPS sandwich.

The MPS should be in canonical form. The energy is computed by sweeping
left-to-right, building the left environment, and reading off the scalar
at the end.
"""
function measure_energy(state::MPSState)
    mps = state.mps
    mpo = state.mpo
    N = length(mps.tensors)
    Tenv = promote_type(eltype(mps.tensors[1]), eltype(mpo.tensors[1]))

    # Start with trivial left boundary
    L = ones(Tenv, 1, 1, 1)

    # Contract site by site from left to right
    for i in 1:N
        L = _contract_left_environment(L, mps.tensors[i], mpo.tensors[i])
    end

    # L is now (1,1,1) — the scalar ⟨ψ|H|ψ⟩
    energy = real(L[1, 1, 1])
    return energy
end

"""
    energy_variance(state::MPSState)

Compute ⟨ψ|H²|ψ⟩ − ⟨ψ|H|ψ⟩². Equals zero (up to truncation) at an exact
eigenstate of H, so it is the gold-standard convergence diagnostic for
DMRG ground-state runs. Costs O(N · χ³ · D² · d) per call.
"""
function energy_variance(state::MPSState)
    mps = state.mps
    mpo = state.mpo
    N = length(mps.tensors)
    Tenv = promote_type(eltype(mps.tensors[1]), eltype(mpo.tensors[1]))

    E = measure_energy(state)

    # 4-index left environment: [bra_bond, upper_mpo_bond, lower_mpo_bond, ket_bond]
    L = ones(Tenv, 1, 1, 1, 1)
    for i in 1:N
        A = mps.tensors[i]
        W = mpo.tensors[i]
        @tensoropt L_new[-1,-2,-3,-4] := L[5,6,7,8] * conj(A)[5,9,-1] *
                                         W[6,-2,9,10] * W[7,-3,10,11] *
                                         A[8,11,-4]
        L = L_new
    end
    H_sq = real(L[1, 1, 1, 1])

    return H_sq - E^2
end

"""
    measure_norm(state::MPSState)

Compute ⟨ψ|ψ⟩ by contracting MPS with itself (no MPO).
"""
function measure_norm(mps::MPS)
    N = length(mps.tensors)
    T = eltype(mps.tensors[1])

    # Transfer matrix contraction: start with (1,1) identity
    C = ones(T, 1, 1)

    for i in 1:N
        A = mps.tensors[i]
        # C[a,b] * conj(A)[a,s,c] * A[b,s,d] → C_new[c,d]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    return real(C[1, 1])
end

"""
    measure_local_observable(mps::MPS, op::Matrix, site_idx::Int)

Compute ⟨ψ|O_site|ψ⟩ for operator `op` acting on site `site_idx`.
The MPS should be normalized.
"""
function measure_local_observable(mps::MPS, op::Matrix, site_idx::Int)
    N = length(mps.tensors)
    @assert 1 ≤ site_idx ≤ N "site_idx must be in 1:$N"
    T = promote_type(eltype(mps.tensors[1]), eltype(op))

    # Build left environment up to site_idx - 1 (just overlap, no MPO)
    C = ones(T, 1, 1)
    for i in 1:site_idx-1
        A = mps.tensors[i]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    # Insert operator at site_idx
    A = mps.tensors[site_idx]
    op_T = convert(Matrix{T}, op)
    @tensoropt C_op[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * op_T[5, 6] * A[4, 6, -2]
    C = C_op

    # Continue contracting to the right (overlap only)
    for i in site_idx+1:N
        A = mps.tensors[i]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    return C[1, 1]
end

"""
    measure_correlation(mps::MPS, op_L::Matrix, site_L::Int, op_R::Matrix, site_R::Int)

Compute ⟨ψ|O_L(site_L) O_R(site_R)|ψ⟩ for sites site_L < site_R.
"""
function measure_correlation(mps::MPS, op_L::Matrix, site_L::Int, op_R::Matrix, site_R::Int)
    N = length(mps.tensors)
    @assert 1 ≤ site_L < site_R ≤ N "Need site_L < site_R in 1:$N"
    T = promote_type(eltype(mps.tensors[1]), eltype(op_L), eltype(op_R))

    # Build left environment up to site_L - 1
    C = ones(T, 1, 1)
    for i in 1:site_L-1
        A = mps.tensors[i]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    # Insert op_L at site_L
    A = mps.tensors[site_L]
    op_L_T = convert(Matrix{T}, op_L)
    @tensoropt C_op[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * op_L_T[5, 6] * A[4, 6, -2]
    C = C_op

    # Propagate (overlap) from site_L+1 to site_R-1
    for i in site_L+1:site_R-1
        A = mps.tensors[i]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    # Insert op_R at site_R
    A = mps.tensors[site_R]
    op_R_T = convert(Matrix{T}, op_R)
    @tensoropt C_op2[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * op_R_T[5, 6] * A[4, 6, -2]
    C = C_op2

    # Continue to the end
    for i in site_R+1:N
        A = mps.tensors[i]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    return C[1, 1]
end

"""
    measure_local_profile(mps::MPS, op::Matrix)

Vector of single-site expectations ⟨ψ|op_i|ψ⟩ for i = 1, …, N.
MPS analog of `ed_local_profile`.
"""
function measure_local_profile(mps::MPS, op::Matrix)
    N = length(mps.tensors)
    return [real(measure_local_observable(mps, op, i)) for i in 1:N]
end

"""
    measure_correlation_matrix(mps::MPS, op1::Matrix, op2::Matrix)

N×N matrix C[i,j] = ⟨ψ|op1_i op2_j|ψ⟩. Off-diagonal entries use
`measure_correlation`; the diagonal uses C[i,i] = ⟨ψ|(op1·op2)_i|ψ⟩. Both
triangles are computed explicitly (C[i,j] and C[j,i] coincide only when
op1 = op2). MPS analog of `ed_correlation_matrix`.

Cost is O(N³): O(N²) pairs, each a full O(N) MPS contraction. Fine for a
one-off measurement; reuse a left-environment sweep if it becomes a bottleneck.
"""
function measure_correlation_matrix(mps::MPS, op1::Matrix, op2::Matrix)
    N = length(mps.tensors)
    op_diag = op1 * op2
    C = zeros(Float64, N, N)
    for i in 1:N
        C[i, i] = real(measure_local_observable(mps, op_diag, i))
        for j in i+1:N
            C[i, j] = real(measure_correlation(mps, op1, i, op2, j))
            C[j, i] = real(measure_correlation(mps, op2, i, op1, j))
        end
    end
    return C
end

# ── Full-space conversions (for small systems / debugging) ────────────────

"""
    mps_to_vector(mps::MPS; d::Int=2)

Contract an MPS into a full d^N state vector.

The returned vector uses the convention that site 1 is the most significant
index, matching the ED basis ordering:
    idx = 1 + (s1-1)*d^(N-1) + (s2-1)*d^(N-2) + ... + (sN-1)

Only practical for small N (d^N fits in memory).
"""
function mps_to_vector(mps::MPS; d::Int = 2)
    N = length(mps.tensors)
    C = mps.tensors[1][1, :, :]
    for i in 2:N
        A = mps.tensors[i]
        chi_in, d_i, chi_out = size(A)
        C = C * reshape(A, chi_in, d_i * chi_out)
        C = reshape(C, size(C, 1) * d_i, chi_out)
    end
    psi = vec(C)
    # Left-to-right contraction puts site 1 as least significant (column-major).
    # Reverse to match ED convention (site 1 = most significant).
    psi_tensor = reshape(psi, ntuple(_ -> d, N))
    psi_tensor = permutedims(psi_tensor, N:-1:1)
    return vec(psi_tensor)
end

"""
    mpo_to_matrix(mpo::MPO; d::Int=2)

Contract an MPO into a full d^N x d^N operator matrix.

Uses the same basis convention as `mps_to_vector`: site 1 is the most
significant index. Only practical for small N.
"""
function mpo_to_matrix(mpo::MPO; d::Int = 2)
    N = length(mpo.tensors)
    D = d^N
    T = ComplexF64

    H = zeros(T, D, D)
    for bra_idx in 0:D-1
        for ket_idx in 0:D-1
            bra = [(bra_idx >> (N - k)) & 1 + 1 for k in 1:N]
            ket = [(ket_idx >> (N - k)) & 1 + 1 for k in 1:N]
            v = T.(mpo.tensors[1][1, :, bra[1], ket[1]])
            for site in 2:N
                Wslice = T.(mpo.tensors[site][:, :, bra[site], ket[site]])
                v = transpose(Wslice) * v
            end
            H[bra_idx + 1, ket_idx + 1] = v[1]
        end
    end
    return H
end
