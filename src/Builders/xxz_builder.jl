# Builders/xxz_builder.jl
#
# MPO construction for the 1D XXZ chain with site-dependent longitudinal field:
#
#   H = J Σ_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1})
#     + Δ Σ_i  Sz_i Sz_{i+1}
#     + Σ_i    h_i  Sz_i
#
# Each site gets its own MPO tensor (different h_i).
# Bond dimension = 5 for spin-1/2.

"""
    build_xxz_mpo(N, J, Delta, h::AbstractVector; d=2, T=Float64)

Build the MPO for the 1D XXZ model with site-dependent longitudinal field.

Returns `MPO{T}` with per-site tensors (bond dimension 5 in the bulk,
1 at the boundaries).

# Arguments
- `N::Int`       : number of sites
- `J::Float64`   : XY coupling  (Sx·Sx + Sy·Sy)
- `Delta::Float64`: Ising anisotropy (Sz·Sz)
- `h::AbstractVector{Float64}`: longitudinal field, length N
- `d::Int=2`     : local Hilbert space dimension
- `T::Type=Float64`: element type of the MPO tensors
"""
function build_xxz_mpo(N::Int, J::Float64, Delta::Float64,
                       h::AbstractVector{Float64};
                       d::Int = 2, T::Type = Float64)

    @assert length(h) == N "Field vector h must have length N=$N, got $(length(h))"
    @assert N >= 2 "Need at least 2 sites"

    ops = spin_ops(d)
    Id = T.(ops[:I])
    Sz = T.(ops[:Z])
    Sp = T.(ops[:Sp])
    Sm = T.(ops[:Sm])

    chi = 5   # MPO bond dimension

    # ── Build per-site bulk W-matrices ────────────────────────────────────
    #
    # W[left_bond, right_bond, bra, ket]   (upper-triangular convention)
    #
    # Bond states:
    #   1 = left idle     (identity propagation, not yet coupled)
    #   2 = S+ placed     (left-site S+ emitted, waiting for S- on right)
    #   3 = S- placed     (left-site S- emitted, waiting for S+ on right)
    #   4 = Sz placed     (left-site Sz emitted, waiting for Sz on right)
    #   5 = right idle    (coupling completed, identity propagation)
    #
    # Row 1 (first row) : left-site operators + on-site field
    # Col 5 (last col)  : right-site operators
    #
    # For coupling  A_i · B_{i+1} :  A goes in first row,  B in last col.
    #
    # W =  | I       (J/2)·S+   (J/2)·S-   Δ·Sz    h_i·Sz |
    #      | 0        0          0          0       S-     |
    #      | 0        0          0          0       S+     |
    #      | 0        0          0          0       Sz     |
    #      | 0        0          0          0        I     |

    tensors = Vector{Array{T,4}}(undef, N)

    for i in 1:N
        W = zeros(T, chi, chi, d, d)

        # row 1 (left idle): left-site coupling operators + on-site field
        W[1, 1, :, :] = Id                     # idle propagation
        W[1, 2, :, :] = (J / 2) .* Sp          # left op for (J/2) S+_i S-_{i+1}
        W[1, 3, :, :] = (J / 2) .* Sm          # left op for (J/2) S-_i S+_{i+1}
        W[1, 4, :, :] = Delta   .* Sz          # left op for Δ Sz_i Sz_{i+1}
        W[1, 5, :, :] = h[i]    .* Sz          # on-site field h_i Sz_i

        # col 5 (last column): right-site coupling operators
        W[2, 5, :, :] = Sm                     # right op: S- pairs with S+ → (J/2) S+_i S-_{i+1}
        W[3, 5, :, :] = Sp                     # right op: S+ pairs with S- → (J/2) S-_i S+_{i+1}
        W[4, 5, :, :] = Sz                     # right op: Sz pairs with Sz → Δ Sz_i Sz_{i+1}
        W[5, 5, :, :] = Id                     # idle propagation

        tensors[i] = W
    end

    # ── Boundary slicing ──────────────────────────────────────────────────
    # Left boundary  : pick first row  (left-idle, index 1)
    # Right boundary : pick last col   (right-idle, index chi)

    tensors[1] = reshape(tensors[1][1, :, :, :], (1, chi, d, d))
    tensors[N] = reshape(tensors[N][:, chi, :, :], (chi, 1, d, d))

    return MPO{T}(tensors)
end

# convenience: uniform longitudinal field
function build_xxz_mpo(N::Int, J::Float64, Delta::Float64,
                       h_uniform::Float64;
                       d::Int = 2, T::Type = Float64)
    return build_xxz_mpo(N, J, Delta, fill(h_uniform, N); d = d, T = T)
end

# convenience: zero field
function build_xxz_mpo(N::Int, J::Float64, Delta::Float64;
                       d::Int = 2, T::Type = Float64)
    return build_xxz_mpo(N, J, Delta, zeros(Float64, N); d = d, T = T)
end
