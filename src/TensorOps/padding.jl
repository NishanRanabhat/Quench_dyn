# TensorOps/padding.jl
#
# Bond-dimension padding for MPS. One-site DMRG/TDVP cannot grow χ on its
# own, so we pad a low-χ MPS into a larger working manifold by filling the
# new directions with small random noise.

"""
    pad_mps(mps::MPS, chi_target::Int; noise=1e-6)

Return a new MPS with each interior bond padded up to
`min(chi_target, d^min(i, N-i))` (the position-aware natural ceiling).
Bonds already at or above their effective target are left untouched.

The original tensor values are placed in the top-left-front block of each
new tensor; the remaining entries are filled with random noise of magnitude
`noise`. The result is re-canonicalized to center=1 and renormalized to
unit norm.

Use this to lift a low-χ MPS (e.g., a DMRG ground state at χ_DMRG) into
the working manifold for one-site TDVP, which itself cannot grow χ.

Trade-off on `noise`:
- Too small (e.g., 1e-14) → padded directions sit at numerical zero, dynamics
  cannot spread weight into them and the bond is effectively dead.
- Too large (e.g., 1e-3) → initial state observables shift visibly.
- Default `1e-6` is the standard compromise: padded weight is O(noise²) ≈
  1e-12 relative to the original state.

Notes:
- Input is not modified.
- Element type of the input MPS is preserved.
- Local dimension `d` is inferred from `size(mps.tensors[1], 2)`.
"""
function pad_mps(mps::MPS{T}, chi_target::Int; noise::Real=1e-6) where T
    N = length(mps.tensors)
    @assert N >= 2 "Need at least 2 sites"
    @assert chi_target >= 1 "chi_target must be ≥ 1"

    d = size(mps.tensors[1], 2)

    # Current bond dimensions (right bond of each site, length N-1).
    current_bonds = Int[size(mps.tensors[i], 3) for i in 1:N-1]

    # Effective target bond at each interior bond i (between sites i, i+1):
    #   ceiling = min(chi_target, natural max from either side)
    #   new_bond = max(current, ceiling)
    new_bonds = Int[
        max(current_bonds[i], min(chi_target, d^min(i, N-i)))
        for i in 1:N-1
    ]

    # Build padded tensors.
    new_tensors = Vector{Array{T,3}}(undef, N)
    new_tensors[1] = _pad_tensor(mps.tensors[1], 1, d, new_bonds[1], noise)
    for i in 2:N-1
        new_tensors[i] = _pad_tensor(mps.tensors[i], new_bonds[i-1], d, new_bonds[i], noise)
    end
    new_tensors[N] = _pad_tensor(mps.tensors[N], new_bonds[N-1], d, 1, noise)

    padded = MPS{T}(new_tensors)

    # Re-canonicalize (SVD-based, keeps all singular values → padded bonds
    # survive) and renormalize to unit norm.
    make_canonical(padded, 1)
    nrm_sq = real(measure_norm(padded))
    if nrm_sq > 0
        padded.tensors[1] ./= sqrt(nrm_sq)
    end

    return padded
end

"""
    _pad_tensor(A, chi_l_new, d, chi_r_new, noise)

Place `A` into the top-left-front block of a new tensor of shape
`(chi_l_new, d, chi_r_new)`; fill the rest with random noise of magnitude
`noise`. Returns the new tensor (copy of `A` if shapes already match).
"""
function _pad_tensor(A::Array{T,3}, chi_l_new::Int, d::Int, chi_r_new::Int, noise::Real) where T
    chi_l_old, d_old, chi_r_old = size(A)
    @assert d_old == d "local dim mismatch: tensor has $d_old, expected $d"
    @assert chi_l_new >= chi_l_old "target left bond $chi_l_new < current $chi_l_old"
    @assert chi_r_new >= chi_r_old "target right bond $chi_r_new < current $chi_r_old"

    if chi_l_new == chi_l_old && chi_r_new == chi_r_old
        return copy(A)
    end

    B = T.(noise) .* randn(T, chi_l_new, d, chi_r_new)
    B[1:chi_l_old, :, 1:chi_r_old] .= A
    return B
end
