# Builders/mpobuilder.jl

# ————————————————————————————————————————————————————————————————
# Pure-spin MPO
# ————————————————————————————————————————————————————————————————
function build_mpo(
    fsm::SpinFSMPath;
    N::Integer,
    d::Integer = 2,
    T::Type   = Float64,
)

    @assert N > 2 "System must have at least 3 sites"
    # operator factory
    phys_ops = spin_ops(d)
    chi        = fsm.chi

    # build the bulk
    bulk = zeros(T, chi, chi, d, d)
    for (row,col,opname,w) in fsm.transitions
        op_mat = phys_ops[opname]
        bulk[row,col,:,:] .+= w*op_mat
    end

    # left / right boundaries
    L = reshape(bulk[chi, :, :, :], (1, chi, d, d))
    R = reshape(bulk[:, 1, :, :], (chi, 1, d, d))

    # assemble N‐site MPO: [L, bulk, bulk, …, bulk, R]
    mids = fill(bulk, N-2)        # N-2 copies of the central tensor
    tensors = [L, mids..., R]     # vector of length N

    return MPO{T}(tensors)
end