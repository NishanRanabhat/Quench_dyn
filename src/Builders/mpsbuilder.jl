"""
    product_state_mps(sites, labels)

Build a bond-dim-1 MPS by looping over `sites[i]` and calling
`state_tensor(sites[i], labels[i])`.
"""

function product_state(sites::Vector{<:AbstractSite}, labels::Vector)
    N = length(sites)
    @assert length(labels)==N
    Ts = [_state_tensor(sites[i], labels[i]) for i in 1:N]
    return MPS{eltype(Ts[1])}(Ts)
end

function random_state(sites::Vector{<:AbstractSite},bond_dim::Int;T=ComplexF64)
    N = length(sites)
    Ts = Vector{Array{T,3}}(undef, N)
    # left edge
    Ts[1] = rand(T, 1,sites[1].dim, bond_dim)
    # bulk
    for i in 2:N-1
        Ts[i] = rand(T, bond_dim,sites[i].dim, bond_dim)
    end
    # right edge
    Ts[N] = rand(T, bond_dim,sites[N].dim, 1)
    return MPS{T}(Ts)
end


