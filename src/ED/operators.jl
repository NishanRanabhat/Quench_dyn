# ED/operators.jl
#
# Kronecker-product embedding of operators into the full Hilbert space.
# These are model-agnostic building blocks.

"""
    embed_operator(op::AbstractMatrix, site::Int, N::Int; d::Int=2)

Embed a single-site operator into the full d^N Hilbert space:

    O_full = I^{site-1} ⊗ op ⊗ I^{N-site}

Returns a d^N × d^N matrix.
"""
function embed_operator(op::AbstractMatrix, site::Int, N::Int; d::Int = 2)
    @assert 1 <= site <= N "Site $site out of range 1:$N"
    T = promote_type(eltype(op), Float64)
    Id = Matrix{T}(I, d, d)
    mats = [i == site ? T.(op) : Id for i in 1:N]
    return foldl(kron, mats)
end

"""
    embed_two_site(op1::AbstractMatrix, op2::AbstractMatrix,
                   site1::Int, site2::Int, N::Int; d::Int=2)

Embed a two-site operator product op1_{site1} ⊗ op2_{site2} into the
full Hilbert space. Sites need not be adjacent.
"""
function embed_two_site(op1::AbstractMatrix, op2::AbstractMatrix,
                        site1::Int, site2::Int, N::Int; d::Int = 2)
    @assert site1 != site2 "Sites must be distinct"
    @assert 1 <= site1 <= N && 1 <= site2 <= N "Sites out of range 1:$N"
    T = promote_type(eltype(op1), eltype(op2), Float64)
    Id = Matrix{T}(I, d, d)
    mats = Vector{Matrix{T}}(undef, N)
    for i in 1:N
        if i == site1
            mats[i] = T.(op1)
        elseif i == site2
            mats[i] = T.(op2)
        else
            mats[i] = Id
        end
    end
    return foldl(kron, mats)
end

"""
    embed_multi_site(ops::Vector{<:AbstractMatrix}, sites::Vector{Int},
                     N::Int; d::Int=2)

Embed a product of operators at specified sites into the full Hilbert space.
`ops[k]` acts on `sites[k]`. All other sites get identity.
"""
function embed_multi_site(ops::Vector{<:AbstractMatrix}, sites::Vector{Int},
                          N::Int; d::Int = 2)
    @assert length(ops) == length(sites) "ops and sites must have same length"
    @assert allunique(sites) "Sites must be distinct"
    @assert all(1 .<= sites .<= N) "Sites out of range 1:$N"
    T = promote_type(eltype.(ops)..., Float64)
    Id = Matrix{T}(I, d, d)
    site_map = Dict(zip(sites, ops))
    mats = [haskey(site_map, i) ? T.(site_map[i]) : Id for i in 1:N]
    return foldl(kron, mats)
end

"""
    basis_state(config::Vector{Int}, N::Int; d::Int=2)

Construct a computational basis state vector in the d^N Hilbert space.

`config` is a length-N vector of local state indices (1-based, from 1 to d).
For spin-1/2: 1 = spin-up (eigenvalue +1/2), 2 = spin-down (-1/2),
matching the convention Sz = diag(+1/2, -1/2).
"""
function basis_state(config::Vector{Int}, N::Int; d::Int = 2)
    @assert length(config) == N
    @assert all(1 .<= config .<= d) "State indices must be in 1:$d"
    D = d^N
    psi = zeros(ComplexF64, D)
    idx = 1
    for i in 1:N
        idx += (config[i] - 1) * d^(N - i)
    end
    psi[idx] = 1.0
    return psi
end

"""
    neel_state(N::Int; start_up::Bool=true, d::Int=2)

Construct the Neel state vector |↑↓↑↓...⟩ or |↓↑↓↑...⟩ in the full
d^N Hilbert space. Uses convention Sz = diag(+1/2, -1/2), so index 1 = up.
"""
function neel_state(N::Int; start_up::Bool = true, d::Int = 2)
    if start_up
        config = [isodd(i) ? 1 : 2 for i in 1:N]
    else
        config = [isodd(i) ? 2 : 1 for i in 1:N]
    end
    return basis_state(config, N; d = d)
end
