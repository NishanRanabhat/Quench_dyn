# ============= Basic Tensor Contractions =============
"""
    _contract_left_environment(L::Array{T,3}, A::Array{T,3}, W::Array{T,4}) where T

Contract left environment tensor with MPS and MPO tensors.
L[a,b,c] * A[a,s,d] * W[b,e,s,s'] * conj(A)[c,s',d] -> L'[d,e,d]
"""
function _contract_left_environment(L::Array, A::Array, W::Array) 
    @tensoropt L_new[-1,-2,-3] := L[5,6,7] * conj(A)[5,4,-1] * W[6,-2,4,8] * A[7,8,-3]
    return L_new
end

"""
    _contract_right_environment(R::Array{T,3}, B::Array{T,3}, W::Array{T,4}) where T

Contract right environment tensor with MPS and MPO tensors.
"""
function _contract_right_environment(R::Array, B::Array, W::Array) 
    @tensoropt R_new[-1,-2,-3] := conj(B)[-1,4,5] * W[-2,6,4,8] * B[-3,8,7]* R[5,6,7]
    return R_new
end

# ============= environment Building =============
"""
    _build_environment(mps::MPS{T}, mpo::MPO{T}, center::Int) where T

Build environment tensors around the orthogonality center.
- env[1:center-1] contains left environments
- env[center+2:N+1] contains right environments
- env[center] and env[center+1] are nothing (at the orthogonality center)
"""

function _build_environment(mps::MPS{Tmps}, mpo::MPO{Tmpo}, center::Int) where {Tmps,Tmpo}
    #get type of environment
    Tenv = promote_type(Tmps, Tmpo)

    N = length(mps.tensors)
    @assert 1 <= center <= N "Center must be between 1 and N"
    
    # Initialize environment array
    env_tensors = Vector{Union{Array{Tenv,3}, Nothing}}(nothing, N+1)
    
    # Right/Left boundary
    env_tensors[N+1] = ones(Tenv, 1, 1, 1)
    
    # Build left environments up to (but not including) center
    for site in 1:center-1
        prev_env = site == 1 ? env_tensors[N+1] : env_tensors[site-1]
        env_tensors[site] = _contract_left_environment(
            prev_env, mps.tensors[site], mpo.tensors[site]
        )
    end
    
    # Build right environments from center+1 to N
    for site in N:-1:center+1
        next_env = env_tensors[site+1]
        env_tensors[site] = _contract_right_environment(
            next_env, mps.tensors[site], mpo.tensors[site]
        )
    end
    
    # env_tensors[center] remain nothing
    
    return Environment{Tenv}(env_tensors)
end

# ============= environment Updates =============
"""
    update_left_environment!(env::environment, site::Int, mps_tensor, mpo_tensor)

Update environment at site i by contracting from site i-1.
Used when moving orthogonality center rightward.
"""
function _update_left_environment(state::MPSState, site::Int)
    N = length(state.mps.tensors)
    @assert 1 <= site <= N "Site must be between 1 and N"
    
    # Get previous environment (use boundary env[N+1] for site 1)
    prev_env = site == 1 ? state.environment.tensors[N+1] : state.environment.tensors[site-1]
    
    # Contract to build left environment at site
    state.environment.tensors[site] = _contract_left_environment(
        prev_env,
        state.mps.tensors[site],
        state.mpo.tensors[site]
    )
end

"""
    update_right_environment!(env::environment, site::Int, mps_tensor, mpo_tensor)

Update environment at site i by contracting from site i+1.
Used when moving orthogonality center leftward.
"""

function _update_right_environment(state::MPSState, site::Int)
    N = length(state.mps.tensors)
    @assert 1 <= site <= N "Site must be between 1 and N"
    
    # Get next environment (already uses env[N+1] naturally for site N)
    next_env = state.environment.tensors[site+1]
    
    # Contract to build right environment at site
    state.environment.tensors[site] = _contract_right_environment(
        next_env,
        state.mps.tensors[site],
        state.mpo.tensors[site]
    )
end