# Core/states.jl

"""
    MPSState{Tmps,Tmpo,Tenv}

Bundles the MPS, MPO, environment tensors, and orthogonality center for
DMRG/TDVP-style sweeps. Constructed from an `MPS` and `MPO`; the constructor
canonicalizes the MPS in-place and builds the matching environment.
"""
mutable struct MPSState{Tmps,Tmpo,Tenv}
    mps::MPS{Tmps}
    mpo::MPO{Tmpo}
    environment::Environment{Tenv}
    center::Int
end

function MPSState(mps::MPS{Tmps}, mpo::MPO{Tmpo}; center=1) where {Tmps,Tmpo}
    # Environment type is the promotion of MPS and MPO types
    Tenv = promote_type(Tmps, Tmpo)
    
    # NO CONVERSIONS OR COPIES! Just use the inputs directly
    make_canonical(mps, center)  # Modifies in-place
    
    # Build environment with natural type promotion
    env = _build_environment(mps, mpo, center)
    
    return MPSState{Tmps,Tmpo,Tenv}(mps, mpo, env, center)
end
