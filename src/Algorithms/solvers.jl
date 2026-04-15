# Algorithms/solvers.jl

# ============= Effective Hamiltonian Types =============

# Abstract supertype
abstract type EffectiveHamiltonian end

# Concrete subtypes
struct OneSiteEffectiveHamiltonian <: EffectiveHamiltonian
    left_env::Array{<:Any,3}
    mpo_tensor::Array{<:Any,4}
    right_env::Array{<:Any,3}
end

struct TwoSiteEffectiveHamiltonian <: EffectiveHamiltonian
    left_env::Array{<:Any,3}
    mpo_tensor1::Array{<:Any,4}
    mpo_tensor2::Array{<:Any,4}
    right_env::Array{<:Any,3}
end

struct ZeroSiteEffectiveHamiltonian <: EffectiveHamiltonian
    left_env::Array{<:Any,3}
    right_env::Array{<:Any,3}
end

# ============= _apply Methods (Hamiltonian-Vector Products) =============

function _apply(H::OneSiteEffectiveHamiltonian, v::Vector)
    chi_l = size(H.left_env, 3)
    chi_r = size(H.right_env, 3)
    d = size(H.mpo_tensor, 4)  # Extract dimension from MPO
    
    # Reshape vector to tensor
    M = reshape(v, chi_l, d, chi_r)
    
    # Contract - matching your MpoToMpsOneSite logic
    @tensoropt M_new[-1,-2,-3] := 
        H.left_env[-1,4,5] * M[5,6,8] * 
        H.mpo_tensor[4,7,-2,6] * 
        H.right_env[-3,7,8]
    
    return vec(M_new)
end

function _apply(H::TwoSiteEffectiveHamiltonian, v::Vector)
    chi_l = size(H.left_env, 3)
    chi_r = size(H.right_env, 3)
    d1 = size(H.mpo_tensor1, 4)
    d2 = size(H.mpo_tensor2, 4)
    
    # Reshape vector to tensor
    Psi2 = reshape(v, chi_l, d1, d2, chi_r)
    
    # Contract - matching your MpoToMpsTwoSite logic
    @tensoropt Psi2_new[-1,-2,-3,-4] := 
        H.left_env[-1,5,6] * Psi2[6,7,9,11] * 
        H.mpo_tensor1[5,8,-2,7] * 
        H.mpo_tensor2[8,10,-3,9] * 
        H.right_env[-4,10,11]
    return vec(Psi2_new)
end

function _apply(H::ZeroSiteEffectiveHamiltonian, v::Vector)
    chi_l = size(H.left_env, 3)
    chi_r = size(H.right_env, 3)
    
    # Reshape vector to matrix
    C = reshape(v, chi_l, chi_r)
    
    # Contract - matching your MpoToMpsOneSiteKeff logic
    @tensoropt C_new[-1,-2] := 
        H.left_env[-1,3,4] * C[4,5] * 
        H.right_env[-2,3,5]
    
    return vec(C_new)
end


# ============= Solver Types =============

"""
#Lanczos solver for eigenvalue problems (DMRG)
"""
#returns default value if not given new
struct LanczosSolver
    krylov_dim::Int
    max_iter::Int
    
    function LanczosSolver(krylov_dim=4, max_iter=100)
        new(krylov_dim, max_iter)
    end
end

"""
#Krylov exponential solver for time evolution (TDVP)
"""

struct KrylovExponential
    krylov_dim::Int
    tol::Float64
    evol_type::String  # :real or :imaginary
    
    function KrylovExponential(krylov_dim=30, tol=1e-12, evol_type="real")
        @assert evol_type in ("real", "imaginary") "evol_type must be real or imaginary"
        new(krylov_dim, tol, evol_type)
    end
end

# ============= Eigenvalue Solver =============

"""
#   solve(solver::LanczosSolver, H, v_init)

#Find the lowest eigenvalue and eigenvector of H using Lanczos algorithm.
"""
function _solve(solver::LanczosSolver, H::EffectiveHamiltonian, v_init::Vector{T};
                tol::Float64=1e-12) where T
    # Handle zero initial vector
    if norm(v_init) == 0
        v_init = randn(T, length(v_init))
    end

    # Initialize
    n = length(v_init)
    V = zeros(T, n, solver.krylov_dim + 1)
    H_mat = zeros(T, solver.krylov_dim, solver.krylov_dim)

    eigenval = real(T)(Inf)
    prev_eigenval = real(T)(Inf)
    eigenvec = v_init / norm(v_init)

    # Lanczos iterations with convergence check
    for iter in 1:solver.max_iter
        # Restart with current best eigenvector
        V[:, 1] = eigenvec / norm(eigenvec)
        H_mat .= zero(T)
        krylov_size = solver.krylov_dim
        # Build Krylov subspace
        for p = 2:solver.krylov_dim+1
            # _apply Hamiltonian
            V[:,p] = _apply(H,V[:,p-1])
            # Orthogonalize using modified Gram-Schmidt
            for g = p-2:1:p-1
                if g >= 1
                    H_mat[p-1,g] = dot(V[:,p],V[:,g]);
                    H_mat[g,p-1] = conj(H_mat[p-1,g]);# Maintain symmetry
                end
            end
            for g = 1:1:p-1
                V[:,p] = V[:,p] - dot(V[:,g],V[:,p])*V[:,g];
            end
            vnorm = norm(V[:,p])
            if vnorm < 1e-12
                # Krylov space exhausted — use subspace built so far
                krylov_size = p - 1
                break
            end
            V[:,p] = V[:,p] / vnorm
        end

        H_sub = Hermitian(0.5*(H_mat[1:krylov_size, 1:krylov_size] + H_mat[1:krylov_size, 1:krylov_size]'))
        G = eigen(H_sub);
        prev_eigenval = eigenval
        eigenval, xloc = findmin(G.values);
        eigenvec = V[:,1:krylov_size]*G.vectors[:,xloc[1]];

        # Converged when eigenvalue stops changing
        if iter > 1 && abs(eigenval - prev_eigenval) < tol * max(1.0, abs(eigenval))
            break
        end
    end
    return eigenvec / norm(eigenvec), eigenval
end

# ============= Time Evolution Solver =============

"""
#    evolve(solver::KrylovExponential, H, v_init, dt)

#Evolve state v_init by time dt under Hamiltonian H using Krylov method.
"""

function _evolve(solver::KrylovExponential, H::EffectiveHamiltonian, v_init::Vector{T}, dt::Real) where T
    # Handle zero initial vector
    if norm(v_init) == 0
        v_init = randn(T, length(v_init))
    end
    
    n = length(v_init)
    V = zeros(T, n, solver.krylov_dim + 1)
    A_mat = zeros(T, solver.krylov_dim, solver.krylov_dim)
    output_vec = zeros(T,n)
    transit_vec = zeros(T,n)

    # Normalize and store norm
    v_norm = norm(v_init)
    V[:, 1] = v_init / v_norm

    for p = 2:solver.krylov_dim+1
        output_vec = 0*output_vec
        V[:,p] = _apply(H,V[:,p-1]) 
        for g = p-2:1:p-1
            if g >= 1
                A_mat[p-1,g] = dot(V[:,p],V[:,g]);
                A_mat[g,p-1] = conj(A_mat[p-1,g]);
            end
        end
        for g = 1:1:p-1
            V[:,p] = (V[:,p] - dot(V[:,g],V[:,p])*V[:,g])
        end
        V[:,p] = V[:,p]/max(norm(V[:,p]),1e-16);
        if p > 3
            output_vec = _evol(A_mat[1:p-1,1:p-1],V,dt,output_vec,solver.evol_type) 
            c = _closeness(transit_vec,output_vec,solver.tol)
            if c == length(output_vec)
                break
            else
                transit_vec = output_vec
            end
        end
    end
    return v_norm * output_vec
end

function _evol(mat, vect, dt, output_vec, evol_type::String)
    if evol_type == "real"
        c = exp(-im * dt * mat) * I(length(mat[:,1]))[:, 1]
    else  # "imaginary"
        c = exp(-dt * mat) * I(length(mat[:,1]))[:, 1]
    end
    for i in 1:length(c)
        output_vec += c[i] * vect[:, i]
    end
    return output_vec
end

function _closeness(list1,list2,cutoff)
    c = 0
    for i in 1:length(list1)
        if abs(list1[i]-list2[i]) <= cutoff
            c += 1
        end
    end
    return c
end

