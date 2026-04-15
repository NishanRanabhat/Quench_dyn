# Core/fsm.jl

# ──────────────────────────────────────────────────────────────────────────────
# 1) Abstract hierarchy
# ──────────────────────────────────────────────────────────────────────────────

abstract type Channel end
abstract type Spin  <: Channel end
# ──────────────────────────────────────────────────────────────────────────────
# 2) Pure-spin channel structs
# ──────────────────────────────────────────────────────────────────────────────

"""
    FiniteRangeCoupling(op1, op2, dx, weight)

Finite-range coupling between spins at sites i and i+dx:

- `op1`, `op2` : names of the operators on the two spins  
- `dx`         : distance between the spins  
- `weight`     : coupling strength  
"""
struct FiniteRangeCoupling <: Spin
    op1::Symbol
    op2::Symbol
    dx::Int
    weight::Float64
end

"""
    ExpChannelCoupling(op1, op2, amplitude, decay)

Exponential two-spin coupling:

- coupling ∝ `amplitude * decay^r` at distance `r`  
- `op1`, `op2` : operator names on the two spins  
- `amplitude`  : overall prefactor  
- `decay`      : base of the exponential  
"""
struct ExpChannelCoupling <: Spin
    op1::Symbol
    op2::Symbol
    amplitude::Float64
    decay::Float64
end

"""
PowerLawCoupling(op1, op2, alpha, bondH, N)

Exponential two-spin coupling:

- coupling ∝ `1/r^alpha` at distance `r`  
- `op1`, `op2` : operator names on the two spins  
- `alpha`  : power of the power law  
- `bondH`  : number of exponentials to approximate the power-law
- `N`  : number of spins
"""
struct PowerLawCoupling <: Spin
    op1::Symbol
    op2::Symbol
    J::Float64
    alpha::Float64
    bondH::Int
    N::Int
end

"""
    Field(op, weight)

Single-site field coupling:

- `op`     : operator acting on the spin  
- `weight` : field strength  
"""
struct Field <: Spin
    op::Symbol
    weight::Float64
end

# ──────────────────────────────────────────────────────────────────────────────
# 3) Pure-spin build path methods
# ──────────────────────────────────────────────────────────────────────────────

# for finite‐range couplings
function _build_path(ns::Int, coupling::FiniteRangeCoupling,
    transitions::Vector{Tuple{Int64, Int64, Symbol, Float64}})
    path = ns+1 : ns+coupling.dx
    # idle → first state : emit op1
    push!(transitions, (path[1], 1, coupling.op1, 1.0))
    # intermediate identity hops
    for i in 2:length(path)
        push!(transitions, (path[i], path[i-1],:I, 1.0))
    end
    # last state → final idle : emit op2
    push!(transitions, (0, path[end], coupling.op2, coupling.weight))
    return path[end], transitions
end

# for exponential couplings
function _build_path(ns::Int, coupling::ExpChannelCoupling,
    transitions::Vector{Tuple{Int64, Int64, Symbol, Float64}})
    path = ns + 1
    # idle → first state : emit op1
    push!(transitions, (path, 1, coupling.op1,1.0))
    # self‑loop with identity and decay 
    push!(transitions, (path, path, :I, coupling.decay))
    # last state → final idle : emit op2
    push!(transitions, (0,    path, coupling.op2,
            coupling.amplitude * coupling.decay))
    return path, transitions
end

function _build_path(ns::Int,coupling::PowerLawCoupling,
    transitions::Vector{Tuple{Int64, Int64, Symbol, Float64}})

    nu, lambda = _power_law_to_exp(coupling.alpha,coupling.bondH,coupling.N)
    path = ns+1 : ns+coupling.bondH
    #loop over the several exponential paths
    for i in 1:length(path)
        push!(transitions,(path[i],1,coupling.op1,1.0))
        push!(transitions,(path[i],path[i],:I,lambda[i]))
        push!(transitions,(0,path[i],coupling.op2,coupling.J*nu[i]*lambda[i]))
    end
    return path[end],transitions
end

# for single‐site fields
function _build_path(ns::Int, coupling::Field,
    transitions::Vector{Tuple{Int64, Int64, Symbol, Float64}})
    # first idle → last : emit op
    push!(transitions, (0, 1, coupling.op, coupling.weight))
    return ns, transitions
end

# ──────────────────────────────────────────────────────────────────────────────
# 4) helper to convert power-law to sum of exponentials
# ──────────────────────────────────────────────────────────────────────────────

"""
function gives (x_i,lambda_i) such that
1/r^a = Sum_{i=1-->n} x_i * (lambda_i)^r + errors
a : interaction strength, a -> infinity is nearest neighbor Ising
    a -> 0 is fully connected Ising.
n : number of exponential sums. Refer to SciPostPhys.12.4.126 appendix C
    for further details.
N : lattice size.
"""

function _power_law_to_exp(a::Float64,n::Integer,N::Integer)

    F = Array{Float64,1}(undef,N)
    @inbounds for k in 1:N
        F[k] = 1/k^a
    end

    M = zeros(N-n+1,n)
    @inbounds for j in 1:n
        @inbounds for i in 1:N-n+1
            M[i,j] = F[i+j-1]
        end
    end

    F1 = qr(M)
    Q1 = F1.Q[1:N-n,1:n]
    Q1_inv = pinv(Q1)
    Q2 = F1.Q[2:N-n+1,1:n]
    V = Q1_inv*Q2

    lambda = real(eigvals(V))
    lam_mat = zeros(N,n)
    @inbounds for i in 1:length(lambda)
        @inbounds for k in 1:N
            lam_mat[k,i] = lambda[i]^k
        end
    end

    nu = lam_mat\F
    return nu, lambda
end

# ──────────────────────────────────────────────────────────────────────────────
# 5) FSMPath wrappers
# ──────────────────────────────────────────────────────────────────────────────

abstract type FSMPath end

"""
    SpinFSMPath(χ, transitions)

Result of `spin_FSM`: χ (bond dimension) and the pure-spin transitions.
"""
struct SpinFSMPath <: FSMPath
    chi::Int
    transitions::Vector{Tuple{Int,Int,Symbol,Float64}}
end

# ──────────────────────────────────────────────────────────────────────────────
# 6) Finite state Machine builder from paths
# ──────────────────────────────────────────────────────────────────────────────

#builds spin finite state machine from different channels
function build_FSM(channels::Vector{<:Spin};ns=1) #ns=1 default value for a fresh FSM

    transitions = Tuple{Int,Int,Symbol,Float64}[]
    if ns == 1
        push!(transitions, (1,1,:I,1.0))
    end

    push!(transitions, (0,0,:I,1.0))

    for ch in channels
        ns, transitions = _build_path(ns, ch, transitions)
    end

    final = ns + 1
    transitions = [
      (s == 0 ? final : s,
       t == 0 ? final : t,
       op, w)
      for (s,t,op,w) in transitions
    ]

    return SpinFSMPath(final, transitions)
end
