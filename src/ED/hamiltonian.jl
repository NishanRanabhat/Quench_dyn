# ED/hamiltonian.jl
#
# Hamiltonian construction using the embedding primitives from operators.jl.
# Currently provides the XXZ model; add other models as new functions.

"""
    build_xxz_hamiltonian(N::Int, J::Float64, Delta::Float64,
                          h::AbstractVector{Float64}; d::Int=2)

Build the full d^N × d^N Hamiltonian matrix for the 1D XXZ chain with
site-dependent longitudinal field:

    H = J Σ_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1})
      + Δ Σ_i  Sz_i Sz_{i+1}
      + Σ_i    h_i  Sz_i

Uses the identity Sx Sx + Sy Sy = (1/2)(S+ S- + S- S+) to keep the
Hamiltonian real-valued.

Returns a `Hermitian{Float64}` matrix.
"""
function build_xxz_hamiltonian(N::Int, J::Float64, Delta::Float64,
                               h::AbstractVector{Float64}; d::Int = 2)
    @assert length(h) == N "Field vector h must have length N=$N"
    @assert N >= 2 "Need at least 2 sites"

    ops = spin_ops(d)
    Sp = Float64.(real.(ops[:Sp]))
    Sm = Float64.(real.(ops[:Sm]))
    Sz = Float64.(real.(ops[:Z]))

    D = d^N
    H = zeros(Float64, D, D)

    # nearest-neighbor couplings
    for i in 1:N-1
        # XY part: J(Sx Sx + Sy Sy) = (J/2)(S+ S- + S- S+)
        H .+= (J / 2) .* embed_two_site(Sp, Sm, i, i + 1, N; d = d)
        H .+= (J / 2) .* embed_two_site(Sm, Sp, i, i + 1, N; d = d)
        # Ising part
        H .+= Delta .* embed_two_site(Sz, Sz, i, i + 1, N; d = d)
    end

    # on-site longitudinal field
    for i in 1:N
        if h[i] != 0.0
            H .+= h[i] .* embed_operator(Sz, i, N; d = d)
        end
    end

    return Hermitian(H)
end

# convenience: uniform field
function build_xxz_hamiltonian(N::Int, J::Float64, Delta::Float64,
                               h_uniform::Float64; d::Int = 2)
    return build_xxz_hamiltonian(N, J, Delta, fill(h_uniform, N); d = d)
end

# convenience: zero field
function build_xxz_hamiltonian(N::Int, J::Float64, Delta::Float64;
                               d::Int = 2)
    return build_xxz_hamiltonian(N, J, Delta, zeros(Float64, N); d = d)
end
