# ED/sectors.jl
#
# Fixed-magnetization (S^z_total) sector construction for the XXZ chain.
#
# The XXZ Hamiltonian (with a longitudinal field) conserves S^z_total, so it is
# block-diagonal in magnetization sectors. Working inside a single sector
#   (i) reaches larger N at fixed cost — the S^z=0 block of N=14 is only
#       C(14,7)=3432 states vs 2^14=16384 in the full space, and
#   (ii) is the *physically correct* ensemble for a quench: the dynamics
#        conserves S^z_total, so a quench launched from the S^z=0 sector
#        thermalizes within that sector, not the full Hilbert space.
#
# Convention (matching spin_ops / ED code): local index 1 = up (S^z=+1/2),
# 2 = down (S^z=-1/2). A config is encoded as an Int bitmask with bit (i-1) set
# iff site i is up. The bond flip amplitude for the XY term (J/2)(S+S- + S-S+)
# is J/2 for any antiparallel nearest-neighbor pair.

"""
    SzSector

Basis of the fixed-magnetization sector with `n_up` up-spins on `N` sites.

Fields:
- `N`       : number of sites
- `n_up`    : number of up-spins (S^z_total = (2·n_up − N)/2)
- `configs` : sorted Vector of Int bitmasks, one per basis state (dim = length)
- `index`   : Dict mapping bitmask → 1-based position in `configs`
- `sz`      : dim × N matrix, `sz[c, j]` = ±1/2 = S^z of site j in config c
"""
struct SzSector
    N::Int
    n_up::Int
    configs::Vector{Int}
    index::Dict{Int,Int}
    sz::Matrix{Float64}
end

Base.length(sec::SzSector) = length(sec.configs)

"""
    sz_sector(N::Int, n_up::Int)

Build the `SzSector` containing every computational-basis state of `N` sites
with exactly `n_up` up-spins. For the physically relevant zero-magnetization
sector use `n_up = N ÷ 2` (requires even `N`).
"""
function sz_sector(N::Int, n_up::Int)
    @assert 0 <= n_up <= N "n_up=$n_up out of range 0:$N"
    configs = Int[]
    for c in 0:(2^N - 1)
        count_ones(c) == n_up && push!(configs, c)
    end
    index = Dict{Int,Int}(c => k for (k, c) in enumerate(configs))
    dim = length(configs)
    sz = Matrix{Float64}(undef, dim, N)
    for (k, c) in enumerate(configs)
        for j in 1:N
            sz[k, j] = ((c >> (j - 1)) & 1) == 1 ? 0.5 : -0.5
        end
    end
    return SzSector(N, n_up, configs, index, sz)
end

"""
    build_xxz_hamiltonian_sector(sec::SzSector, J, Delta,
                                 h::AbstractVector{Float64})

Dense real-symmetric XXZ Hamiltonian projected onto the sector `sec`:

    H = J Σ_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1})
      + Δ Σ_i  Sz_i Sz_{i+1}
      + Σ_i    h_i Sz_i

built directly in the sector basis (no 2^N intermediate). Open boundaries.
Returns a `Hermitian{Float64}` of size dim × dim.
"""
function build_xxz_hamiltonian_sector(sec::SzSector, J::Float64, Delta::Float64,
                                      h::AbstractVector{Float64})
    N = sec.N
    @assert length(h) == N "Field vector h must have length N=$N"
    dim = length(sec)
    H = zeros(Float64, dim, dim)
    for (k, c) in enumerate(sec.configs)
        # diagonal: Ising bonds + longitudinal field
        diag = 0.0
        for i in 1:N-1
            diag += Delta * sec.sz[k, i] * sec.sz[k, i+1]
        end
        for i in 1:N
            diag += h[i] * sec.sz[k, i]
        end
        H[k, k] += diag
        # off-diagonal: XY hopping flips one antiparallel NN pair, amplitude J/2
        for i in 1:N-1
            up_i  = (c >> (i - 1)) & 1
            up_i1 = (c >> i) & 1
            if up_i != up_i1
                cflip = c ⊻ ((1 << (i - 1)) | (1 << i))
                kflip = sec.index[cflip]
                H[kflip, k] += J / 2
            end
        end
    end
    return Hermitian(H)
end

# convenience: uniform / zero field
function build_xxz_hamiltonian_sector(sec::SzSector, J::Float64, Delta::Float64,
                                      h_uniform::Float64)
    return build_xxz_hamiltonian_sector(sec, J, Delta, fill(h_uniform, sec.N))
end
function build_xxz_hamiltonian_sector(sec::SzSector, J::Float64, Delta::Float64)
    return build_xxz_hamiltonian_sector(sec, J, Delta, zeros(Float64, sec.N))
end

"""
    build_lr_xxz_hamiltonian_sector(sec::SzSector, J, Delta, alpha,
                                    h::AbstractVector{Float64})

Long-range XXZ chain with a power-law, STAGGERED-SIGN Ising channel and
nearest-neighbor XY exchange, projected onto the sector `sec`:

    H = J Σ_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1})
      + J Δ Σ_{i<j} (−1)^{r+1} r^{−α} Sz_i Sz_j ,   r = j − i
      + Σ_i h_i Sz_i

The alternating sign makes every bond SUPPORT the staggered (Néel) pattern —
AFM at odd r, FM at even r — i.e. the sublattice-gauge image of Dyson's
unfrustrated ferromagnet, so the Z2 staggered order has a finite-temperature
transition for α ≤ 2 (a uniform-sign AFM tail would instead frustrate the
even-r bonds and destroy the ordered phase). A uniform-sign FM channel is
also the wrong model here: its order parameter (total S^z) is conserved by
XXZ dynamics. With the staggered channel the order parameter m_s is
non-conserved (Model-A-like) while total S^z conservation — and hence the
sector construction — is untouched. r = 1 reproduces the short-range sign
convention (+Δ, AFM), and α → ∞ recovers `build_xxz_hamiltonian_sector`.
Open boundaries, no Kac rescaling (α > 1 keeps the energy density additive).
"""
function build_lr_xxz_hamiltonian_sector(sec::SzSector, J::Float64, Delta::Float64,
                                         alpha::Float64, h::AbstractVector{Float64})
    N = sec.N
    @assert length(h) == N "Field vector h must have length N=$N"
    Jz = [J * Delta * (-1.0)^(r + 1) / r^alpha for r in 1:N-1]
    dim = length(sec)
    H = zeros(Float64, dim, dim)
    for (k, c) in enumerate(sec.configs)
        diag = 0.0
        for i in 1:N-1, j in i+1:N
            diag += Jz[j-i] * sec.sz[k, i] * sec.sz[k, j]
        end
        for i in 1:N
            diag += h[i] * sec.sz[k, i]
        end
        H[k, k] += diag
        for i in 1:N-1
            up_i  = (c >> (i - 1)) & 1
            up_i1 = (c >> i) & 1
            if up_i != up_i1
                cflip = c ⊻ ((1 << (i - 1)) | (1 << i))
                H[sec.index[cflip], k] += J / 2
            end
        end
    end
    return Hermitian(H)
end

function build_lr_xxz_hamiltonian_sector(sec::SzSector, J::Float64, Delta::Float64,
                                         alpha::Float64)
    return build_lr_xxz_hamiltonian_sector(sec, J, Delta, alpha, zeros(Float64, sec.N))
end
