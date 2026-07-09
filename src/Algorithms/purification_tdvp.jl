# Algorithms/purification_tdvp.jl
#
# Imaginary-time cooling of an ancilla purification via the existing 2-site
# TDVP sweep (Approach A). Cools a β=0 purification (bond dim 1) under a
# purified MPO H⊗I, growing χ via the 2-site SVD, until a target inverse
# temperature β or a target energy ⟨H⟩ is reached.
#
# Time bookkeeping: one full round-trip (a :right sweep followed by a :left
# sweep) advances imaginary time by `opts.dt` (the standard 2nd-order 2-site
# TDVP step). Tracing the ancilla makes ρ ∝ e^{-2τH}, so β = 2τ. The exact
# τ↔β factor is confirmed against ED in tests/test_purification.jl.
#
# All observables are ratios ⟨·⟩ = measure_*/measure_norm, so the state's
# overall normalization is irrelevant; the orthogonality-center tensor is
# rescaled each round-trip only to keep the numbers in range (e^{-2τH} would
# otherwise over/underflow).

"""
    cool_purification!(state, solver, opts; target_beta=nothing,
                       target_energy=nothing, max_roundtrips=2000, record=false)

Imaginary-time cool the purified `state` (an `MPSState` whose MPO is a purified
H⊗I and whose solver has `evol_type="imaginary"`). Stops at the first round-trip
where β ≥ `target_beta` (if given) or ⟨H⟩ ≤ `target_energy` (if given); one of
the two must be supplied. Mutates `state`.

Returns a NamedTuple `(β, energy, max_chi, βs, Es)` where `βs, Es` are the
per-round-trip traces (populated only if `record=true`), and `energy` is the
normalized ⟨H⟩ = Tr(ρH).
"""
function cool_purification!(state::MPSState, solver::KrylovExponential,
                           opts::TDVPOptions;
                           target_beta::Union{Nothing,Real} = nothing,
                           target_energy::Union{Nothing,Real} = nothing,
                           max_roundtrips::Int = 2000, record::Bool = false)
    @assert (target_beta !== nothing) || (target_energy !== nothing) "supply target_beta or target_energy"
    @assert solver.evol_type == "imaginary" "solver must be evol_type=\"imaginary\""

    τ = 0.0
    max_chi = 0
    E = measure_energy(state) / measure_norm(state.mps)
    βs = Float64[]; Es = Float64[]
    record && (push!(βs, 0.0); push!(Es, E))

    for _ in 1:max_roundtrips
        tdvp_sweep(state, solver, opts, :right)
        res = tdvp_sweep(state, solver, opts, :left)
        max_chi = max(max_chi, res.max_chi)
        τ += opts.dt

        # rescale the orthogonality center to unit norm (range control only)
        nrm = measure_norm(state.mps)
        state.mps.tensors[state.center] ./= sqrt(nrm)

        β = 2τ
        E = measure_energy(state) / measure_norm(state.mps)
        record && (push!(βs, β); push!(Es, E))

        if target_beta !== nothing && β ≥ target_beta - 1e-12
            return (β = β, energy = E, max_chi = max_chi, βs = βs, Es = Es)
        end
        if target_energy !== nothing && E ≤ target_energy
            return (β = β, energy = E, max_chi = max_chi, βs = βs, Es = Es)
        end
    end
    return (β = 2τ, energy = E, max_chi = max_chi, βs = βs, Es = Es)
end

"""
    thermal_sz_from_purification(state; d=2)

Thermal one- and two-point S^z data from a cooled purified MPS `state`:
`m[j] = Tr(ρ S^z_j)` and `C[j,k] = Tr(ρ S^z_j S^z_k)`, normalized by Tr ρ.
Uses `Sz ⊗ I_anc` on the merged local index. Returns `(m, C)` ready for the
`coarsening.jl` observables — the MPS/purification analog of
`thermal_sz_correlations`.
"""
function thermal_sz_from_purification(state::MPSState; d::Int = 2)
    mps = state.mps
    N = length(mps.tensors)
    T = eltype(mps.tensors[1])
    Sz = embed_physical(Matrix{T}(spin_ops(d)[:Z]); d = d)
    Z = measure_norm(mps)
    m = [real(measure_local_observable(mps, Sz, i)) / Z for i in 1:N]
    C = zeros(Float64, N, N)
    SzSz = Sz * Sz
    for i in 1:N
        C[i, i] = real(measure_local_observable(mps, SzSz, i)) / Z
        for j in i+1:N
            c = real(measure_correlation(mps, Sz, i, Sz, j)) / Z
            C[i, j] = c
            C[j, i] = c
        end
    end
    return m, C
end
