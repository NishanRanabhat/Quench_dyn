# tests/diagnostic_entanglement_degeneracy.jl
#
# Test the mechanism behind the ~10^4 gap in 2-site-TDVP parity-odd integration
# error between the exact eigenvector (ψ_B) and the DMRG ground state (ψ_A),
# two states that agree to ~1e-7.
#
# Hypothesis: ψ_B is an exact, fully symmetric eigenstate, so its entanglement
# (Schmidt) spectrum is DEGENERATE on every bond. Inside a degenerate Schmidt
# subspace the SVD basis is arbitrary, and the SVD sign/basis convention is not
# spin-flip (parity) equivariant → TDVP injects a coherent parity-odd error.
# DMRG converges only to ~1e-7, which SPLITS the degeneracies, removing the
# ambiguity and the injection. A 1e-7 perturbation → 10^4 error change because
# it lifts an exact degeneracy (a discontinuous change), not a smooth one.
#
# Prediction: ψ_B has many near-exact Schmidt degeneracies (relative gaps
# ~machine eps); ψ_A has the same multiplets split by ~1e-7. Also report
# ‖ψ_A − ψ_B‖ to nail the "10^4 from 1e-7" claim.

using LinearAlgebra
using Printf

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

const N = 12; const J = 1.0; const d = 2; const Δi = 0.5
const h = zeros(N); const χ = 64
const n_sweeps_dmrg = 40; const cutoff_dmrg = 1e-12

function run_dmrg(mps_init, mpo, χ, n_sweeps, cutoff)
    state = MPSState(mps_init, mpo; center=1)
    solver = LanczosSolver(4, 14); opts = DMRGOptions(χ, cutoff, d)
    local res
    for sweep in 1:n_sweeps
        res = dmrg_sweep(state, solver, opts, isodd(sweep) ? :right : :left)
    end
    return state.mps
end

"""Schmidt spectra of a full state vector on every bond i|i+1 (site 1 = MSB)."""
function bond_spectra_from_vector(psi, N; d=2)
    specs = Vector{Vector{Float64}}(undef, N-1)
    psi_mat = reshape(Vector{ComplexF64}(psi), (1, d^N))
    chi_l = 1
    for i in 1:N-1
        psi_mat = reshape(psi_mat, (chi_l*d, d^(N-i)))
        F = svd(psi_mat)
        specs[i] = copy(F.S)
        psi_mat = Diagonal(F.S) * F.Vt
        chi_l = length(F.S)
    end
    return specs
end

"""Count near-degenerate consecutive Schmidt pairs and the smallest relative gap,
considering only singular values that carry real weight (> tol of the largest)."""
function degeneracy_summary(S; weight_tol=1e-6, deg_tol=1e-6)
    Ssig = sort(S[S .> weight_tol*maximum(S)]; rev=true)
    ndeg = 0; mingap = Inf
    for k in 1:length(Ssig)-1
        relgap = abs(Ssig[k]-Ssig[k+1]) / Ssig[k]
        mingap = min(mingap, relgap)
        if relgap < deg_tol; ndeg += 1; end
    end
    return length(Ssig), ndeg, mingap
end

# ─── build the two states ────────────────────────────────────────────────────
mpo_i = build_xxz_mpo(N, J, Δi, h; d=d)
H_i   = build_xxz_hamiltonian(N, J, Δi, h; d=d)
eig_i = diagonalize(H_i)
ψB = Vector{ComplexF64}(eig_i.vectors[:, 1]); ψB ./= norm(ψB)

sites = [SpinSite(0.5; T=ComplexF64) for _ in 1:N]
labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
mpsA = run_dmrg(product_state(sites, labels), mpo_i, χ, n_sweeps_dmrg, cutoff_dmrg)
ψA = mps_to_vector(mpsA; d=d); ψA ./= norm(ψA)

# phase-align A to B before distance
ψA .*= conj(dot(ψB, ψA)) / abs(dot(ψB, ψA))
@printf "\n‖ψ_A − ψ_B‖ = %.3e    |⟨ψ_A|ψ_B⟩| = %.12f\n\n" norm(ψA-ψB) abs(dot(ψB,ψA))

specA = bond_spectra_from_vector(ψA, N; d=d)
specB = bond_spectra_from_vector(ψB, N; d=d)

println("Per-bond Schmidt degeneracy (weight>1e-6·λmax, 'deg' = rel.gap<1e-6):")
@printf "  %-6s | %-22s | %-22s\n" "bond" "EXACT ψ_B" "DMRG ψ_A"
@printf "  %-6s | %-7s %-6s %-8s | %-7s %-6s %-8s\n" "" "n_sig" "n_deg" "min_gap" "n_sig" "n_deg" "min_gap"
for i in 1:N-1
    nB, dB, gB = degeneracy_summary(specB[i])
    nA, dA, gA = degeneracy_summary(specA[i])
    @printf "  %-6d | %-7d %-6d %-8.1e | %-7d %-6d %-8.1e\n" i nB dB gB nA dA gA
end

# central bond: show the actual top singular values side by side
println("\nCentral bond (6) — top Schmidt values, exact vs DMRG:")
SB = sort(specB[6]; rev=true); SA = sort(specA[6]; rev=true)
@printf "  %-4s %-16s %-16s %-10s\n" "k" "λ_B (exact)" "λ_A (DMRG)" "|ΔB rel.gap|"
for k in 1:min(12, length(SB))
    gap = k < length(SB) ? abs(SB[k]-SB[k+1])/SB[k] : NaN
    @printf "  %-4d %-16.10e %-16.10e %-10.1e\n" k SB[k] SA[k] gap
end
