# tests/test_coarsening_observables.jl
#
# Unit tests for the coarsening observables:
#   Analysis/coarsening.jl  — connected_matrix, staggered_correlator,
#                             domain_length, structure_factor,
#                             staggered_magnetization_sq
#   TensorOps/measurements.jl — measure_local_profile, measure_correlation_matrix
#
# Strategy: analytic checks against closed-form answers for the pure-array
# analysis layer, and consistency of the MPS producers against the independent
# ED producers on an ENTANGLED (DMRG) state, including asymmetric operators.

using LinearAlgebra
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuenchDyn

const PASS = Ref(0)
const FAIL = Ref(0)

function check(name, cond)
    if cond
        PASS[] += 1
        println("  ✓ ", name)
    else
        FAIL[] += 1
        println("  ✗ FAIL: ", name)
    end
end

approx(a, b; tol = 1e-10) = maximum(abs.(a .- b)) ≤ tol

println("="^70)
println("  TEST: coarsening observables")
println("="^70)

# ══════════════════════════════════════════════════════════════════════════
# 1. connected_matrix
# ══════════════════════════════════════════════════════════════════════════
println("\n── connected_matrix ──")
let
    C = [1.0 2.0; 3.0 4.0]
    m = [1.0, 2.0]
    Cc = connected_matrix(C, m)
    # Cc[i,j] = C[i,j] - m[i] m[j]
    expected = [1.0 - 1.0  2.0 - 2.0; 3.0 - 2.0  4.0 - 4.0]
    check("Cc = C - m mᵀ (explicit 2×2)", approx(Cc, expected))
    check("Cc = 0 when C = m mᵀ", approx(connected_matrix(m * m', m), zeros(2, 2)))
end

# ══════════════════════════════════════════════════════════════════════════
# 2. staggered_correlator  (synthetic matrix with known answer)
#    M[i,j] = (-1)^(i-j) exp(-|i-j|/ξ)  ⇒  G(r) = exp(-r/ξ)
# ══════════════════════════════════════════════════════════════════════════
println("\n── staggered_correlator ──")
let
    N = 20
    ξ = 4.0
    M = [(-1.0)^(i - j) * exp(-abs(i - j) / ξ) for i in 1:N, j in 1:N]
    rs, G = staggered_correlator(M; rmax = 8)
    Gref = [exp(-r / ξ) for r in 0:8]
    check("G(r) = exp(-r/ξ) for staggered exponential", approx(G, Gref; tol = 1e-12))
    check("rs == 0:rmax", collect(rs) == collect(0:8))
    check("G(0) == 1", isapprox(G[1], 1.0; atol = 1e-12))

    # bulk restriction: average only interior reference sites, same answer
    # (translation-invariant synthetic M ⇒ bulk window must not change G)
    _, Gb = staggered_correlator(M; rmax = 8, bulk = 6:15)
    check("bulk window leaves translation-invariant G unchanged", approx(G, Gb; tol = 1e-12))
end

# ══════════════════════════════════════════════════════════════════════════
# 3. domain_length
# ══════════════════════════════════════════════════════════════════════════
println("\n── domain_length ──")
let
    # (a) explicit crossing: G = [1, 0.5, -0.5] over rs = 0:2
    rs = 0:2
    G = [1.0, 0.5, -0.5]
    # integral: sum positive part up to first zero = (1 + 0.5)/1 = 1.5
    check("integral length (explicit)", isapprox(domain_length(rs, G; method = :integral), 1.5))
    # firstzero: crossing between r=1 (0.5) and r=2 (-0.5): 1 + 0.5/(0.5+0.5) = 1.5
    check("firstzero length (explicit interp)", isapprox(domain_length(rs, G; method = :firstzero), 1.5))

    # (b) pure exponential, no zero crossing
    ξ = 3.0
    rmax = 30
    rs2 = 0:rmax
    G2 = [exp(-r / ξ) for r in rs2]
    geom = sum(exp(-r / ξ) for r in 0:rmax)              # = Σ G / G(0), G(0)=1
    check("integral length = geometric sum (exp)", isapprox(domain_length(rs2, G2; method = :integral), geom))
    check("firstzero returns rmax when no zero", isapprox(domain_length(rs2, G2; method = :firstzero), float(rmax)))

    # (c) guard: G(0) ≤ 0 must error
    threw = false
    try
        domain_length(0:1, [-1.0, 0.5])
    catch
        threw = true
    end
    check("asserts on G(0) ≤ 0", threw)
end

# ══════════════════════════════════════════════════════════════════════════
# 4. structure_factor
# ══════════════════════════════════════════════════════════════════════════
println("\n── structure_factor ──")
let
    N = 12
    qs = range(0, 2π; length = 25)

    # (a) M = c·I  ⇒  S(q) = c for all q
    c = 0.25
    S_id = structure_factor(c * Matrix{Float64}(I, N, N), qs)
    check("S(q) = c (constant) for M = c·I", approx(S_id, fill(c, length(qs)); tol = 1e-10))

    # (b) M[i,j] = (-1)^(i-j)  ⇒  peak at q=π with S(π) = N
    Mstag = [(-1.0)^(i - j) for i in 1:N, j in 1:N]
    Sst = structure_factor(Mstag, qs)
    ipk = argmax(Sst)
    check("staggered M peaks at q = π", isapprox(qs[ipk], float(π); atol = 1e-9))
    check("S(π) = N for all-staggered M", isapprox(structure_factor(Mstag, [float(π)])[1], float(N); atol = 1e-9))
    check("S(q) is real (finite)", all(isfinite, Sst))
end

# ══════════════════════════════════════════════════════════════════════════
# 5. staggered_magnetization_sq  (and its S(π)/N identity)
# ══════════════════════════════════════════════════════════════════════════
println("\n── staggered_magnetization_sq ──")
let
    N = 12
    Mstag = [(-1.0)^(i - j) for i in 1:N, j in 1:N]
    check("⟨m_s²⟩ = 1 for all-staggered M", isapprox(staggered_magnetization_sq(Mstag), 1.0; atol = 1e-12))
    check("⟨m_s²⟩ = 1/N for M = I", isapprox(staggered_magnetization_sq(Matrix{Float64}(I, N, N)), 1 / N; atol = 1e-12))
    # identity ⟨m_s²⟩ = S(π)/N
    M = [(-1.0)^(i - j) * exp(-abs(i - j) / 3.0) for i in 1:N, j in 1:N]
    check("⟨m_s²⟩ == S(π)/N", isapprox(staggered_magnetization_sq(M), structure_factor(M, [float(π)])[1] / N; atol = 1e-12))
end

# ══════════════════════════════════════════════════════════════════════════
# 6. MPS producers vs ED producers
# ══════════════════════════════════════════════════════════════════════════
println("\n── MPS measure_* vs ED (product state, exact) ──")
let
    N = 10
    d = 2
    ops = spin_ops(d)
    Sz = Matrix{ComplexF64}(ops[:Z])
    SzR = real(Sz)

    sites = [SpinSite(0.5; T = ComplexF64) for _ in 1:N]
    labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
    mps = product_state(sites, labels)
    psi = neel_state(N; d = d)

    check("measure_local_profile == ed_local_profile (Neel)",
          approx(measure_local_profile(mps, Sz), ed_local_profile(psi, SzR, N)))
    check("measure_correlation_matrix == ed_correlation_matrix (Neel)",
          approx(measure_correlation_matrix(mps, Sz, Sz), ed_correlation_matrix(psi, SzR, SzR, N)))
end

println("\n── MPS measure_* vs ED on an ENTANGLED DMRG state ──")
let
    N = 8
    d = 2
    J = 1.0
    Δ = 0.8
    h = [0.5 * (-1.0)^i for i in 1:N]            # staggered field ⇒ non-uniform ⟨Sz⟩, breaks parity
    ops = spin_ops(d)
    Sz = Matrix{ComplexF64}(ops[:Z]);  SzR = real(Sz)
    Sx = Matrix{ComplexF64}(ops[:X]);  SxR = real(Sx)
    Sp = Matrix{ComplexF64}(ops[:Sp]); SpR = real(Sp)
    Sm = Matrix{ComplexF64}(ops[:Sm]); SmR = real(Sm)

    mpo = build_xxz_mpo(N, J, Δ, h; d = d)
    s_sites = [SpinSite(0.5; T = ComplexF64) for _ in 1:N]
    s_labels = [isodd(i) ? (:Z, 1) : (:Z, 2) for i in 1:N]
    state = MPSState(product_state(s_sites, s_labels), mpo; center = 1)
    solver = LanczosSolver(4, 14)
    opts = DMRGOptions(16, 1e-12, d)             # χ=16 = full Hilbert at N=8 ⇒ exact
    for sweep in 1:20
        dmrg_sweep(state, solver, opts, isodd(sweep) ? :right : :left)
    end
    mps = state.mps
    psi = mps_to_vector(mps; d = d)              # same state as a dense vector

    check("MPS state is normalized", isapprox(real(measure_norm(mps)), 1.0; atol = 1e-10))

    # one-point profile (non-uniform, parity broken)
    m_mps = measure_local_profile(mps, Sz)
    m_ed  = ed_local_profile(psi, SzR, N)
    check("local profile MPS == ED (entangled)", approx(m_mps, m_ed; tol = 1e-9))
    check("profile is genuinely non-uniform (test non-vacuous)", maximum(abs.(m_mps)) > 0.05)

    # symmetric two-point ⟨Sz_i Sz_j⟩
    Czz_mps = measure_correlation_matrix(mps, Sz, Sz)
    Czz_ed  = ed_correlation_matrix(psi, SzR, SzR, N)
    check("⟨Sz Sz⟩ matrix MPS == ED (entangled)", approx(Czz_mps, Czz_ed; tol = 1e-9))
    check("⟨Sz Sz⟩ has nonzero off-diagonal (non-vacuous)", maximum(abs.(Czz_mps - Diagonal(Czz_mps))) > 1e-3)

    # asymmetric operators ⟨S+_i S-_j⟩ — exercises BOTH triangles with op1≠op2
    Cpm_mps = measure_correlation_matrix(mps, Sp, Sm)
    Cpm_ed  = ed_correlation_matrix(psi, SpR, SmR, N)
    check("⟨S+ S-⟩ matrix MPS == ED (asymmetric ops)", approx(Cpm_mps, Cpm_ed; tol = 1e-9))
    check("⟨S+ S-⟩ off-diagonal nonzero (non-vacuous)", maximum(abs.(Cpm_mps - Diagonal(Cpm_mps))) > 1e-3)

    # direct proof the two triangles carry the correct operator order
    Czx = measure_correlation_matrix(mps, Sz, Sx)
    i, j = 2, 6
    check("C[i,j] == ⟨Sz_i Sx_j⟩ (upper triangle order)",
          isapprox(Czx[i, j], real(measure_correlation(mps, Sz, i, Sx, j)); atol = 1e-12))
    check("C[j,i] == ⟨Sx_i Sz_j⟩ (lower triangle order)",
          isapprox(Czx[j, i], real(measure_correlation(mps, Sx, i, Sz, j)); atol = 1e-12))
    # ⟨Sz Sx⟩ is U(1)-forbidden (changes total Sz) ⇒ structurally 0; both paths
    # return roundoff ~1e-9, so compare with a tolerance above that floor.
    check("⟨Sz Sx⟩ ≈ 0 (U(1) selection rule, MPS)", maximum(abs.(Czx)) < 1e-7)
    check("⟨Sz Sx⟩ matrix MPS == ED (near-zero)", approx(Czx, ed_correlation_matrix(psi, SzR, SxR, N); tol = 1e-7))
    check("diagonal C[i,i] == ⟨(Sz·Sz)_i⟩ = 1/4",
          isapprox(Czz_mps[3, 3], 0.25; atol = 1e-10))
end

# ══════════════════════════════════════════════════════════════════════════
# 7. End-to-end: full pipeline on an ED quench state (physical sanity)
# ══════════════════════════════════════════════════════════════════════════
println("\n── end-to-end on an ED quench state ──")
let
    N = 10
    d = 2
    SzR = real(Matrix(spin_ops(d)[:Z]))
    _, psi0 = ground_state(build_xxz_hamiltonian(N, 1.0, 0.8; d = d))
    eig_f = diagonalize(build_xxz_hamiltonian(N, 1.0, 3.0; d = d))
    psit = ed_time_evolve(eig_f, psi0, 2.0)

    m = ed_local_profile(psit, SzR, N)
    C = ed_correlation_matrix(psit, SzR, SzR, N)
    Cc = connected_matrix(C, m)
    rs, G = staggered_correlator(Cc; rmax = N ÷ 2)
    L = domain_length(rs, G; method = :integral)

    check("h=0 quench: ⟨Sz_i⟩ ≡ 0 (parity)", maximum(abs.(m)) < 1e-10)
    check("connected == full ⟨m_s²⟩ when m≡0",
          isapprox(staggered_magnetization_sq(Cc), staggered_magnetization_sq(C); atol = 1e-12))
    check("G(0) > 0 and G decays (G(1) < G(0))", G[1] > 0 && G[2] < G[1])
    check("domain length finite and positive", isfinite(L) && L > 0)
    check("S(q) peaks at q=π", isapprox(range(0, 2π; length = 41)[argmax(structure_factor(Cc, range(0, 2π; length = 41)))], float(π); atol = 0.2))
end

# ══════════════════════════════════════════════════════════════════════════
println("\n" * "="^70)
println("  RESULTS:  $(PASS[]) passed,  $(FAIL[]) failed")
println("="^70)
FAIL[] == 0 || error("$(FAIL[]) test(s) failed")
