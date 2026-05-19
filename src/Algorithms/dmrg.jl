# Algorithms/dmrg.jl

"""
    dmrg_sweep_one_site(state, solver, options, direction)

Single-site DMRG sweep. Bond-preserving (no SVD/truncation), so `chi_max` and
`cutoff` in `options` are ignored. Caller is responsible for ensuring the
initial MPS already has the working bond dimension — single-site DMRG cannot
grow χ from a product state.

Right sweep solves at sites 1..N-1 (orthog center at `state.center` must be 1
at entry; ends at N). Left sweep solves at sites N..2 (entry center N; ends 1).
"""
function dmrg_sweep_one_site(state::MPSState, solver::LanczosSolver, options::DMRGOptions, direction::Symbol)
    N = length(state.mps.tensors)
    last_energy = 0.0
    max_chi = 0

    if direction == :right
        for site in 1:N-1
            chi_l, d, chi_r = size(state.mps.tensors[site])

            left_env  = site == 1 ? state.environment.tensors[N+1] : state.environment.tensors[site-1]
            right_env = state.environment.tensors[site+1]
            Heff = OneSiteEffectiveHamiltonian(left_env, state.mpo.tensors[site], right_env)

            evec, eval = _solve(solver, Heff, vec(state.mps.tensors[site]))
            last_energy = real(eval)

            F = qr(reshape(evec, (chi_l*d, chi_r)))
            new_chi = size(F.R, 1)
            Q_full = Matrix(F.Q)
            Q = size(Q_full, 2) > new_chi ? Q_full[:, 1:new_chi] : Q_full
            R = Matrix(F.R)

            state.mps.tensors[site] = reshape(Q, (chi_l, d, new_chi))
            _update_left_environment(state, site)

            @tensoropt new_next[-1,-2,-3] := R[-1,4] * state.mps.tensors[site+1][4,-2,-3]
            state.mps.tensors[site+1] = new_next
            state.environment.tensors[site+1] = nothing
            state.center = site + 1
            max_chi = max(max_chi, new_chi)
        end
    else
        for site in N:-1:2
            chi_l, d, chi_r = size(state.mps.tensors[site])

            left_env  = state.environment.tensors[site-1]
            right_env = state.environment.tensors[site+1]
            Heff = OneSiteEffectiveHamiltonian(left_env, state.mpo.tensors[site], right_env)

            evec, eval = _solve(solver, Heff, vec(state.mps.tensors[site]))
            last_energy = real(eval)

            F = lq(reshape(evec, (chi_l, d*chi_r)))
            new_chi = size(F.L, 2)
            L = Matrix(F.L)
            Q_full = Matrix(F.Q)
            Q = size(Q_full, 1) > new_chi ? Q_full[1:new_chi, :] : Q_full

            state.mps.tensors[site] = reshape(Q, (new_chi, d, chi_r))
            _update_right_environment(state, site)

            @tensoropt new_prev[-1,-2,-3] := state.mps.tensors[site-1][-1,-2,4] * L[4,-3]
            state.mps.tensors[site-1] = new_prev
            state.environment.tensors[site-1] = nothing
            state.center = site - 1
            max_chi = max(max_chi, new_chi)
        end
    end

    return (E=last_energy, max_chi=max_chi)
end

function dmrg_sweep(state::MPSState,solver::LanczosSolver,options::DMRGOptions,direction::Symbol)
    N = length(state.mps.tensors)
    last_energy = 0.0
    max_trunc   = 0.0
    total_trunc = 0.0
    max_chi     = 0
    if direction == :right sites = 1:N-1 else sites=N:-1:2 end
    if direction == :right
        for site in sites
            @tensoropt theta[-1,-2,-3,-4] := state.mps.tensors[site][-1,-2,5]*state.mps.tensors[site+1][5,-3,-4]
            chi_l,d1,d2,chi_r = size(theta)

            left_env = site == 1 ? state.environment.tensors[N+1] : state.environment.tensors[site-1]
            right_env = state.environment.tensors[site+2]
            Heff = TwoSiteEffectiveHamiltonian(left_env,state.mpo.tensors[site],state.mpo.tensors[site+1],right_env)

            evec,eval = _solve(solver,Heff,vec(theta))
            last_energy = real(eval)
            U,S,V,disc = _svd_truncate(reshape(evec,(chi_l*d1,d2*chi_r)),options.chi_max,options.cutoff)
            state.mps.tensors[site] = reshape(U,(chi_l,d1,:))
            @tensoropt state.mps.tensors[site+1][-1,-2,-3] := Diagonal(S)[-1,4]*reshape(V,(:,d2,chi_r))[4,-2,-3]
            state.center = site+1

            max_trunc    = max(max_trunc, disc)
            total_trunc += disc
            max_chi      = max(max_chi, length(S))

            _update_left_environment(state,site)
        end
    else
        for site in sites
            @tensoropt theta[-1,-2,-3,-4] := state.mps.tensors[site-1][-1,-2,5]*state.mps.tensors[site][5,-3,-4]
            chi_l,d1,d2,chi_r = size(theta)

            left_env = site == 2 ? state.environment.tensors[N+1] : state.environment.tensors[site-2]
            right_env = state.environment.tensors[site+1]
            Heff = TwoSiteEffectiveHamiltonian(left_env,state.mpo.tensors[site-1],state.mpo.tensors[site],right_env)

            evec,eval = _solve(solver,Heff,vec(theta))
            last_energy = real(eval)
            U,S,V,disc = _svd_truncate(reshape(evec,(chi_l*d1,d2*chi_r)),options.chi_max,options.cutoff)
            state.mps.tensors[site] = reshape(V,(:,d2,chi_r))
            @tensoropt state.mps.tensors[site-1][-1,-2,-3] := reshape(U,(chi_l,d1,:))[-1,-2,4]*Diagonal(S)[4,-3]
            state.center = site-1

            max_trunc    = max(max_trunc, disc)
            total_trunc += disc
            max_chi      = max(max_chi, length(S))

            _update_right_environment(state,site)
        end
    end
    return (E=last_energy, max_trunc=max_trunc, total_trunc=total_trunc, max_chi=max_chi)
end
