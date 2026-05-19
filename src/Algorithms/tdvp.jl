"""
This file consists of functions for one and two site TDVP (for both MPS and MPDO using LPTN) algorithms.
The two seminal papers are:
1) https://doi.org/10.1103/PhysRevLett.107.070601 : for more mathematical background behind TDVP
2) https://doi.org/10.1103/PhysRevB.94.165116 : for more visual approach to TDVP

But the best explanation and a ready to use recipe for TDVP and several other timedependent MPS methods
are found in https://doi.org/10.1016/j.aop.2019.167998
"""

# Algorithms/tdvp.jl

"""
    tdvp_sweep_one_site(state, solver, options, direction)

Single-site TDVP sweep with the standard symmetric forward/back-evolve recipe:
each site is forward-evolved by `+dt/2` under a one-site H_eff, then the bond
matrix from the QR (or LQ on the way back) is back-evolved by `-dt/2` under
the zero-site H_eff before being absorbed into the next site.

Bond-preserving (no SVD/truncation), so `chi_max` and `cutoff` in `options`
are ignored. Caller must ensure the initial MPS has the working bond
dimension — single-site TDVP cannot grow χ from a product state.

Right sweep covers sites 1..N (entry center 1; ends N). Left sweep covers
N..1 (entry center N; ends 1). Boundary sites (i=N in right, i=1 in left)
are evolved but not QR/LQ-decomposed.
"""
function tdvp_sweep_one_site(state::MPSState, solver::KrylovExponential, options::TDVPOptions, direction::Symbol)
    N = length(state.mps.tensors)
    max_chi = 0

    if direction == :right
        for site in 1:N
            chi_l, d, chi_r = size(state.mps.tensors[site])

            left_env  = site == 1 ? state.environment.tensors[N+1] : state.environment.tensors[site-1]
            right_env = state.environment.tensors[site+1]
            Heff = OneSiteEffectiveHamiltonian(left_env, state.mpo.tensors[site], right_env)

            M_evolved = _evolve(solver, Heff, vec(state.mps.tensors[site]), options.dt/2)

            if site == N
                state.mps.tensors[site] = reshape(M_evolved, (chi_l, d, chi_r))
                max_chi = max(max_chi, chi_r)
            else
                F = qr(reshape(M_evolved, (chi_l*d, chi_r)))
                new_chi = size(F.R, 1)
                Q_full = Matrix(F.Q)
                Q = size(Q_full, 2) > new_chi ? Q_full[:, 1:new_chi] : Q_full
                R = Matrix(F.R)

                state.mps.tensors[site] = reshape(Q, (chi_l, d, new_chi))
                _update_left_environment(state, site)

                Heff_zero = ZeroSiteEffectiveHamiltonian(state.environment.tensors[site],
                                                        state.environment.tensors[site+1])
                R_evolved = _evolve(solver, Heff_zero, vec(R), -options.dt/2)
                R_new = reshape(R_evolved, size(R))

                @tensoropt new_next[-1,-2,-3] := R_new[-1,4] * state.mps.tensors[site+1][4,-2,-3]
                state.mps.tensors[site+1] = new_next
                state.environment.tensors[site+1] = nothing
                state.center = site + 1
                max_chi = max(max_chi, new_chi)
            end
        end
    else
        for site in N:-1:1
            chi_l, d, chi_r = size(state.mps.tensors[site])

            left_env  = site == 1 ? state.environment.tensors[N+1] : state.environment.tensors[site-1]
            right_env = state.environment.tensors[site+1]
            Heff = OneSiteEffectiveHamiltonian(left_env, state.mpo.tensors[site], right_env)

            M_evolved = _evolve(solver, Heff, vec(state.mps.tensors[site]), options.dt/2)

            if site == 1
                state.mps.tensors[site] = reshape(M_evolved, (chi_l, d, chi_r))
                max_chi = max(max_chi, chi_l)
            else
                F = lq(reshape(M_evolved, (chi_l, d*chi_r)))
                new_chi = size(F.L, 2)
                L = Matrix(F.L)
                Q_full = Matrix(F.Q)
                Q = size(Q_full, 1) > new_chi ? Q_full[1:new_chi, :] : Q_full

                state.mps.tensors[site] = reshape(Q, (new_chi, d, chi_r))
                _update_right_environment(state, site)

                Heff_zero = ZeroSiteEffectiveHamiltonian(state.environment.tensors[site-1],
                                                        state.environment.tensors[site])
                L_evolved = _evolve(solver, Heff_zero, vec(L), -options.dt/2)
                L_new = reshape(L_evolved, size(L))

                @tensoropt new_prev[-1,-2,-3] := state.mps.tensors[site-1][-1,-2,4] * L_new[4,-3]
                state.mps.tensors[site-1] = new_prev
                state.environment.tensors[site-1] = nothing
                state.center = site - 1
                max_chi = max(max_chi, new_chi)
            end
        end
    end

    return (max_chi=max_chi,)
end

function tdvp_sweep(state::MPSState,solver::KrylovExponential,options::TDVPOptions,direction::Symbol)
    N = length(state.mps.tensors)
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

            theta = _evolve(solver,Heff,vec(theta),(options.dt)/2)
            U,S,V,disc = _svd_truncate(reshape(theta,(chi_l*d1,d2*chi_r)),options.chi_max,options.cutoff)
            state.mps.tensors[site] = reshape(U,(chi_l,d1,:))
            @tensoropt state.mps.tensors[site+1][-1,-2,-3] := Diagonal(S)[-1,4]*reshape(V,(:,d2,chi_r))[4,-2,-3]
            state.center = site+1

            max_trunc    = max(max_trunc, disc)
            total_trunc += disc
            max_chi      = max(max_chi, length(S))

            if site != N-1
                _update_left_environment(state,site)

                left_env = state.environment.tensors[site]
                right_env = state.environment.tensors[site+2]
                Heff = OneSiteEffectiveHamiltonian(left_env,state.mpo.tensors[site+1],right_env)

                theta = _evolve(solver,Heff,vec(state.mps.tensors[site+1]),-(options.dt)/2)
                state.mps.tensors[site+1] = reshape(theta,(:,d2,chi_r))
            end
        end
    else
        for site in sites
            @tensoropt theta[-1,-2,-3,-4] := state.mps.tensors[site-1][-1,-2,5]*state.mps.tensors[site][5,-3,-4]
            chi_l,d1,d2,chi_r = size(theta)

            left_env = site == 2 ? state.environment.tensors[N+1] : state.environment.tensors[site-2]
            right_env = state.environment.tensors[site+1]
            Heff = TwoSiteEffectiveHamiltonian(left_env,state.mpo.tensors[site-1],state.mpo.tensors[site],right_env)

            theta = _evolve(solver,Heff,vec(theta),(options.dt)/2)
            U,S,V,disc = _svd_truncate(reshape(theta,(chi_l*d1,d2*chi_r)),options.chi_max,options.cutoff)
            state.mps.tensors[site] = reshape(V,(:,d2,chi_r))
            @tensoropt state.mps.tensors[site-1][-1,-2,-3] := reshape(U,(chi_l,d1,:))[-1,-2,4]*Diagonal(S)[4,-3]
            state.center = site-1

            max_trunc    = max(max_trunc, disc)
            total_trunc += disc
            max_chi      = max(max_chi, length(S))

            if site != 2
                _update_right_environment(state,site)

                left_env = state.environment.tensors[site-2]
                right_env = state.environment.tensors[site]
                Heff = OneSiteEffectiveHamiltonian(left_env,state.mpo.tensors[site-1],right_env)

                theta = _evolve(solver,Heff,vec(state.mps.tensors[site-1]),-(options.dt)/2)
                state.mps.tensors[site-1] = reshape(theta,(chi_l,d1,:))
            end
        end
    end
    return (max_trunc=max_trunc, total_trunc=total_trunc, max_chi=max_chi)
end
