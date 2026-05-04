"""
This file consists of functions for one and two site TDVP (for both MPS and MPDO using LPTN) algorithms.
The two seminal papers are:
1) https://doi.org/10.1103/PhysRevLett.107.070601 : for more mathematical background behind TDVP
2) https://doi.org/10.1103/PhysRevB.94.165116 : for more visual approach to TDVP

But the best explanation and a ready to use recipe for TDVP and several other timedependent MPS methods
are found in https://doi.org/10.1016/j.aop.2019.167998
"""

# Algorithms/tdvp.jl

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
