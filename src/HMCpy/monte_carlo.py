"""
monte_carlo.py -- HMC accept/reject step and momentum refresh.

Supports both pure-gauge and dynamical-fermion (pseudofermion) HMC.

For pure gauge:
    H = T(P) + S_W(U)

For Nf=2 Wilson fermions (one pseudofermion field):
    H = T(P) + S_W(U) + S_pf(phi, U)
    S_pf = phi^dag (D^dag D)^{-1} phi

One HMC trajectory:
  1. Sample momenta  P ~ exp(-T(P))
  2. (If dynamical fermions) Sample phi and compute S_pf_old
  3. Run symplectic MD trajectory  (U, P) -> (U', P')
  4. Metropolis accept/reject on delta_H = H_new - H_old
"""

from typing import Callable

import torch
from qcd_ml.qcd.dirac import dirac_wilson, dirac_wilson_clover

from .fermion import (
    apply_DDdag_inv,
    pseudofermion_action,
    wilson_clover_fermion_force,
    wilson_fermion_force,
)
from .integrator import leapfrog, omf4
from .physics import (
    gauge_force,
    hamiltonian,
    kinetic_energy,
    wilson_gauge_action,
)
from .utility import su3_generators

# ---------------------------------------------------------------------------
# Momentum sampling
# ---------------------------------------------------------------------------


def sample_momenta(U: torch.Tensor) -> torch.Tensor:
    """
    Sample conjugate momenta P ~ exp(-tr(P^2)).

    Each P_mu(x) is a traceless Hermitian 3x3 matrix drawn via
        P_mu(x) = sum_i p_mu^(i)(x) T_i
    where p_mu^(i)(x) are random normal distributed real numbers and T_i are
    the Gell-Mann matrices.

    Parameters
    ----------
    U : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Used only to determine shape, dtype, and device.

    Returns
    -------
    P : torch.Tensor, same shape as U -- traceless Hermitian
    """
    lattice_sizes = U.shape[1:5]

    ps = torch.randn(4, *lattice_sizes, 8, dtype=torch.double).to(torch.cdouble)

    return torch.einsum("...i,ikl->...kl", ps, su3_generators).to(U.device)


# ---------------------------------------------------------------------------
# Metropolis accept/reject step
# ---------------------------------------------------------------------------


def metropolis_accept(
    delta_H: torch.Tensor,
) -> bool:
    """
    Metropolis criterion:  accept with probability min(1, exp(-delta_H)).

    Parameters
    ----------
    delta_H : real scalar tensor,  H_new - H_old

    Returns
    -------
    accepted : bool
    """
    if delta_H.item() <= 0.0:
        return True
    prob = torch.exp(-delta_H.cpu()).item()
    r = torch.rand(1).item()
    return r < prob


# ---------------------------------------------------------------------------
# Full HMC update  (pure gauge or dynamical fermions)
# ---------------------------------------------------------------------------


def hmc_step(
    U: torch.Tensor,
    beta: float,
    n_steps: int,
    step_size: float,
    integrator: str = "leapfrog",
    dynamic: bool = False,
    mass_parameter: float | None = None,
    csw: float = 1,
    GMRES_kwargs: dict | None = None,
    solver: Callable | None = None,
) -> tuple[torch.Tensor, bool, float]:
    """
    One complete HMC update.

    Pure-gauge mode (default)
    -------------------------
    Omit `dynamic` or pass `False`.

    Dynamical-fermion mode (2 degenerate fermion flavors)
    -------------------------------------
    Pass `dynamic=True`.
    A pseudofermion field is sampled automatically.

    Parameters
    ----------
    U                  : current gauge field [4, Lx, Ly, Lz, Lt, Nc, Nc]
    beta               : inverse bare coupling
    n_steps            : number of steps in integration of molecular dynamics
    step_size          : molecular dynamics step size epsilon
    integrator         : "leapfrog" (default) or "omf2"
    dynamic            : Toggle whether to use dynamic fermions
    mass_parameter     : Mass parameter for Dirac operator
    csw                : csw factor for clover-term in Wilson-clover Dirac
                         operator
    GMRES_kwargs       : Keyword arguments for GMRES
    solver             : Solver function to use in apply_DDdag_inv (default: GMRES)

    Returns
    -------
    U_out    : updated gauge field
    accepted : bool
    delta_H  : float -- energy violation (use for step-size tuning)
    """

    # ---- Helpers ----
    lattice_sizes = U.shape[1:5]

    # ---- Momentum refresh ----
    P = sample_momenta(U)

    # ---- Pseudofermion sampling (if dynamical fermions) ----
    chi = None
    phi = None
    S_pf_old = 0
    use_clover = csw != 0.0 and csw is not None
    if dynamic:
        # Sample phi from Gaussian distribution
        chi = torch.randn(
            *lattice_sizes, 4, 3, dtype=torch.cdouble, device=U.device
        )

        # Create Dirac operator for current gauge field
        if use_clover:
            D_old = dirac_wilson_clover(U, mass_parameter, csw=csw)
        else:
            D_old = dirac_wilson(U, mass_parameter)
        phi = D_old(chi)

        # Compute pseudofermion action
        # S_pf = phi^dag (DD^dag)^-1 phi = chi^dag chi
        S_pf_old = pseudofermion_action(chi, chi)

    # ---- Initial Hamiltonian ----
    H_old = hamiltonian(U, P, beta) + S_pf_old

    # ---- MD trajectory ----
    force = lambda U: gauge_force(U, beta)
    if dynamic:
        if use_clover:

            def force(U):
                D = dirac_wilson_clover(U, mass_parameter, csw=csw)
                psi = apply_DDdag_inv(
                    phi, D, GMRES_kwargs=GMRES_kwargs, solver=solver
                )
                return gauge_force(U, beta) + wilson_clover_fermion_force(
                    U, psi, D, csw
                )

        else:

            def force(U):
                D = dirac_wilson(U, mass_parameter)
                psi = apply_DDdag_inv(
                    phi, D, GMRES_kwargs=GMRES_kwargs, solver=solver
                )
                return gauge_force(U, beta) + wilson_fermion_force(U, psi, D)

    integrator_kwargs = dict(
        n_steps=n_steps,
        force=force,
        step_size=step_size,
    )
    if integrator == "leapfrog":
        U_new, P_new = leapfrog(U, P, **integrator_kwargs)
    elif integrator == "omf4":
        U_new, P_new = omf4(U, P, **integrator_kwargs)
    else:
        raise ValueError(
            f"Unknown integrator '{integrator}'. Choose 'leapfrog' or 'omf4'."
        )

    # ---- Proposed Hamiltonian ----
    S_pf_new = 0
    if dynamic:
        # Create Dirac operator for new gauge field
        if use_clover:
            D_new = dirac_wilson_clover(U_new, mass_parameter, csw=csw)
        else:
            D_new = dirac_wilson(U_new, mass_parameter)

        # Solve (D_new D_new^dag) psi = phi for psi
        psi = apply_DDdag_inv(
            phi, D_new, GMRES_kwargs=GMRES_kwargs, solver=solver
        )

        # Compute pseudofermion action S_pf = phi^dag psi
        S_pf_new = pseudofermion_action(phi, psi)

    H_new = kinetic_energy(P_new) + wilson_gauge_action(U_new, beta) + S_pf_new

    dH = H_new - H_old

    # ---- Accept / reject ----
    accepted = metropolis_accept(dH)
    U_out = U_new if accepted else U

    return U_out, accepted, dH.item()
