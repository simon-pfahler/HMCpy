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

Conventions match HMCpy / qcd_ml: U shape [4, Lx, Ly, Lz, Lt, Nc, Nc].
"""

from typing import Callable

import torch
from qcd_ml.qcd.dirac import dirac_wilson_clover
from qcd_ml.util.solver import GMRES

from .fermion import gell_mann_matrices
from .integrator import leapfrog
from .physics import (
    gauge_force,
    hamiltonian,
    kinetic_energy,
    wilson_gauge_action,
)

# ---------------------------------------------------------------------------
# Momentum sampling
# ---------------------------------------------------------------------------


def sample_momenta(U: torch.Tensor) -> torch.Tensor:
    """
    Sample conjugate momenta P ~ exp(-tr(P^2)).

    Each P_mu(x) is a traceless anti-Hermitian 3x3 matrix drawn via
        P_mu(x) = sum_i p_mu^(i)(x) T_i
    where p_mu^(i)(x) are random normal distributed real numbers and T_i are
    the Gell-Mann matrices.

    Parameters
    ----------
    U : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Used only to determine shape, dtype, and device.

    Returns
    -------
    P : torch.Tensor, same shape as U -- traceless anti-Hermitian
    """
    lattice_sizes = U.shape[1:5]

    ps = torch.randn(4, *lattice_sizes, 8, dtype=torch.double).to(torch.cdouble)

    return torch.einsum("...i,ikl->...kl", ps, gell_mann_matrices).to(U.device)


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
    phi = None
    S_pf_old = 0
    if dynamic:
        chi = torch.randn(
            *lattice_sizes, 4, 3, dtype=torch.cdouble, device=U.device
        )

        D_old = dirac_wilson_clover(U, mass_parameter, csw)
        phi = D_old(chi)

        S_pf_old = torch.einsum(chi.conj(), chi).real.sum()

    # ---- Initial Hamiltonian ----
    H_old = kinetic_energy(P) + wilson_gauge_action(U, beta) + S_pf_old

    # ---- MD trajectory ----
    force = lambda U: gauge_force(U, beta)
    integrator_kwargs = dict(
        n_steps=n_steps,
        force=force,
        step_size=step_size,
    )
    if integrator == "leapfrog":
        U_new, P_new = leapfrog(U, P, **integrator_kwargs)
    else:
        raise ValueError(
            f"Unknown integrator '{integrator}'. Choose 'leapfrog'."
        )

    # ---- Proposed Hamiltonian ----
    S_pf_new = 0
    if dynamic:
        D_new = dirac_wilson_clover(U_new, mass_parameter, csw)
        varphi = GMRES(D_new, phi, phi, **GMRES_kwargs)
        S_pf_new = (varphi.conj() * varphi).real.sum()

    H_new = kinetic_energy(P_new) + wilson_gauge_action(U_new, beta) + S_pf_new

    dH = H_new - H_old

    # ---- Accept / reject ----
    accepted = metropolis_accept(dH)
    U_out = U_new if accepted else U

    return U_out, accepted, dH.item()
