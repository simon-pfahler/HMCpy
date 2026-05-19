"""
integrator.py -- Symplectic integrators for HMC.
"""

from typing import Callable

import torch

from .utility import exp_update_U

# ---------------------------------------------------------------------------
# Leapfrog (Störmer-Verlet) integrator
# ---------------------------------------------------------------------------


def leapfrog(
    U: torch.Tensor,
    P: torch.Tensor,
    n_steps: int,
    force: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
    U_update: Callable[
        [torch.Tensor, torch.Tensor, float], torch.Tensor
    ] = exp_update_U,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Leapfrog integrator for HMC molecular dynamics.

    Parameters
    ----------
    U, P       : initial phase-space point
    n_steps    : number of leapfrog steps
    force      : Function to obtain the force for a given field U
    step_size  : MD step size epsilon
    U_update   : Function to update the field U given P

    Returns
    -------
    U_new, P_new
    """

    P = P + (step_size / 2) * force(U)
    for _ in range(n_steps - 1):
        U = U_update(U, P, step_size)
        P = P + step_size * force(U)
    U = U_update(U, P, step_size)
    P = P + (step_size / 2) * force(U)

    return U, P


# ---------------------------------------------------------------------------
# OMF4 (Omelyan, Mryglod, Folk) integrator
# ---------------------------------------------------------------------------


def omf4(
    U: torch.Tensor,
    P: torch.Tensor,
    n_steps: int,
    force: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
    U_update: Callable[
        [torch.Tensor, torch.Tensor, float], torch.Tensor
    ] = exp_update_U,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    OMF4 (Omelyan-Mryglod-Folk) 4th-order symplectic integrator.

    11-stage, non-gradient (no force-gradient terms) decomposition algorithm
    from https://doi.org/10.1016/S0010-4655(02)00754-3, p. 292, Variant 8.

    This is an extended Forest-Ruth-like integrator with optimized coefficients
    to minimize 5th-order truncation errors. It uses 5 force evaluations per step
    and achieves 4th-order accuracy.

    Parameters
    ----------
    U, P       : initial phase-space point (gauge fields and momenta)
    n_steps    : number of integration steps
    force      : Function to obtain the force for a given field U
    step_size  : MD step size epsilon
    U_update   : Function to update the field U given P and step size

    Returns
    -------
    U_new, P_new : phase-space point after n_steps of OMF4 integration
    """

    # Coefficients from Omelyan et al., Eq. (71), Variant 8 (non-gradient, order 4)
    rho = 0.2539785108410595
    theta = -0.03230286765269967
    vartheta = 0.08398315262876693
    lam = 0.6822365335719091

    # Pre-compute derived coefficients for position/momentum updates
    c_pos = (1 - 2 * (lam + vartheta)) / 2
    c_mom = 1 - 2 * (theta + rho)

    for _ in range(n_steps):
        U = U_update(U, P, vartheta * step_size)
        P = P + force(U) * rho * step_size
        U = U_update(U, P, lam * step_size)
        P = P + force(U) * theta * step_size
        U = U_update(U, P, c_pos * step_size)
        P = P + force(U) * c_mom * step_size
        U = U_update(U, P, c_pos * step_size)
        P = P + force(U) * theta * step_size
        U = U_update(U, P, lam * step_size)
        P = P + force(U) * rho * step_size
        U = U_update(U, P, vartheta * step_size)

    return U, P
