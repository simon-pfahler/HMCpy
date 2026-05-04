"""
integrator.py -- Symplectic (leapfrog / Störmer-Verlet) integrator for HMC.

The molecular-dynamics equations of motion are:

    dU_mu(x)/dt  = P_mu(x) . U_mu(x)
    dP_mu(x)/dt  = F_gauge(U) + F_fermion(phi, U)

The total force is the sum of the gauge force and (optionally) the fermion
force from a pseudofermion field phi.

SU(3) link update: U_mu <- exp(eps * P_mu) . U_mu
  -- torch.linalg.matrix_exp for the matrix exponential.

Conventions match qcd_ml / HMCpy: U shape [4, Lx, Ly, Lz, Lt, Nc, Nc].
"""

from typing import Callable

import torch

from .physics import gauge_force, reunitarize

# ---------------------------------------------------------------------------
# Type alias for a total force function
# ---------------------------------------------------------------------------

# A ForceCallable takes U and returns a tensor of the same shape.
ForceCallable = Callable[[torch.Tensor], torch.Tensor]


def make_total_force(
    beta: float,
    phi: torch.Tensor | None = None,
    dirac_op=None,
    use_analytic_force: bool = True,
    cg_max_iter: int = 1000,
    cg_tol: float = 1e-12,
    fd_delta: float = 1e-6,
) -> ForceCallable:
    """
    Build a total force function  F(U) = F_gauge(U) [+ F_fermion(phi, U)].

    Parameters
    ----------
    beta               : inverse bare coupling for the gauge force
    phi                : pseudofermion field (None for pure gauge)
    dirac_op           : DiracOperator instance (required if phi is not None)
    use_analytic_force : if True, use fermion_force_wilson_analytic (fast);
                         if False, use the finite-difference fermion_force (slow
                         but operator-agnostic).  Default True.
    cg_max_iter, cg_tol : CG solver parameters passed to the fermion force
    fd_delta           : finite-difference step (only used when use_analytic_force=False)

    Returns
    -------
    force_fn : callable  U -> torch.Tensor (same shape as U)
    """
    if phi is not None and dirac_op is None:
        raise ValueError("dirac_op must be provided when phi is not None.")

    def force_fn(U: torch.Tensor) -> torch.Tensor:
        F = gauge_force(U, beta)
        if phi is not None:
            if use_analytic_force:
                from .fermion import fermion_force_wilson_analytic

                F = F + fermion_force_wilson_analytic(
                    phi,
                    U,
                    dirac_op,
                    cg_max_iter=cg_max_iter,
                    cg_tol=cg_tol,
                )
            else:
                from .fermion import fermion_force

                F = F + fermion_force(
                    phi,
                    U,
                    dirac_op,
                    cg_max_iter=cg_max_iter,
                    cg_tol=cg_tol,
                    delta=fd_delta,
                )
        return F

    return force_fn


# ---------------------------------------------------------------------------
# SU(3) exponential link update
# ---------------------------------------------------------------------------


def _exp_update_U(U: torch.Tensor, P: torch.Tensor, eps: float) -> torch.Tensor:
    """
    U_mu(x) <- exp(i * eps * P_mu(x)) . U_mu(x)

    P is su(3)-valued so exp(i * eps * P) is SU(3).
    """
    return torch.matmul(torch.linalg.matrix_exp(1j * eps * P), U)


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
    ] = _exp_update_U,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Leapfrog integrator for HMC molecular dynamics.

    Parameters
    ----------
    U, P       : initial phase-space point
    beta       : inverse coupling
    n_steps    : number of leapfrog steps
    force      : Function to obtain the force for a given field U
    U_update   : Function to update the field U given P
    step_size  : MD step size epsilon
    phi        : pseudofermion field (None = pure gauge)

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    OMF4 integrator for HMC molecular dynamics.

    Parameters
    ----------
    U, P       : initial phase-space point
    beta       : inverse coupling
    n_steps    : number of leapfrog steps
    force      : Function to obtain the force for a given field U
    step_size  : MD step size epsilon
    phi        : pseudofermion field (None = pure gauge)

    Returns
    -------
    U_new, P_new
    """

    # Optimal 4th-order OMF coefficients (from https://doi.org/10.1016/S0010-4655(02)00754-3)
    rho = 0.2539785108410595
    theta = -0.03230286765269967
    vartheta = 0.08398315262876693
    lam = 0.6822365335719091

    P = P + rho * step_size * force(U)
    for _ in range(n_steps - 1):
        U = _exp_update_U(U, P, lam * step_size)
        P = P + theta * step_size * force(U)
        U = _exp_update_U(U, P, (0.5 - lam) * step_size)
        P = P + (1.0 - 2.0 * (theta + rho)) * step_size * force(U)
        U = _exp_update_U(U, P, (0.5 - lam) * step_size)
        P = P + theta * step_size * force(U)
        U = _exp_update_U(U, P, lam * step_size)
        P = P + 2.0 * rho * step_size * force(U)  # merged boundary kick

    # Final step — close symmetrically without merging
    U = _exp_update_U(U, P, lam * step_size)
    P = P + theta * step_size * force(U)
    U = _exp_update_U(U, P, (0.5 - lam) * step_size)
    P = P + (1.0 - 2.0 * (theta + rho)) * step_size * force(U)
    U = _exp_update_U(U, P, (0.5 - lam) * step_size)
    P = P + theta * step_size * force(U)
    U = _exp_update_U(U, P, lam * step_size)
    P = P + rho * step_size * force(U)

    return U, P
