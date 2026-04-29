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

import torch
from typing import Callable
from .physics import gauge_force


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
                    phi, U, dirac_op,
                    cg_max_iter=cg_max_iter, cg_tol=cg_tol,
                )
            else:
                from .fermion import fermion_force
                F = F + fermion_force(
                    phi, U, dirac_op,
                    cg_max_iter=cg_max_iter, cg_tol=cg_tol,
                    delta=fd_delta,
                )
        return F

    return force_fn


# ---------------------------------------------------------------------------
# SU(3) exponential link update
# ---------------------------------------------------------------------------

def _exp_update_U(U: torch.Tensor, P: torch.Tensor, eps: float) -> torch.Tensor:
    """
    U_mu(x) <- exp(eps * P_mu(x)) . U_mu(x)

    P is su(3)-valued so exp(eps*P) is SU(3).
    """
    return torch.matmul(torch.linalg.matrix_exp(eps * P), U)


# ---------------------------------------------------------------------------
# Leapfrog (Störmer-Verlet) integrator
# ---------------------------------------------------------------------------

def leapfrog(
    U: torch.Tensor,
    P: torch.Tensor,
    beta: float,
    n_steps: int,
    step_size: float,
    phi: torch.Tensor | None = None,
    dirac_op=None,
    use_analytic_force: bool = True,
    cg_max_iter: int = 1000,
    cg_tol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Leapfrog integrator for HMC molecular dynamics.

    Algorithm:
        P_{1/2}   = P_0 + (eps/2) * F(U_0)
        for k = 1 .. n_steps-1:
            U_k   = exp(eps * P_{k-1/2}) . U_{k-1}
            P_{k+1/2} = P_{k-1/2} + eps * F(U_k)
        U_n   = exp(eps * P_{n-1/2}) . U_{n-1}
        P_n   = P_{n-1/2} + (eps/2) * F(U_n)

    Parameters
    ----------
    U, P       : initial phase-space point
    beta       : inverse coupling
    n_steps    : number of leapfrog steps
    step_size  : MD step size epsilon
    phi        : pseudofermion field (None = pure gauge)
    dirac_op   : DiracOperator (required if phi is not None)
    use_analytic_force : see make_total_force
    cg_max_iter, cg_tol : CG solver parameters

    Returns
    -------
    U_new, P_new
    """
    eps   = step_size
    force = make_total_force(beta, phi=phi, dirac_op=dirac_op,
                             use_analytic_force=use_analytic_force,
                             cg_max_iter=cg_max_iter, cg_tol=cg_tol)

    P = P + (eps / 2) * force(U)
    for _ in range(n_steps - 1):
        U = _exp_update_U(U, P, eps)
        P = P + eps * force(U)
    U = _exp_update_U(U, P, eps)
    P = P + (eps / 2) * force(U)

    return U, P


# ---------------------------------------------------------------------------
# 2nd-order Omelyan-Mryglod-Folk (OMF2) integrator
# ---------------------------------------------------------------------------

_OMF2_LAMBDA = 0.1931833275037836  # optimal 2nd-order OMF coefficient


def omf2(
    U: torch.Tensor,
    P: torch.Tensor,
    beta: float,
    n_steps: int,
    step_size: float,
    phi: torch.Tensor | None = None,
    dirac_op=None,
    use_analytic_force: bool = True,
    cg_max_iter: int = 1000,
    cg_tol: float = 1e-12,
    lam: float = _OMF2_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    2nd-order Omelyan-Mryglod-Folk (OMF2) integrator.

    Step pattern per integration step: [lam, 1/2, 1-2*lam, 1/2, lam].
    More accurate than leapfrog at the same number of force evaluations.

    Parameters
    ----------
    U, P       : initial phase-space point
    beta       : inverse coupling
    n_steps    : number of OMF2 steps
    step_size  : total step size epsilon
    phi        : pseudofermion field (None = pure gauge)
    dirac_op   : DiracOperator (required if phi is not None)
    use_analytic_force : see make_total_force
    cg_max_iter, cg_tol : CG solver parameters
    lam        : OMF2 parameter (default: optimal value ~0.193)

    Returns
    -------
    U_new, P_new
    """
    eps   = step_size
    force = make_total_force(beta, phi=phi, dirac_op=dirac_op,
                             use_analytic_force=use_analytic_force,
                             cg_max_iter=cg_max_iter, cg_tol=cg_tol)

    P = P + lam * eps * force(U)
    for _ in range(n_steps - 1):
        U = _exp_update_U(U, P, 0.5 * eps)
        P = P + (1 - 2 * lam) * eps * force(U)
        U = _exp_update_U(U, P, 0.5 * eps)
        P = P + 2 * lam * eps * force(U)

    U = _exp_update_U(U, P, 0.5 * eps)
    P = P + (1 - 2 * lam) * eps * force(U)
    U = _exp_update_U(U, P, 0.5 * eps)
    P = P + lam * eps * force(U)

    return U, P
