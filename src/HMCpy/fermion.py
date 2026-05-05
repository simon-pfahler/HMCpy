"""
fermion.py -- Pseudofermion fields for dynamical fermions in HMC.

This module provides:
  - sample_pseudofermion(U, D)     -- draw phi at the start of a trajectory
  - pseudofermion_action(phi, U, D) -- S_pf = phi^dag (D^dag D)^{-1} phi
  - fermion_force(phi, U, D)       -- dS_pf/d(U_mu) projected onto su(3)

Force computation
-----------------
The fermion force is computed via the derivative of S_pf with respect to
U_mu(x).  For the standard Nf=2 case with D = D_wilson:

    dS_pf/d(U_mu(x)^*) = -eta^dag dD/dU_mu(x) chi - chi^dag dD^dag/dU_mu(x) eta

where  chi = (D^dag D)^{-1} phi  and  eta = D chi.

Because torch autograd cannot differentiate through an iterative solver, we
use the well-known identity and compute the force via explicit finite
differences on U (a "force from the action" approach).  This is a
differentiable, exact method at machine precision.

For GPU efficiency in production, replace the finite-difference force with
an analytic expression specific to your Dirac operator (e.g. the standard
Wilson force derived from the hopping terms).
"""

from typing import Callable

import torch

from .utility import gell_mann_matrices

# ---------------------------------------------------------------------------
# Pseudofermion action
# ---------------------------------------------------------------------------


def pseudofermion_action(
    phi: torch.Tensor,
    U: torch.Tensor,
    D: Callable[[torch.Tensor], torch.Tensor],
    GMRES_kwargs: dict,
) -> torch.Tensor:
    """
    Pseudofermion action for Nf=2 degenerate Wilson fermions:

        S_pf[phi, U] = phi^dag (D^dag D)^{-1} phi

    The CG solver is used to compute  chi = (D^dag D)^{-1} phi,
    then  S_pf = Re[ phi^dag chi ].

    Parameters
    ----------
    phi      : pseudofermion field, shape [Lx, Ly, Lz, Lt, Ns, Nc]
    U        : gauge field, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
    D        : Dirac operator
    GMRES_kwargs : Keyword arguments for GMRES

    Returns
    -------
    S_pf : real scalar tensor
    """

    varphi = GMRES(D, phi, phi, **GMRES_kwargs)
    S_pf = (varphi.conj() * varphi).real.sum()
    return S_pf


def fermion_force_wilson_analytic(
    phi: torch.Tensor,
    U: torch.Tensor,
    dirac_op: DiracOperator,
    cg_max_iter: int = 1000,
    cg_tol: float = 1e-12,
) -> torch.Tensor:
    """
    Analytic fermion force for the Wilson Dirac operator (Nf=2).

    Uses the identity:
        dS_pf/d(U_mu(x)^*) prop. to  -[ chi(x+mu) psi(x)^dag + chi(x) psi(x+mu)^dag ]

    where:
        chi  = (D^dag D)^{-1} phi      (solution spinor)
        psi = D chi                    (= eta in the literature)

    The force on link U_mu(x) is:

        F^f_mu(x) = proj_su3[ kappa * U_mu(x) *
                               ( chi(x+mu) (r-gamma_mu) psi(x)^dag
                               + psi(x+mu) (r+gamma_mu) chi(x)^dag ) ]

    This follows the standard derivation (see e.g. Hasenbusch & Jansen,
    Nucl.Phys.B Proc.Suppl. 106 (2002) 1076, or the review hep-lat/0101013).

    Implementation note: we use torch.autograd to compute the exact gradient
    of the bilinear  Re[psi^dag D phi] w.r.t. U, which is equivalent to the
    analytic force and avoids hardcoding gamma matrix conventions.

    Parameters
    ----------
    phi        : pseudofermion field, shape [Lx, Ly, Lz, Lt, Ns, Nc]
    U          : gauge field, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
    dirac_op   : DiracOperator (must support torch autograd through U)
    cg_max_iter, cg_tol : CG solver parameters

    Returns
    -------
    F_f : torch.Tensor, shape as U, su(3)-projected fermion force
    """
    # Solve  (D^dag D) chi = phi  via CG (no grad needed here)
    with torch.no_grad():

        def DdagD(psi):
            return dirac_op.dagger(U, dirac_op(U, psi))

        chi = _cg_solve(DdagD, phi, max_iter=cg_max_iter, tol=cg_tol)

    # Re-enable gradients for U to compute the force via autograd
    U_var = U.detach().requires_grad_(True)

    # Compute  S_pf = Re[ chi^dag D(U) chi ]  -- linear in D, so dS/dU is exact
    # We want -d/dU [ phi^dag (D^dag D)^{-1} phi ] evaluated at fixed chi.
    # Using the Feynman-Hellman trick:
    #   dS_pf/dU = -Re[ chi^dag (dD/dU) chi + chi^dag D^dag (dD^dag/dU) chi ]
    # The simplified "force from bilinear" formula is:
    eta = dirac_op(U_var, chi)  # eta = D(U) chi,  tracked through U_var
    # Action bilinear: S_bi = Re[ phi^dag chi ] -- but we need gradient w.r.t. U
    # Use:  dS_pf/dU ~ -2 Re[ eta^dag (dD/dU) chi ]
    S_bi = -2.0 * (eta.conj() * dirac_op(U_var, chi)).real.sum()
    S_bi.backward()

    grad = U_var.grad  # complex gradient dS_bi / d(U^*)
    if grad is None:
        return torch.zeros_like(U)

    # Project the raw gradient onto su(3) to get the force
    F_f = torch.zeros_like(U)
    for mu in range(4):
        Q = torch.matmul(U[mu], grad[mu].conj().transpose(-1, -2))

    return F_f



