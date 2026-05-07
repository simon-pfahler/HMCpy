"""
fermion.py -- Pseudofermion fields for dynamical fermions in HMC.

This module provides:
  - apply_DDdag_inv(phi, D) -- Solves (DD^dag) chi = phi for chi
  - pseudofermion_action(phi, D, chi) -- S_pf = phi^dag chi where chi = (DD^dag)^{-1} phi
"""

from typing import Callable

import torch
from qcd_ml.qcd.dirac import gamma
from qcd_ml.util.solver import GMRES

gamma5 = gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]


def apply_gamma5(psi: torch.Tensor) -> torch.Tensor:
    """
    Apply gamma_5 to a spinor field.

    Parameters
    ----------
    psi : spinor field [Lx, Ly, Lz, Lt, Ns, Nc]

    Returns
    -------
    gamma5 @ psi : spinor field [Lx, Ly, Lz, Lt, Ns, Nc]
    """
    return torch.einsum("ij,...jc->...ic", gamma5, psi)


# ---------------------------------------------------------------------------
# Solve (DD^dag) chi = phi
# ---------------------------------------------------------------------------


def apply_DDdag_inv(
    phi: torch.Tensor,
    D: Callable[[torch.Tensor], torch.Tensor],
    GMRES_kwargs: dict | None = None,
) -> torch.Tensor:
    """
    Solve (DD^dag) chi = phi for chi using gamma_5-hermiticity.

    Uses D^dag psi = gamma5 @ D(gamma5 @ psi) from gamma_5-hermiticity.
    Solves (DD^dag) chi = phi via GMRES treating DD^dag as an operator.

    Alternatively, using the identity (DD^dag)^{-1} = gamma5 (D^dag D)^{-1} gamma5,
    we could solve in two steps:
        eta = D^{-1} gamma5 phi
        chi = gamma5 D^{-1} gamma5 eta
    But the direct approach is more stable.

    Parameters
    ----------
    phi : pseudofermion field, shape [Lx, Ly, Lz, Lt, Ns, Nc]
    D : Dirac operator (callable: D(psi) returns D psi)
    GMRES_kwargs : optional dict of keyword arguments for GMRES solver

    Returns
    -------
    chi : spinor field [Lx, Ly, Lz, Lt, Ns, Nc], solution to (DD^dag) chi = phi
    """

    def DDdag_op(psi: torch.Tensor) -> torch.Tensor:
        """Operator: (DD^dag) psi = D (gamma5 @ D (gamma5 @ psi))"""
        # D^dag psi = gamma5 @ D(gamma5 @ psi) from gamma_5-hermiticity
        gamma5_psi = apply_gamma5(psi)
        D_gamma5_psi = D(gamma5_psi)
        Ddag_psi = apply_gamma5(D_gamma5_psi)
        return D(Ddag_psi)

    chi, _ = GMRES(DDdag_op, phi, phi, **(GMRES_kwargs or {}))
    return chi


def pseudofermion_action(
    phi: torch.Tensor,
    chi: torch.Tensor,
) -> torch.Tensor:
    """
    Pseudofermion action: S_pf = phi^dag (DD^dag)^{-1} phi.

    Parameters
    ----------
    phi : pseudofermion field, shape [Lx, Ly, Lz, Lt, Ns, Nc]
    chi : solution to (DD^dag) chi = phi, shape [Lx, Ly, Lz, Lt, Ns, Nc]

    Returns
    -------
    S_pf : real scalar tensor, phi^dag chi where chi = (DD^dag)^{-1} phi
    """
    S_pf = (phi.conj() * chi).sum().real
    return S_pf


# ---------------------------------------------------------------------------
# Fermion force for Wilson Dirac operator
# ---------------------------------------------------------------------------
