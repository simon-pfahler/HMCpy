"""
fermion.py -- Pseudofermion fields for dynamical fermions in HMC.

This module provides:
  - apply_DDdag_inv(phi, D) -- Solves (DD^dag) chi = phi for chi
  - pseudofermion_action(phi, chi) -- S_pf = phi^dag chi where chi = (DD^dag)^{-1} phi
  - wilson_fermion_force(U, psi) -- Wilson fermion force for HMC
"""

from typing import Callable

import torch
from qcd_ml.base.hop import v_hop
from qcd_ml.qcd.dirac import gamma as gamma_list
from qcd_ml.util.solver import GMRES

from .utility import gell_mann_matrices

gamma = torch.stack(gamma_list)
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

    chi, _ = GMRES(DDdag_op, phi.clone(), phi.clone(), **(GMRES_kwargs or {}))
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


def wilson_fermion_force(
    U: torch.Tensor,
    psi: torch.Tensor,
    D: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Fermion force for the Wilson Dirac operator.

    Parameters
    ----------
    U : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Gauge field (Nc=3 for SU(3))
    psi : torch.Tensor, shape [Lx, Ly, Lz, Lt, Ns, Nc]
        Pseudofermion field: ψ = (D D^dag)^{-1} φ (Ns=4 for Dirac spinors)
    D : Callable[[torch.Tensor], torch.Tensor]
        Dirac operator (e.g., dirac_wilson_clover from qcd_ml)

    Returns
    -------
    F : torch.Tensor, same shape as U [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Fermion force, traceless Hermitian at each link
    """

    gens = 0.5 * gell_mann_matrices

    eye_spin = torch.eye(psi.shape[-2], dtype=U.dtype, device=U.device)

    Ddag_psi = apply_gamma5(D(apply_gamma5(psi.clone())))

    zeta_1 = torch.einsum("...sc,mst->m...tc", psi.conj(), eye_spin - gamma)
    zeta_2 = torch.einsum("...sc,mst->m...tc", psi.conj(), eye_spin + gamma)
    xi_1 = torch.stack([v_hop(U, mu, -1, Ddag_psi) for mu in range(4)])
    Ti_Ddag_psi = torch.einsum("icd,...d->i...c", gens, Ddag_psi)
    xi_2 = torch.stack(
        [
            torch.stack([v_hop(U, mu, 1, Ti_Ddag_psi[i]) for mu in range(4)])
            for i in range(8)
        ]
    )
    f_1 = torch.einsum("m...sc,icd,m...sd->im...", zeta_1, gens, xi_1)
    f_2 = torch.einsum("m...sc,im...sc->im...", zeta_2, xi_2)
    f_2 = torch.stack(
        [torch.roll(f_2[:, mu], -1, dims=mu + 1) for mu in range(4)], dim=1
    )

    F = torch.einsum(
        "icd,im...->m...cd",
        gens,
        (f_1 + f_2).imag.to(torch.cdouble),
    )

    return F
