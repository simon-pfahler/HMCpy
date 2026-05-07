"""
fermion.py -- Pseudofermion fields for dynamical fermions in HMC.

This module provides:
  - apply_DDdag_inv(phi, D) -- Solves (DD^dag) chi = phi for chi
  - pseudofermion_action(phi, D, chi) -- S_pf = phi^dag chi where chi = (DD^dag)^{-1} phi
  - wilson_fermion_force(U, psi, D) -- Wilson fermion force for HMC
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


def wilson_fermion_force(
    U: torch.Tensor,
    psi: torch.Tensor,
    D: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Fermion force for the Wilson Dirac operator.

    The force is computed from the derivative of the pseudofermion action
    S = φ^dag (D D^dag)^{-1} φ with respect to the gauge field U, where
    ψ = (D D^dag)^{-1} φ.

    Using the formula:
        ∂S/∂U_μ(z) = -ψ^dag [∂D/∂U_μ(z) D^dag + D ∂D^dag/∂U_μ(z)] ψ

    For the Wilson Dirac operator, the key derivatives are:
        - ∂D(z|z+μ)/∂U_μ(z) = 1/2 (γ_μ - I) ⊗ I_color
        - ∂D(z+μ|z)/∂U_μ(z) = -1/2 (γ_μ + I) ⊗ I_color

    And from gamma-5 hermiticity: D^dag = γ_5 D γ_5, which implies
    ∂D^dag/∂U_μ(z) = (∂D/∂U_μ(z))^dag.

    For Wilson fermions in Euclidean space (γ_μ^2 = I), the D ∂D^dag/∂U terms
    vanish, leaving only:
        F_μ(z) = -2i/3 ψ^dag [∂D/∂U_μ(z) D^dag ψ]

    This gives a matrix in color space:
        F_μ(z)_{a,b} = -2i/3 [ ψ^dag(z)_{s,a} (γ_μ - I)_{s,s'} (D^dag ψ)(z+μ)_{s',b}
                           - ψ^dag(z+μ)_{s,a} (γ_μ + I)_{s,s'} (D^dag ψ)(z)_{s',b} ]

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
    Nc = U.shape[-1]
    Ns = psi.shape[4]
    eye_color = torch.eye(Nc, dtype=U.dtype, device=U.device)
    eye_spin = torch.eye(Ns, dtype=U.dtype, device=U.device)

    # Compute D^dag psi using gamma-5 hermiticity: D^dag psi = gamma5 D gamma5 psi
    psi_g5 = torch.einsum("ij,...jc->...ic", gamma5, psi)
    D_psi_g5 = D(psi_g5)
    D_dag_psi = torch.einsum("ij,...jc->...ic", gamma5, D_psi_g5)

    # Precompute shifted D_dag_psi for forward hops
    # This is fine as D(x|y) = D(x+a|y+a)
    D_dag_psi_fwd = [torch.roll(D_dag_psi, -1, dims=mu) for mu in range(4)]

    F = torch.zeros_like(U)

    for mu in range(4):
        gamma_mu_minus_I = gamma[mu] - eye_spin
        gamma_mu_plus_I = gamma[mu] + eye_spin

        # Term 1: ψ^dag(z) (γ_μ - I) H_-μ (D^dag ψ)(z+μ)
        term1 = torch.einsum(
            "...sa,ss,...sc->...ac",
            psi.conj(),
            gamma_mu_minus_I,
            D_dag_psi_fwd[mu],
        )

        # Term 2: ψ^dag(z+μ) (γ_μ + I) (D^dag ψ)(z)
        psi_fwd = torch.roll(psi, -1, dims=mu)
        term2 = torch.einsum(
            "...sa,ss,...sb->...ab", psi_fwd.conj(), gamma_mu_plus_I, D_dag_psi
        )

        # Compute force from user's formula: F = -2i/3 [ (γ_μ - I) term - (γ_μ + I) term + h.c. ]
        # Since we compute matrix elements, we explicitly Hermitianize the result.
        F_mu = (-2j / 3) * (term1 - term2)

        # Project to traceless Hermitian (su(Nc) Lie algebra)
        F_mu_herm = 0.5 * (F_mu + F_mu.conj().transpose(-1, -2))
        trace = torch.einsum("...ii->...", F_mu_herm) / Nc
        F_mu = F_mu_herm - trace.unsqueeze(-1).unsqueeze(-1) * eye_color

        F[mu] = F_mu

    return F
