"""
fermion.py -- Pseudofermion fields for dynamical fermions in HMC.

This module provides:
  - apply_DDdag_inv(phi, D) -- Solves (DD^dag) chi = phi for chi
  - pseudofermion_action(phi, chi) -- S_pf = phi^dag chi where chi = (DD^dag)^{-1} phi
  - wilson_fermion_force(U, psi) -- Wilson fermion force for HMC
  - wilson_clover_fermion_force(U, psi, csw) -- Wilson-clover fermion force for HMC
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

    zeta_1 = torch.einsum("...sc,mst->m...tc", psi.conj(), gamma - eye_spin)
    zeta_2 = torch.einsum("...sc,mst->m...tc", psi.conj(), gamma + eye_spin)
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

    return -F


# ---------------------------------------------------------------------------
# Fermion force for Wilson-Clover Dirac operator
# ---------------------------------------------------------------------------


def wilson_clover_fermion_force(
    U: torch.Tensor,
    psi: torch.Tensor,
    D: Callable[[torch.Tensor], torch.Tensor],
    csw: float,
) -> torch.Tensor:
    """
    Fermion force for the Wilson-Clover Dirac operator.

    The Wilson-Clover operator is D_WC = D_W + csw * D_Clover where the clover
    term improves the action to O(a) order.

    Parameters
    ----------
    U : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Gauge field (Nc=3 for SU(3))
    psi : torch.Tensor, shape [Lx, Ly, Lz, Lt, Ns, Nc]
        Pseudofermion field: ψ = (D D^dag)^{-1} φ (Ns=4 for Dirac spinors)
    D : Callable[[torch.Tensor], torch.Tensor]
        Dirac operator (e.g., dirac_wilson_clover from qcd_ml)
    csw : float
        Clover coefficient (improvement coefficient)

    Returns
    -------
    F : torch.Tensor, same shape as U [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Fermion force, traceless Hermitian at each link
    """
    # First compute the Wilson fermion force
    F_wilson = wilson_fermion_force(U, psi, D)

    # If csw is zero, return just the Wilson force
    if csw == 0.0:
        return F_wilson

    gens = 0.5 * gell_mann_matrices

    Ddag_psi = apply_gamma5(D(apply_gamma5(psi.clone())))

    # Compute sigma_{μν} = (i/2) [γ_μ, γ_ν]
    sigma_mn = torch.zeros((4, 4, 4, 4), dtype=torch.cdouble, device=U.device)
    for mu in range(4):
        for nu in range(4):
            sigma_mn[mu, nu] = 0.5 * (
                gamma[mu] @ gamma[nu] - gamma[nu] @ gamma[mu]
            )

    def compute_P(
        mu: int, nu: int, p: int, i: int, field: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute P_{μ,ν,i}^{(p)} applied to field.

        This is a plaquette in (-μ, -ν)-direction, with the generator T_i
        inserted at position p.

        Parameters
        ----------
        mu, nu : direction indices (0-3 for x,y,z,t)
        p : which P term (0-4)
        i : generator index (0-7)
        field : input spinor field

        Returns
        -------
        Result of P_{μ,ν,i}^{(p)} applied to field
        """
        result = field.clone()

        if p == 4:
            result = torch.einsum("cd,...d->...c", gens[i], result)
        result = v_hop(U, nu, -1, result)

        if p == 3:
            result = torch.einsum("cd,...d->...c", gens[i], result)
        result = v_hop(U, mu, -1, result)

        if p == 2:
            result = torch.einsum("cd,...d->...c", gens[i], result)
        result = v_hop(U, nu, 1, result)

        if p == 1:
            result = torch.einsum("cd,...d->...c", gens[i], result)
        result = v_hop(U, mu, 1, result)

        if p == 0:
            result = torch.einsum("cd,...d->...c", gens[i], result)

        return result

    lattice_dims = [psi.shape[d] for d in range(4)]

    # Initialize force accumulation for each generator and direction
    # f_clover[i, sigma, x] is a scalar (the coefficient for T_i at direction sigma and site x)
    f_clover = torch.zeros(
        (8, 4, *lattice_dims), dtype=torch.cdouble, device=U.device
    )

    for i in range(8):  # loop over generators
        for sigma in range(4):  # loop over direction σ
            for mu in range(4):  # loop over direction μ
                if mu == sigma:
                    continue

                sigma_sigma_mu = sigma_mn[sigma, mu]

                # p=0: P_{-σ,-μ,i}^{(0)} - P_{-σ,μ,i}^{(0)}
                # Note: In our convention, negative directions are -mu-1
                P0_term1 = compute_P(sigma, mu, 0, i, Ddag_psi)
                P0_term2 = compute_P(sigma, -mu - 1, 0, i, Ddag_psi)
                p0 = P0_term2 - P0_term1

                # p=1: P_{μ,-σ,i}^{(1)}(z+μ) - P_{-μ,-σ,i}^{(1)}(z-μ)
                P1a_pos = compute_P(mu, -sigma - 1, 1, i, Ddag_psi)
                P1a_neg = compute_P(-mu - 1, -sigma - 1, 1, i, Ddag_psi)
                P1a_pos = torch.roll(P1a_pos, -1, dims=[mu])
                P1a_neg = torch.roll(P1a_neg, 1, dims=[mu])

                # p=1: P_{σ,-μ,i}^{(1)}(z+σ) - P_{σ,μ,i}^{(1)}(z+σ)
                P1b_pos = compute_P(sigma, -mu - 1, 1, i, Ddag_psi)
                P1b_neg = compute_P(sigma, mu, 1, i, Ddag_psi)
                P1b_pos = torch.roll(P1b_pos, -1, dims=[sigma])
                P1b_neg = torch.roll(P1b_neg, -1, dims=[sigma])

                p1 = P1a_pos - P1a_neg + P1b_pos - P1b_neg

                # p=2: P_{μ,σ,i}^{(2)}(z+σ+μ) - P_{-μ,σ,i}^{(2)}(z+σ-μ)
                P2a_pos = compute_P(mu, sigma, 2, i, Ddag_psi)
                P2a_neg = compute_P(-mu - 1, sigma, 2, i, Ddag_psi)
                P2a_pos = torch.roll(P2a_pos, [-1, -1], dims=[sigma, mu])
                P2a_neg = torch.roll(P2a_neg, [-1, 1], dims=[sigma, mu])

                # p=2: P_{σ,μ,i}^{(2)}(z+σ+μ) - P_{σ,-μ,i}^{(2)}(z+σ-μ)
                P2b_pos = compute_P(sigma, mu, 2, i, Ddag_psi)
                P2b_neg = compute_P(sigma, -mu - 1, 2, i, Ddag_psi)
                P2b_pos = torch.roll(P2b_pos, [-1, -1], dims=[sigma, mu])
                P2b_neg = torch.roll(P2b_neg, [-1, 1], dims=[sigma, mu])

                p2 = P2a_pos - P2a_neg + P2b_pos - P2b_neg

                # p=3: P_{-μ,σ,i}^{(3)}(z+σ) - P_{μ,σ,i}^{(3)}(z+σ)
                P3a_pos = compute_P(-mu - 1, sigma, 3, i, Ddag_psi)
                P3a_neg = compute_P(mu, sigma, 3, i, Ddag_psi)
                P3a_pos = torch.roll(P3a_pos, -1, dims=[sigma])
                P3a_neg = torch.roll(P3a_neg, -1, dims=[sigma])

                # p=3: P_{-σ,μ,i}^{(3)}(z+μ) - P_{-σ,-μ,i}^{(3)}(z-μ)
                P3b_pos = compute_P(-sigma - 1, mu, 3, i, Ddag_psi)
                P3b_neg = compute_P(-sigma - 1, -mu - 1, 3, i, Ddag_psi)
                P3b_pos = torch.roll(P3b_pos, -1, dims=[mu])
                P3b_neg = torch.roll(P3b_neg, 1, dims=[mu])

                p3 = P3a_pos - P3a_neg + P3b_pos - P3b_neg

                # p=4: P_{μ,-σ,i}^{(4)} - P_{-μ,-σ,i}^{(4)}
                P4_pos = compute_P(mu, -sigma - 1, 4, i, Ddag_psi)
                P4_neg = compute_P(-mu - 1, -sigma - 1, 4, i, Ddag_psi)
                p4 = P4_pos - P4_neg

                # Sum all P terms
                P_total = p0 + p1 + p2 + p3 + p4

                # Apply sigma_{σμ} to spin index
                P_total = torch.einsum(
                    "st,...t c->...s c", sigma_sigma_mu, P_total
                )

                # Contract with psi^dag and take imaginary part
                f_clover[i, sigma] -= (
                    csw
                    / 8
                    * torch.einsum(
                        "...sc,...sc->...", psi.conj(), P_total
                    ).imag.to(torch.cdouble)
                )

    # Combine generator contributions with T_i matrices
    F_clover = torch.einsum("im...,icd->m...cd", f_clover, gens)

    # Combine Wilson and clover forces
    F_total = F_wilson + F_clover

    return F_total
