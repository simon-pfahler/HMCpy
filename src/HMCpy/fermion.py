"""
fermion.py -- Pseudofermion fields for dynamical fermions in HMC.

This module provides:
  - wilson_fermion_force(U, psi, Ddag_inv_psi) -- Wilson fermion force for HMC
  - wilson_clover_fermion_force(U, psi, Ddag_inv_psi, csw) -- Wilson-clover fermion force for HMC
"""

import torch
from qcd_ml.base.hop import v_hop
from qcd_ml.qcd.dirac import gamma as gamma_list

from .utility import su3_generators

gamma = torch.stack(gamma_list)
gamma5 = gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]

# ---------------------------------------------------------------------------
# Fermion force for Wilson Dirac operator
# ---------------------------------------------------------------------------


def wilson_fermion_force(
    U: torch.Tensor,
    psi: torch.Tensor,
    Ddag_inv_psi: torch.Tensor,
) -> torch.Tensor:
    """
    Fermion force for the Wilson Dirac operator.

    Parameters
    ----------
    U : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Gauge field (Nc=3 for SU(3))
    psi : torch.Tensor, shape [Lx, Ly, Lz, Lt, Ns, Nc]
        Pseudofermion field: ψ = D^{-1} φ (Ns=4 for Dirac spinors)
    Ddag_inv_psi: torch.Tensor, shape[Lx, Ly, Lz, Lt, Ns, Nc]
        Pseudofermion field: D^{-dag} ψ (Ns=4 for Dirac spinors)

    Returns
    -------
    F : torch.Tensor, same shape as U [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Fermion force, traceless Hermitian at each link
    """
    device = psi.device

    eye_spin = torch.eye(psi.shape[-2], dtype=U.dtype)

    gen_device = su3_generators.to(device)

    zeta_1 = torch.einsum(
        "...sc,mst->m...tc", Ddag_inv_psi.conj(), (gamma - eye_spin).to(device)
    )
    zeta_2 = torch.einsum(
        "...sc,mst->m...tc", Ddag_inv_psi.conj(), (gamma + eye_spin).to(device)
    )
    xi_1 = torch.stack([v_hop(U, mu, -1, psi) for mu in range(4)])
    Ti_psi = torch.einsum("icd,...d->i...c", gen_device, psi)
    xi_2 = torch.stack(
        [
            torch.stack([v_hop(U, mu, 1, Ti_psi[i]) for mu in range(4)])
            for i in range(8)
        ]
    )
    f_1 = torch.einsum("m...sc,icd,m...sd->im...", zeta_1, gen_device, xi_1)
    f_2 = torch.einsum("m...sc,im...sc->im...", zeta_2, xi_2)
    f_2 = torch.stack(
        [torch.roll(f_2[:, mu], -1, dims=mu + 1) for mu in range(4)], dim=1
    )

    F = torch.einsum(
        "icd,im...->m...cd",
        gen_device,
        (f_1 + f_2).imag.to(torch.cdouble),
    )

    return -F


# ---------------------------------------------------------------------------
# Fermion force for Wilson-Clover Dirac operator
# ---------------------------------------------------------------------------


def wilson_clover_fermion_force(
    U: torch.Tensor,
    psi: torch.Tensor,
    Ddag_inv_psi: torch.Tensor,
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
    Ddag_inv_psi : torch.Tensor, shape [Lx, Ly, Lz, Lt, Ns, Nc]
        Pseudofermion field: D^{-dag} ψ (Ns=4 for Dirac spinors)
    csw : float
        Clover coefficient (improvement coefficient)

    Returns
    -------
    F : torch.Tensor, same shape as U [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Fermion force, traceless Hermitian at each link
    """
    # First compute the Wilson fermion force
    F_wilson = wilson_fermion_force(U, psi, Ddag_inv_psi)

    # If csw is zero, return just the Wilson force
    if csw == 0.0:
        return F_wilson

    device = psi.device

    gamma_device = gamma.to(device)
    gen_device = su3_generators.to(device)

    # Compute sigma_{μν} = 1/2 [γ_μ, γ_ν]
    sigma_mn = torch.zeros((4, 4, 4, 4), dtype=torch.cdouble, device=U.device)
    for mu in range(4):
        for nu in range(4):
            sigma_mn[mu, nu] = 0.5 * (
                gamma_device[mu] @ gamma_device[nu]
                - gamma_device[nu] @ gamma_device[mu]
            )

    lattice_dims = [psi.shape[d] for d in range(4)]

    # Initialize force accumulation for each generator and direction
    f_clover = torch.zeros(
        (8, 4, *lattice_dims), dtype=torch.cdouble, device=U.device
    )

    def compute_P(
        mu: int,
        nu: int,
        mudir: int,
        nudir: int,
        p: int,
        i: int,
        field: torch.Tensor,
    ) -> torch.Tensor:
        """Plaquette in (μ, ν)-direction with generator T_i at position p."""
        result = field.clone()
        gen = gen_device[i]

        if p == 0:
            result = torch.einsum("cd,...d->...c", gen, result)
        result = v_hop(U, mu, mudir, result)
        if p == 1:
            result = torch.einsum("cd,...d->...c", gen, result)
        result = v_hop(U, nu, nudir, result)
        if p == 2:
            result = torch.einsum("cd,...d->...c", gen, result)
        result = v_hop(U, mu, -mudir, result)
        if p == 3:
            result = torch.einsum("cd,...d->...c", gen, result)
        result = v_hop(U, nu, -nudir, result)
        if p == 4:
            result = torch.einsum("cd,...d->...c", gen, result)

        return result

    def _make_contrib(
        a: int | str,
        b: int | str,
        mudir: int,
        nudir: int,
        p: int,
        i: int,
        sigma: int,
        out_pattern: str = "...pc",
    ) -> torch.Tensor:
        """Create clover force einsum term. a, b can be "mu", "sigma", or int."""

        def resolve(x: int | str, mu: int) -> int:
            if x == "mu":
                return mu
            elif x == "sigma":
                return sigma
            elif type(x) == int:
                return x
            return 0

        P_results = torch.stack(
            [
                compute_P(
                    resolve(a, mu), resolve(b, mu), mudir, nudir, p, i, psi
                )
                for mu in range(4)
            ]
        )
        return torch.einsum(
            "mpr,m...rc->" + out_pattern, sigma_mn[sigma], P_results
        )

    # contributions with no offset
    for i in range(8):
        for sigma in range(4):
            contrib = +_make_contrib("mu", sigma, 1, 1, 4, i, sigma, "...pc")
            contrib -= _make_contrib(sigma, "mu", 1, -1, 0, i, sigma, "...pc")
            contrib += _make_contrib(sigma, "mu", 1, 1, 0, i, sigma, "...pc")
            contrib -= _make_contrib("mu", sigma, -1, 1, 4, i, sigma, "...pc")
            f_clover[i, sigma] += torch.einsum(
                "...sc,...sc->...", Ddag_inv_psi.conj(), contrib
            )

    # contributions with offset mu
    for i in range(8):
        for sigma in range(4):
            contrib = +_make_contrib(sigma, "mu", 1, -1, 3, i, sigma, "m...pc")
            contrib += _make_contrib("mu", sigma, -1, 1, 1, i, sigma, "m...pc")
            inner = torch.einsum(
                "...sc,m...sc->m...", Ddag_inv_psi.conj(), contrib
            )
            f_clover[i, sigma] += torch.einsum(
                "m...->...",
                torch.stack(
                    [torch.roll(e, -1, mu) for mu, e in enumerate(inner)]
                ),
            )

    # contributions with offset -mu
    for i in range(8):
        for sigma in range(4):
            contrib = -_make_contrib(sigma, "mu", 1, 1, 3, i, sigma, "m...pc")
            contrib -= _make_contrib("mu", sigma, 1, 1, 1, i, sigma, "m...pc")
            inner = torch.einsum(
                "...sc,m...sc->m...", Ddag_inv_psi.conj(), contrib
            )
            f_clover[i, sigma] += torch.einsum(
                "m...->...",
                torch.stack(
                    [torch.roll(e, 1, mu) for mu, e in enumerate(inner)]
                ),
            )

    # contributions with offset sigma
    for i in range(8):
        for sigma in range(4):
            contrib = +_make_contrib(sigma, "mu", -1, 1, 1, i, sigma, "...pc")
            contrib -= _make_contrib("mu", sigma, -1, -1, 3, i, sigma, "...pc")
            contrib += _make_contrib("mu", sigma, 1, -1, 3, i, sigma, "...pc")
            contrib -= _make_contrib(sigma, "mu", -1, -1, 1, i, sigma, "...pc")
            f_clover[i, sigma] += torch.roll(
                torch.einsum("...sc,...sc->...", Ddag_inv_psi.conj(), contrib),
                -1,
                sigma,
            )

    # contributions with offset sigma+mu
    for i in range(8):
        for sigma in range(4):
            contrib = +_make_contrib("mu", sigma, -1, -1, 2, i, sigma, "m...pc")
            contrib += _make_contrib(sigma, "mu", -1, -1, 2, i, sigma, "m...pc")
            inner = torch.einsum(
                "...sc,m...sc->m...", Ddag_inv_psi.conj(), contrib
            )
            f_clover[i, sigma] += torch.einsum(
                "m...->...",
                torch.stack(
                    [
                        torch.roll(e, [-1, -1], [sigma, mu])
                        for mu, e in enumerate(inner)
                    ]
                ),
            )

    # contributions with offset sigma-mu
    for i in range(8):
        for sigma in range(4):
            contrib = -_make_contrib(sigma, "mu", -1, 1, 2, i, sigma, "m...pc")
            contrib -= _make_contrib("mu", sigma, 1, -1, 2, i, sigma, "m...pc")
            inner = torch.einsum(
                "...sc,m...sc->m...", Ddag_inv_psi.conj(), contrib
            )
            f_clover[i, sigma] += torch.einsum(
                "m...->...",
                torch.stack(
                    [
                        torch.roll(e, [-1, 1], [sigma, mu])
                        for mu, e in enumerate(inner)
                    ]
                ),
            )

    # Combine generator contributions with T_i matrices
    F_clover = torch.einsum(
        "icd,im...->m...cd", gen_device, f_clover.imag.to(torch.cdouble)
    )

    # Combine Wilson and clover forces
    F_total = F_wilson + csw / 8 * F_clover

    return F_total
