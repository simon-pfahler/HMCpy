"""Tests for HMCpy.fermion module: pseudofermion action, derivatives, force."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test.utils import DTYPE, NC, L, cold_start, conjugate_gradient, hot_start
from typing import Callable

from HMCpy.fermion import pseudofermion_action
from HMCpy.utility import gell_mann_matrices
from src.HMCpy.fermion import apply_DDdag_inv, apply_gamma5

GMRES_OPTS = {"maxiter": 1000, "eps": 1e-10, "inner_iter": 10}


def _numerical_force_comp(
    U: torch.Tensor,
    phi: torch.Tensor,
    D_factory: Callable,
    mu: int,
    site: tuple,
    a: int,
    eps: float = 1e-5,
    GMRES_kwargs: dict | None = None,
) -> float:
    """Numerical fermion force component via central finite difference."""
    from qcd_ml.util.solver import GMRES

    gen = 0.5 * gell_mann_matrices[a]
    exp_p = torch.linalg.matrix_exp(1j * eps * gen)
    exp_m = torch.linalg.matrix_exp(-1j * eps * gen)
    idx = (mu,) + site

    def solve_DDdag(phi_in, D):
        def DDdag_op(psi):
            return D(apply_gamma5(D(apply_gamma5(psi))))

        chi, _ = GMRES(
            DDdag_op, phi_in, torch.zeros_like(phi_in), **(GMRES_kwargs or {})
        )
        return chi

    U_p = U.clone()
    U_p[idx] = exp_p @ U[idx]
    U_m = U.clone()
    U_m[idx] = exp_m @ U[idx]

    chi_p = solve_DDdag(phi, D_factory(U_p))
    chi_m = solve_DDdag(phi, D_factory(U_m))

    return (
        pseudofermion_action(phi, chi_p) - pseudofermion_action(phi, chi_m)
    ).item() / (2 * eps)


def _autograd_force_comp(
    U: torch.Tensor,
    phi: torch.Tensor,
    D_factory: Callable,
    mu: int,
    site: tuple,
    a: int,
    CG_kwargs: dict | None = None,
) -> float:
    """Autograd fermion force component."""
    idx = (mu,) + site
    gen = 0.5 * gell_mann_matrices[a]
    U_grad = U.detach().clone().requires_grad_(True)

    def S_pf(U_in):
        D = D_factory(U_in)

        def DDdag_op(psi):
            return D(apply_gamma5(D(apply_gamma5(psi))))

        chi = conjugate_gradient(DDdag_op, phi, **(CG_kwargs or {}))
        return pseudofermion_action(phi, chi).real

    grad_full = torch.autograd.grad(S_pf(U_grad), U_grad, create_graph=False)[0]
    link_grad = (
        grad_full[idx]
        if grad_full is not None
        else torch.zeros(NC, NC, dtype=DTYPE)
    )

    return (link_grad.conj() * (1j * gen @ U[idx])).sum().real.item()


def _analytic_force_comp(
    F: torch.Tensor, mu: int, site: tuple, a: int
) -> float:
    """Analytic fermion force component from force tensor."""
    return -(
        2 * torch.trace(0.5 * gell_mann_matrices[a] @ F[(mu,) + site])
    ).real.item()


class TestFermion:
    @pytest.mark.parametrize(
        "U_fn, name", [(cold_start, "cold"), (hot_start, "hot")]
    )
    def test_pseudofermion_action_pos_finite(self, U_fn, name):
        """Pseudofermion action is positive and finite."""
        from qcd_ml.qcd.dirac import dirac_wilson

        U = U_fn()
        m = 0.1
        D = dirac_wilson(U, mass_parameter=m)
        # Normalized uniform phi: norm = sqrt(L^4 * 4 * NC) for ones
        phi = torch.ones(L, L, L, L, 4, NC, dtype=DTYPE)
        phi_norm = phi.norm()
        phi = phi / phi_norm
        chi = apply_DDdag_inv(phi, D, GMRES_kwargs=GMRES_OPTS)
        S = pseudofermion_action(phi, chi)

        assert (
            S.item() > 0
        ), f"[{name}] Action should be positive, got {S.item()}"
        assert torch.isfinite(S), f"[{name}] Action should be finite"

        if name == "cold":
            assert S == pytest.approx(
                1 / m**2, abs=1e-2
            ), f"[{name}] Expected 1/m^2=100, got {S.item()}"

    def test_pseudofermion_action_real(self):
        """Pseudofermion action is real for Hermitian DDdagger."""
        from qcd_ml.qcd.dirac import dirac_wilson

        U = cold_start()
        D = dirac_wilson(U, mass_parameter=0.1)
        phi = torch.randn(L, L, L, L, 4, NC, dtype=DTYPE)
        chi = apply_DDdag_inv(phi, D, GMRES_kwargs=GMRES_OPTS)

        assert (pseudofermion_action(phi, chi)).isreal()
        assert torch.isfinite(pseudofermion_action(phi, chi))

    def test_pseudofermion_action_scales_quadratically(self):
        """Action scales as phi^2 (S ~ phi^dag chi, chi ~ phi)."""
        from qcd_ml.qcd.dirac import dirac_wilson

        U = cold_start()
        D = dirac_wilson(U, mass_parameter=0.1)
        phi = torch.randn(L, L, L, L, 4, NC, dtype=DTYPE)

        chi1 = apply_DDdag_inv(phi, D, GMRES_kwargs=GMRES_OPTS)
        chi2 = apply_DDdag_inv(2 * phi, D, GMRES_kwargs=GMRES_OPTS)

        S1 = pseudofermion_action(phi, chi1)
        S2 = pseudofermion_action(2 * phi, chi2)

        assert S2.item() / S1.item() == pytest.approx(4.0, rel=1e-5)

    @pytest.mark.parametrize(
        "U_fn, name", [(cold_start, "cold"), (hot_start, "hot")]
    )
    def test_fermion_force_matches_gradient(self, U_fn, name):
        """Wilson fermion force matches numerical and autograd gradients."""
        from qcd_ml.qcd.dirac import dirac_wilson

        from src.HMCpy.fermion import wilson_fermion_force

        torch.manual_seed(123)
        U = U_fn()
        D = dirac_wilson(U, mass_parameter=0.1)

        chi = torch.randn(L, L, L, L, 4, NC, dtype=DTYPE) + 1j * torch.randn(
            L, L, L, L, 4, NC, dtype=DTYPE
        )
        chi = chi / chi.norm()
        psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
        F = wilson_fermion_force(U, psi, D)

        mu, site = 0, (0, 0, 0, 0)
        num_kw = {"maxiter": 2000, "eps": 1e-8, "inner_iter": 20}
        aut_kw = {"maxiter": 2000, "tol": 1e-10}

        def D_factory(U_in):
            return dirac_wilson(U_in, mass_parameter=0.1)

        for a in range(8):
            num = _numerical_force_comp(
                U, chi, D_factory, mu, site, a, GMRES_kwargs=num_kw
            )
            aut = _autograd_force_comp(
                U, chi, D_factory, mu, site, a, CG_kwargs=aut_kw
            )
            ana = _analytic_force_comp(F, mu, site, a)

            assert num == pytest.approx(
                ana, abs=1e-8, rel=1e-2
            ), f"[{name}] num={num:.6e} != ana={ana:.6e} (a={a})"
            assert aut == pytest.approx(
                ana, abs=1e-8, rel=1e-2
            ), f"[{name}] aut={aut:.6e} != ana={ana:.6e} (a={a})"
