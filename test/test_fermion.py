"""Tests for HMCpy.fermion module: pseudofermion action, derivatives, force."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test.conftest import (
    cold_start,
    conjugate_gradient,
    hot_start,
)
from typing import Callable

from qcd_ml.base.operations import v_spin_const_transform

from HMCpy.fermion import pseudofermion_action
from HMCpy.utility import gell_mann_matrices
from src.HMCpy.fermion import (
    apply_DDdag_inv,
    gamma5,
    wilson_clover_fermion_force,
)

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
            return D(
                v_spin_const_transform(
                    gamma5, D(v_spin_const_transform(gamma5, psi))
                )
            )

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
            return D(
                v_spin_const_transform(
                    gamma5, D(v_spin_const_transform(gamma5, psi))
                )
            )

        chi = conjugate_gradient(DDdag_op, phi, **(CG_kwargs or {}))
        return pseudofermion_action(phi, chi).real

    grad_full = torch.autograd.grad(S_pf(U_grad), U_grad, create_graph=False)[0]
    link_grad = (
        grad_full[idx]
        if grad_full is not None
        else torch.zeros(3, 3, dtype=torch.complex128)
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
    def test_pseudofermion_action_pos_finite(self, U_fn, name, mass, L, NC, dtype):
        """Pseudofermion action is positive and finite."""
        from qcd_ml.qcd.dirac import dirac_wilson

        U = U_fn()
        m = mass
        D = dirac_wilson(U, mass_parameter=m)
        # Normalized uniform phi: norm = sqrt(L^4 * 4 * NC) for ones
        phi = torch.ones(L, L, L, L, 4, NC, dtype=dtype)
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

    def test_pseudofermion_action_real(self, mass, L, NC, dtype):
        """Pseudofermion action is real for Hermitian DDdagger."""
        from qcd_ml.qcd.dirac import dirac_wilson

        U = cold_start()
        D = dirac_wilson(U, mass_parameter=mass)
        phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        chi = apply_DDdag_inv(phi, D, GMRES_kwargs=GMRES_OPTS)

        assert (pseudofermion_action(phi, chi)).isreal()
        assert torch.isfinite(pseudofermion_action(phi, chi))

    def test_pseudofermion_action_scales_quadratically(self, mass, L, NC, dtype):
        """Action scales as phi^2 (S ~ phi^dag chi, chi ~ phi)."""
        from qcd_ml.qcd.dirac import dirac_wilson

        U = cold_start()
        D = dirac_wilson(U, mass_parameter=mass)
        phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)

        chi1 = apply_DDdag_inv(phi, D, GMRES_kwargs=GMRES_OPTS)
        chi2 = apply_DDdag_inv(2 * phi, D, GMRES_kwargs=GMRES_OPTS)

        S1 = pseudofermion_action(phi, chi1)
        S2 = pseudofermion_action(2 * phi, chi2)

        assert S2.item() / S1.item() == pytest.approx(4.0, rel=1e-5)

    @pytest.mark.parametrize(
        "U_fn, name", [(cold_start, "cold"), (hot_start, "hot")]
    )
    def test_fermion_force_matches_gradient(self, U_fn, name, mass, L, NC, dtype):
        """Wilson fermion force matches numerical and autograd gradients."""
        from qcd_ml.qcd.dirac import dirac_wilson

        from src.HMCpy.fermion import wilson_fermion_force

        torch.manual_seed(123)
        U = U_fn()
        D = dirac_wilson(U, mass_parameter=mass)

        chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype) + 1j * torch.randn(
            L, L, L, L, 4, NC, dtype=dtype
        )
        chi = chi / chi.norm()
        psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
        F = wilson_fermion_force(U, psi, D)

        mu, site = 0, (0, 0, 0, 0)
        num_kw = {"maxiter": 2000, "eps": 1e-8, "inner_iter": 20}
        aut_kw = {"maxiter": 2000, "tol": 1e-10}

        def D_factory(U_in):
            return dirac_wilson(U_in, mass_parameter=mass)

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

    @pytest.mark.parametrize(
        "U_fn, name", [(cold_start, "cold"), (hot_start, "hot")]
    )
    def test_wilson_force_is_hermitian(self, U_fn, name, mass, L, NC, dtype):
        """Wilson fermion force is Hermitian: F = F^dag."""
        from qcd_ml.qcd.dirac import dirac_wilson

        from src.HMCpy.fermion import wilson_fermion_force

        torch.manual_seed(100)
        U = U_fn()
        D = dirac_wilson(U, mass_parameter=mass)

        chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        chi = chi / chi.norm()
        psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
        F = wilson_fermion_force(U, psi, D)

        for mu in range(4):
            assert torch.allclose(
                F[mu], F[mu].conj().transpose(-1, -2), atol=1e-12
            ), f"[{name}] Wilson force not Hermitian at mu={mu}"

    @pytest.mark.parametrize(
        "U_fn, name", [(cold_start, "cold"), (hot_start, "hot")]
    )
    def test_wilson_force_is_traceless(self, U_fn, name, mass, L, NC, dtype):
        """Wilson fermion force is traceless at every link."""
        from qcd_ml.qcd.dirac import dirac_wilson

        from src.HMCpy.fermion import wilson_fermion_force

        torch.manual_seed(101)
        U = U_fn()
        D = dirac_wilson(U, mass_parameter=mass)

        chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        chi = chi / chi.norm()
        psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
        F = wilson_fermion_force(U, psi, D)

        for mu in range(4):
            trace = torch.einsum("...ii->...", F[mu])
            assert torch.allclose(
                trace, torch.zeros_like(trace), atol=1e-12
            ), f"[{name}] Wilson force not traceless at mu={mu}"


class TestCloverFermion:
    """Tests for Wilson-Clover fermion force."""

    @pytest.mark.parametrize(
        "U_fn, name", [(cold_start, "cold"), (hot_start, "hot")]
    )
    def test_clover_force_matches_wilson_at_csw_zero(self, U_fn, name, mass, L, NC, dtype):
        """Wilson-Clover force with csw=0 matches Wilson force."""
        from qcd_ml.qcd.dirac import dirac_wilson, dirac_wilson_clover

        from src.HMCpy.fermion import wilson_fermion_force

        torch.manual_seed(123)
        U = U_fn()
        m = mass
        csw = 0.0

        chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        chi = chi / chi.norm()

        # Compute Wilson force
        D_wilson = dirac_wilson(U, mass_parameter=m)
        psi_wilson = apply_DDdag_inv(chi, D_wilson, GMRES_kwargs=GMRES_OPTS)
        F_wilson = wilson_fermion_force(U, psi_wilson, D_wilson)

        # Compute Wilson-Clover force with csw=0
        D_clover = dirac_wilson_clover(U, mass_parameter=m, csw=csw)
        psi_clover = apply_DDdag_inv(chi, D_clover, GMRES_kwargs=GMRES_OPTS)
        F_clover = wilson_clover_fermion_force(U, psi_clover, D_clover, csw)

        # They should be equal
        assert torch.allclose(
            F_wilson, F_clover, atol=1e-12, rtol=1e-12
        ), f"[{name}] Forces differ by {(F_wilson - F_clover).abs().max().item():.6e}"

    @pytest.mark.parametrize(
        "U_fn, name", [(cold_start, "cold"), (hot_start, "hot")]
    )
    def test_clover_force_matches_gradient(self, U_fn, name, mass, L, NC, dtype):
        """Wilson-Clover fermion force matches numerical and autograd gradients."""
        from qcd_ml.qcd.dirac import dirac_wilson_clover

        torch.manual_seed(456)
        U = U_fn()
        m = mass
        csw = 1.0

        D = dirac_wilson_clover(U, mass_parameter=m, csw=csw)

        chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        chi = chi / chi.norm()
        psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
        F = wilson_clover_fermion_force(U, psi, D, csw)

        mu, site = 0, (0, 0, 0, 0)
        num_kw = {"maxiter": 2000, "eps": 1e-8, "inner_iter": 20}
        aut_kw = {"maxiter": 2000, "tol": 1e-10}

        def D_factory(U_in):
            return dirac_wilson_clover(U_in, mass_parameter=m, csw=csw)

        for a in range(8):
            num = _numerical_force_comp(
                U, chi, D_factory, mu, site, a, GMRES_kwargs=num_kw
            )
            aut = _autograd_force_comp(
                U, chi, D_factory, mu, site, a, CG_kwargs=aut_kw
            )
            ana = _analytic_force_comp(F, mu, site, a)

            assert num == pytest.approx(
                ana, abs=1e-6, rel=1e-2
            ), f"[{name}] num={num:.6e} != ana={ana:.6e} (a={a})"
            assert aut == pytest.approx(
                ana, abs=1e-6, rel=1e-2
            ), f"[{name}] aut={aut:.6e} != ana={ana:.6e} (a={a})"

    @pytest.mark.parametrize(
        "U_fn, name", [(cold_start, "cold"), (hot_start, "hot")]
    )
    def test_clover_force_shape_and_properties(self, U_fn, name, mass, L, NC, dtype):
        """Wilson-Clover force has correct shape and is traceless Hermitian."""
        from qcd_ml.qcd.dirac import dirac_wilson_clover

        torch.manual_seed(789)
        U = U_fn()
        m = mass
        csw = 1.5

        D = dirac_wilson_clover(U, mass_parameter=m, csw=csw)

        chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        chi = chi / chi.norm()
        psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
        F = wilson_clover_fermion_force(U, psi, D, csw)

        # Check shape
        assert (
            F.shape == U.shape
        ), f"[{name}] Force shape {F.shape} != U shape {U.shape}"

        # Check traceless
        for mu in range(4):
            trace = torch.einsum("...ii->...", F[mu])
            assert torch.allclose(
                trace, torch.zeros_like(trace), atol=1e-12
            ), f"[{name}] Force not traceless at mu={mu}"

        # Check Hermitian: F = F^dag
        for mu in range(4):
            assert torch.allclose(
                F[mu], F[mu].conj().transpose(-1, -2), atol=1e-12
            ), f"[{name}] Force not Hermitian at mu={mu}"

        # Check finite
        assert torch.all(
            torch.isfinite(F)
        ), f"[{name}] Force has non-finite values"

    @pytest.mark.parametrize(
        "U_fn, name", [(cold_start, "cold"), (hot_start, "hot")]
    )
    def test_clover_force_is_hermitian(self, U_fn, name, mass, L, NC, dtype):
        """Wilson-Clover fermion force is Hermitian: F = F^dag."""
        from qcd_ml.qcd.dirac import dirac_wilson_clover

        torch.manual_seed(790)
        U = U_fn()
        m = mass
        csw = 1.5

        D = dirac_wilson_clover(U, mass_parameter=m, csw=csw)

        chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        chi = chi / chi.norm()
        psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
        F = wilson_clover_fermion_force(U, psi, D, csw)

        for mu in range(4):
            assert torch.allclose(
                F[mu], F[mu].conj().transpose(-1, -2), atol=1e-12
            ), f"[{name}] Wilson-Clover force not Hermitian at mu={mu}"

    @pytest.mark.parametrize(
        "U_fn, name", [(cold_start, "cold"), (hot_start, "hot")]
    )
    def test_clover_force_is_traceless(self, U_fn, name, mass, L, NC, dtype):
        """Wilson-Clover fermion force is traceless at every link."""
        from qcd_ml.qcd.dirac import dirac_wilson_clover

        torch.manual_seed(791)
        U = U_fn()
        m = mass
        csw = 1.5

        D = dirac_wilson_clover(U, mass_parameter=m, csw=csw)

        chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        chi = chi / chi.norm()
        psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
        F = wilson_clover_fermion_force(U, psi, D, csw)

        for mu in range(4):
            trace = torch.einsum("...ii->...", F[mu])
            assert torch.allclose(
                trace, torch.zeros_like(trace), atol=1e-12
            ), f"[{name}] Wilson-Clover force not traceless at mu={mu}"


class TestGaugeTransformation:
    """Tests for gauge transformation properties of forces."""

    def test_wilson_fermion_force_transforms_correctly(self, mass, L, NC, dtype):
        r"""Wilson fermion force transforms as F_μ(x) -> Omega(x) F_μ(x) Omega^\dag(x)."""
        from test.conftest import apply_gauge_transform, random_SU3

        from qcd_ml.qcd.dirac import dirac_wilson

        from src.HMCpy.fermion import wilson_fermion_force

        torch.manual_seed(12346)
        U = hot_start()
        m = mass

        D = dirac_wilson(U, mass_parameter=m)

        chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        chi = chi / chi.norm()
        psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)

        # Compute original Wilson fermion force
        F_original = wilson_fermion_force(U, psi, D)

        # Generate random gauge transformation
        Omega = random_SU3(seed=1000)
        Omega_field = Omega.expand(*U.shape[1:5], NC, NC)

        # Transform psi: psi(x, s, c) -> Omega(x, c, c') * psi(x, s, c')
        # Result: psi_transformed(x, s, c)
        psi_transformed = torch.einsum("...ac,...sc->...sa", Omega_field, psi)

        # Apply gauge transformation to U
        U_transformed = apply_gauge_transform(U, Omega_field)

        # Create new Dirac operator with transformed U
        D_transformed = dirac_wilson(U_transformed, mass_parameter=m)

        # Compute transformed Wilson fermion force
        F_transformed = wilson_fermion_force(
            U_transformed, psi_transformed, D_transformed
        )

        # Expected: F_μ(x) -> Omega(x) F_μ(x) Omega^\dag(x)
        F_expected = torch.zeros_like(F_transformed)
        Omega_dag = Omega_field.conj().transpose(-1, -2)
        for mu in range(4):
            F_expected[mu] = torch.einsum(
                "...ab,...bc,...cd->...ad",
                Omega_field,
                F_original[mu],
                Omega_dag,
            )

        # Check transformation
        for mu in range(4):
            assert torch.allclose(
                F_transformed[mu],
                F_expected[mu],
                atol=1e-10,
            ), f"Wilson fermion force at mu={mu} does not transform correctly"

    def test_wilson_clover_fermion_force_transforms_correctly(self, mass, L, NC, dtype):
        r"""Wilson-Clover fermion force transforms as F_μ(x) -> Omega(x) F_μ(x) Omega^\dag(x)."""
        from test.conftest import apply_gauge_transform, random_SU3

        from qcd_ml.qcd.dirac import dirac_wilson_clover

        torch.manual_seed(12347)
        U = hot_start()
        m = mass
        csw = 1.0

        D = dirac_wilson_clover(U, mass_parameter=m, csw=csw)

        chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        chi = chi / chi.norm()
        psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)

        # Compute original Wilson-Clover fermion force
        F_original = wilson_clover_fermion_force(U, psi, D, csw)

        # Generate random gauge transformation
        Omega = random_SU3(seed=1001)
        Omega_field = Omega.expand(*U.shape[1:5], NC, NC)

        # Transform psi: psi(x, s, c) -> Omega(x, c, c') * psi(x, s, c')
        # Result: psi_transformed(x, s, c)
        psi_transformed = torch.einsum("...ac,...sc->...sa", Omega_field, psi)

        # Apply gauge transformation to U
        U_transformed = apply_gauge_transform(U, Omega_field)

        # Create new Dirac operator with transformed U
        D_transformed = dirac_wilson_clover(
            U_transformed, mass_parameter=m, csw=csw
        )

        # Compute transformed Wilson-Clover fermion force
        F_transformed = wilson_clover_fermion_force(
            U_transformed, psi_transformed, D_transformed, csw
        )

        # Expected: F_μ(x) -> Omega(x) F_μ(x) Omega^\dag(x)
        F_expected = torch.zeros_like(F_transformed)
        Omega_dag = Omega_field.conj().transpose(-1, -2)
        for mu in range(4):
            F_expected[mu] = torch.einsum(
                "...ab,...bc,...cd->...ad",
                Omega_field,
                F_original[mu],
                Omega_dag,
            )

        # Check transformation
        for mu in range(4):
            assert torch.allclose(
                F_transformed[mu],
                F_expected[mu],
                atol=1e-10,
            ), f"Wilson-Clover fermion force at mu={mu} does not transform correctly"
