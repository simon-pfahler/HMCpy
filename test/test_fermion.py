"""Tests for HMCpy.fermion module: pseudofermion action, derivatives, force."""

from typing import Callable

import pytest
import torch
from conftest import (
    GMRES_OPTS,
    apply_gauge_transform,
    cold_start,
    conjugate_gradient,
    hot_start,
    random_SU3,
)
from qcd_ml.base.operations import v_spin_const_transform
from qcd_ml.qcd.dirac import (
    dirac_wilson,
    dirac_wilson_clover,
    dirac_wilson_clover_dag,
    dirac_wilson_dag,
)
from qcd_ml.util.solver import GMRES

from HMCpy.fermion import (
    gamma5,
    wilson_clover_fermion_force,
    wilson_fermion_force,
)
from HMCpy.utility import su3_generators


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

    gen = su3_generators[a]
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

    return ((phi.conj() * (chi_p - chi_m)).sum().real.item()) / (2 * eps)


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
    gen = su3_generators[a]
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
        return (phi.conj() * chi).sum().real

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
    return -(2 * torch.trace(su3_generators[a] @ F[(mu,) + site])).real.item()


class TestFermionAction:
    """Tests for pseudofermion action."""

    @pytest.mark.parametrize("U_fn", [cold_start, hot_start])
    def test_pseudofermion_action_pos_finite(self, U_fn, mass, L, NC, dtype):
        """Pseudofermion action is positive and finite."""

        U = U_fn()
        m = mass
        D = dirac_wilson(U, mass_parameter=m)
        # Normalized uniform phi: norm = sqrt(L^4 * 4 * NC) for ones
        phi = torch.ones(L, L, L, L, 4, NC, dtype=dtype)
        phi_norm = phi.norm()
        phi = phi / phi_norm

        # Compute chi = (DD^dag)^{-1} phi using DDdag operator
        gamma5_device = gamma5.to(phi.device)

        def DDdag_op(psi_in: torch.Tensor) -> torch.Tensor:
            """Operator: (DD^dag) psi = D (gamma5 @ D (gamma5 @ psi))"""
            gamma5_psi = v_spin_const_transform(gamma5_device, psi_in)
            D_gamma5_psi = D(gamma5_psi)
            Ddag_psi = v_spin_const_transform(gamma5_device, D_gamma5_psi)
            return D(Ddag_psi)

        chi, _ = GMRES(DDdag_op, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
        S = (phi.conj() * chi).sum().real

        assert S.item() > 0, f"Action should be positive, got {S.item()}"
        assert torch.isfinite(S), "Action should be finite"

        if U_fn.__name__ == "cold_start":
            assert S == pytest.approx(
                1 / m**2, abs=1e-2
            ), f"Expected 1/m^2=100, got {S.item()}"

    def test_pseudofermion_action_real(self, mass, L, NC, dtype):
        """Pseudofermion action is real for Hermitian DDdagger."""

        U = cold_start()
        D = dirac_wilson(U, mass_parameter=mass)
        phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)

        # Compute chi = (DD^dag)^{-1} phi
        gamma5_device = gamma5.to(phi.device)

        def DDdag_op(psi_in: torch.Tensor) -> torch.Tensor:
            gamma5_psi = v_spin_const_transform(gamma5_device, psi_in)
            D_gamma5_psi = D(gamma5_psi)
            Ddag_psi = v_spin_const_transform(gamma5_device, D_gamma5_psi)
            return D(Ddag_psi)

        chi, _ = GMRES(DDdag_op, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
        S = (phi.conj() * chi).sum().real

        assert S.isreal()
        assert torch.isfinite(S)

    def test_pseudofermion_action_scales_quadratically(
        self, mass, L, NC, dtype
    ):
        """Action scales as phi^2 (S ~ phi^dag chi, chi ~ phi)."""

        U = cold_start()
        D = dirac_wilson(U, mass_parameter=mass)
        phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)

        # Compute DDdag operator
        gamma5_device = gamma5.to(phi.device)

        def DDdag_op(psi_in: torch.Tensor) -> torch.Tensor:
            gamma5_psi = v_spin_const_transform(gamma5_device, psi_in)
            D_gamma5_psi = D(gamma5_psi)
            Ddag_psi = v_spin_const_transform(gamma5_device, D_gamma5_psi)
            return D(Ddag_psi)

        chi1, _ = GMRES(DDdag_op, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
        chi2, _ = GMRES(DDdag_op, (2 * phi).clone(), (2 * phi).clone(), **(GMRES_OPTS or {}))

        S1 = (phi.conj() * chi1).sum().real
        S2 = ((2 * phi).conj() * chi2).sum().real

        assert S2.item() / S1.item() == pytest.approx(4.0, rel=1e-5)


class TestFermionForce:
    """Tests for fermion forces."""

    @pytest.mark.parametrize("fermion_type", ["wilson", "clover"])
    @pytest.mark.parametrize("U_fn", [cold_start, hot_start])
    def test_fermion_force_matches_gradient(
        self, fermion_type, U_fn, mass, L, NC, dtype
    ):
        """Fermion force matches numerical and autograd gradients."""
        if fermion_type == "wilson":

            torch.manual_seed(123)
            U = U_fn()
            D = dirac_wilson(U, mass_parameter=mass)
            Ddag = dirac_wilson_dag(U, mass_parameter=mass)

            phi = torch.randn(L, L, L, L, 4, NC, dtype=torch.complex128)
            phi = phi / phi.norm()
            # New interface: psi = D^{-1} phi, Ddag_inv_psi = (D^dag)^{-1} psi = (DD^dag)^{-1} phi
            psi, _ = GMRES(D, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
            Ddag_inv_psi, _ = GMRES(Ddag, psi.clone(), psi.clone(), **(GMRES_OPTS or {}))
            F = wilson_fermion_force(U, psi, Ddag_inv_psi)

            def D_factory(U_in):
                return dirac_wilson(U_in, mass_parameter=mass)

        else:  # clover

            torch.manual_seed(456)
            U = U_fn()
            m = mass
            csw = 1.0
            D = dirac_wilson_clover(U, mass_parameter=m, csw=csw)
            Ddag = dirac_wilson_clover_dag(U, mass_parameter=m, csw=csw)

            phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
            phi = phi / phi.norm()
            psi, _ = GMRES(D, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
            Ddag_inv_psi, _ = GMRES(Ddag, psi.clone(), psi.clone(), **(GMRES_OPTS or {}))

            F = wilson_clover_fermion_force(U, psi, Ddag_inv_psi, csw)

            def D_factory(U_in):
                return dirac_wilson_clover(U_in, mass_parameter=m, csw=csw)

        mu, site = 0, (0, 0, 0, 0)
        num_kw = {"maxiter": 2000, "eps": 1e-8, "inner_iter": 20}
        aut_kw = {"maxiter": 2000, "tol": 1e-10}

        for a in range(8):
            num = _numerical_force_comp(
                U, phi, D_factory, mu, site, a, GMRES_kwargs=num_kw
            )
            aut = _autograd_force_comp(
                U, phi, D_factory, mu, site, a, CG_kwargs=aut_kw
            )
            ana = _analytic_force_comp(F, mu, site, a)

            assert num == pytest.approx(
                ana, abs=1e-8, rel=1e-2
            ), f"num={num:.6e} != ana={ana:.6e} (a={a})"
            assert aut == pytest.approx(
                ana, abs=1e-8, rel=1e-2
            ), f"aut={aut:.6e} != ana={ana:.6e} (a={a})"

    @pytest.mark.parametrize("fermion_type", ["wilson", "clover"])
    @pytest.mark.parametrize("U_fn", [cold_start, hot_start])
    def test_fermion_force_properties(
        self, fermion_type, U_fn, mass, L, NC, dtype
    ):
        """Fermion force has correct properties (Hermitian and traceless)."""
        if fermion_type == "wilson":

            torch.manual_seed(100)
            U = U_fn()
            D = dirac_wilson(U, mass_parameter=mass)
            Ddag = dirac_wilson_dag(U, mass_parameter=mass)
            phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
            phi = phi / phi.norm()
            psi, _ = GMRES(D, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
            Ddag_inv_psi, _ = GMRES(Ddag, psi.clone(), psi.clone(), **(GMRES_OPTS or {}))
            F = wilson_fermion_force(U, psi, Ddag_inv_psi)
        else:  # clover

            torch.manual_seed(790)
            U = U_fn()
            m = mass
            csw = 1.5
            D = dirac_wilson_clover(U, mass_parameter=m, csw=csw)
            Ddag = dirac_wilson_clover_dag(U, mass_parameter=m, csw=csw)
            phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
            phi = phi / phi.norm()
            psi, _ = GMRES(D, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
            Ddag_inv_psi, _ = GMRES(Ddag, psi.clone(), psi.clone(), **(GMRES_OPTS or {}))
            F = wilson_clover_fermion_force(U, psi, Ddag_inv_psi, csw)

        # Check Hermitian: F = F^dag
        for mu in range(4):
            assert torch.allclose(
                F[mu], F[mu].adjoint(), atol=1e-12
            ), f"Force not Hermitian at mu={mu}"

        # Check traceless
        for mu in range(4):
            trace = torch.einsum("...ii->...", F[mu])
            assert torch.allclose(
                trace, torch.zeros_like(trace), atol=1e-12
            ), f"Force not traceless at mu={mu}"

    @pytest.mark.parametrize("U_fn", [cold_start, hot_start])
    def test_clover_force_matches_wilson_at_csw_zero(
        self, U_fn, mass, L, NC, dtype
    ):
        """Wilson-Clover force with csw=0 matches Wilson force."""

        torch.manual_seed(123)
        U = U_fn()
        m = mass
        csw = 0.0

        phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        phi = phi / phi.norm()

        # Compute Wilson force
        D_wilson = dirac_wilson(U, mass_parameter=m)
        Ddag_wilson = dirac_wilson_dag(U, mass_parameter=m)
        psi_wilson, _ = GMRES(D_wilson, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
        Ddag_inv_psi_wilson, _ = GMRES(Ddag_wilson, psi_wilson.clone(), psi_wilson.clone(), **(GMRES_OPTS or {}))
        F_wilson = wilson_fermion_force(U, psi_wilson, Ddag_inv_psi_wilson)

        # Compute Wilson-Clover force with csw=0
        D_clover = dirac_wilson_clover(U, mass_parameter=m, csw=csw)
        Ddag_clover = dirac_wilson_clover_dag(U, mass_parameter=m, csw=csw)
        psi_clover, _ = GMRES(D_clover, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
        Ddag_inv_psi_clover, _ = GMRES(Ddag_clover, psi_clover.clone(), psi_clover.clone(), **(GMRES_OPTS or {}))
        F_clover = wilson_clover_fermion_force(U, psi_clover, Ddag_inv_psi_clover, csw)

        # They should be equal
        assert torch.allclose(
            F_wilson, F_clover, atol=1e-12, rtol=1e-12
        ), f"Forces differ by {(F_wilson - F_clover).abs().max().item():.6e}"

    @pytest.mark.parametrize("fermion_type", ["wilson", "clover"])
    @pytest.mark.parametrize("U_fn", [cold_start, hot_start])
    def test_force_shape(self, fermion_type, U_fn, mass, L, NC, dtype):
        """Fermion force has correct shape and finite values."""
        if fermion_type == "wilson":

            torch.manual_seed(789)
            U = U_fn()
            D = dirac_wilson(U, mass_parameter=mass)
            Ddag = dirac_wilson_dag(U, mass_parameter=mass)
            phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
            phi = phi / phi.norm()
            psi, _ = GMRES(D, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
            Ddag_inv_psi, _ = GMRES(Ddag, psi.clone(), psi.clone(), **(GMRES_OPTS or {}))
            F = wilson_fermion_force(U, psi, Ddag_inv_psi)
        else:  # clover

            torch.manual_seed(789)
            U = U_fn()
            m = mass
            csw = 1.5
            D = dirac_wilson_clover(U, mass_parameter=m, csw=csw)
            Ddag = dirac_wilson_clover_dag(U, mass_parameter=m, csw=csw)
            phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
            phi = phi / phi.norm()
            psi, _ = GMRES(D, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
            Ddag_inv_psi, _ = GMRES(Ddag, psi.clone(), psi.clone(), **(GMRES_OPTS or {}))
            F = wilson_clover_fermion_force(U, psi, Ddag_inv_psi, csw)

        # Check shape
        assert F.shape == U.shape, f"Force shape {F.shape} != U shape {U.shape}"

        # Check finite
        assert torch.all(torch.isfinite(F)), f"Force has non-finite values"


class TestGaugeTransformation:
    """Tests for gauge transformation properties of forces."""

    def test_wilson_fermion_force_transforms_correctly(
        self, mass, L, NC, dtype
    ):
        r"""Wilson fermion force transforms as F_μ(x) -> Omega(x) F_μ(x) Omega^dag(x)."""
        torch.manual_seed(12346)
        U = hot_start()
        m = mass

        D = dirac_wilson(U, mass_parameter=m)
        Ddag = dirac_wilson_dag(U, mass_parameter=m)

        phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        phi = phi / phi.norm()
        psi, _ = GMRES(D, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
        Ddag_inv_psi, _ = GMRES(Ddag, psi.clone(), psi.clone(), **(GMRES_OPTS or {}))

        # Compute original Wilson fermion force
        F_original = wilson_fermion_force(U, psi, Ddag_inv_psi)

        # Generate random gauge transformation
        Omega = random_SU3(seed=1000)
        Omega_field = Omega.expand(*U.shape[1:5], NC, NC)

        # Transform psi and Ddag_inv_psi: psi(x, s, c) -> Omega(x, c, c') * psi(x, s, c')
        psi_transformed = torch.einsum("...ac,...sc->...sa", Omega_field, psi)
        Ddag_inv_psi_transformed = torch.einsum("...ac,...sc->...sa", Omega_field, Ddag_inv_psi)

        # Apply gauge transformation to U
        U_transformed = apply_gauge_transform(U, Omega_field)

        # Create new Dirac operator with transformed U
        D_transformed = dirac_wilson(U_transformed, mass_parameter=m)
        Ddag_transformed = dirac_wilson_dag(U_transformed, mass_parameter=m)

        # Compute transformed Wilson fermion force
        F_transformed = wilson_fermion_force(
            U_transformed, psi_transformed, Ddag_inv_psi_transformed
        )

        # Expected: F_μ(x) -> Omega(x) F_μ(x) Omega^dag(x)
        F_expected = torch.zeros_like(F_transformed)
        Omega_dag = Omega_field.adjoint()
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

    def test_wilson_clover_fermion_force_transforms_correctly(
        self, mass, L, NC, dtype
    ):
        r"""Wilson-Clover fermion force transforms as F_μ(x) -> Omega(x) F_μ(x) Omega^dag(x)."""
        torch.manual_seed(12347)
        U = hot_start()
        m = mass
        csw = 1.0

        D = dirac_wilson_clover(U, mass_parameter=m, csw=csw)
        Ddag = dirac_wilson_clover_dag(U, mass_parameter=m, csw=csw)

        phi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
        phi = phi / phi.norm()
        psi, _ = GMRES(D, phi.clone(), phi.clone(), **(GMRES_OPTS or {}))
        Ddag_inv_psi, _ = GMRES(Ddag, psi.clone(), psi.clone(), **(GMRES_OPTS or {}))

        # Compute original Wilson-Clover fermion force
        F_original = wilson_clover_fermion_force(U, psi, Ddag_inv_psi, csw)

        # Generate random gauge transformation
        Omega = random_SU3(seed=1001)
        Omega_field = Omega.expand(*U.shape[1:5], NC, NC)

        # Transform psi and Ddag_inv_psi: psi(x, s, c) -> Omega(x, c, c') * psi(x, s, c')
        psi_transformed = torch.einsum("...ac,...sc->...sa", Omega_field, psi)
        Ddag_inv_psi_transformed = torch.einsum("...ac,...sc->...sa", Omega_field, Ddag_inv_psi)

        # Apply gauge transformation to U
        U_transformed = apply_gauge_transform(U, Omega_field)

        # Create new Dirac operator with transformed U
        D_transformed = dirac_wilson_clover(
            U_transformed, mass_parameter=m, csw=csw
        )
        Ddag_transformed = dirac_wilson_clover_dag(
            U_transformed, mass_parameter=m, csw=csw
        )

        # Compute transformed Wilson-Clover fermion force
        F_transformed = wilson_clover_fermion_force(
            U_transformed, psi_transformed, Ddag_inv_psi_transformed, csw
        )

        # Expected: F_μ(x) -> Omega(x) F_μ(x) Omega^dag(x)
        F_expected = torch.zeros_like(F_transformed)
        Omega_dag = Omega_field.adjoint()
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
