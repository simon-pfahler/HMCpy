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
from qcd_ml.qcd.dirac import dirac_wilson, dirac_wilson_clover
from qcd_ml.util.solver import GMRES

from HMCpy.fermion import (
    apply_DDdag_inv,
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

    return (
        (phi.conj() * chi_p).sum().real - (phi.conj() * chi_m).sum().real
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

            chi = torch.randn(L, L, L, L, 4, NC, dtype=torch.complex128)
            chi = chi / chi.norm()
            psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
            F = wilson_fermion_force(U, psi, D)

            def D_factory(U_in):
                return dirac_wilson(U_in, mass_parameter=mass)

        else:  # clover

            torch.manual_seed(456)
            U = U_fn()
            csw = 1.0
            D = dirac_wilson_clover(U, mass_parameter=mass, csw=csw)

            chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
            chi = chi / chi.norm()
            psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)

            F = wilson_clover_fermion_force(U, psi, D, csw)

            def D_factory(U_in):
                return dirac_wilson_clover(U_in, mass_parameter=mass, csw=csw)

        mu, site = 0, (0, 0, 0, 0)
        num_kw = {"maxiter": 2000, "eps": 1e-8, "inner_iter": 20}
        aut_kw = {"maxiter": 2000, "tol": 1e-10}

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
            chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
            chi = chi / chi.norm()
            psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
            F = wilson_fermion_force(U, psi, D)
        else:  # clover

            torch.manual_seed(790)
            U = U_fn()
            csw = 1.5
            D = dirac_wilson_clover(U, mass_parameter=mass, csw=csw)
            chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
            chi = chi / chi.norm()
            psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
            F = wilson_clover_fermion_force(U, psi, D, csw)

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
        ), f"Forces differ by {(F_wilson - F_clover).abs().max().item():.6e}"

    @pytest.mark.parametrize("fermion_type", ["wilson", "clover"])
    @pytest.mark.parametrize("U_fn", [cold_start, hot_start])
    def test_force_shape(self, fermion_type, U_fn, mass, L, NC, dtype):
        """Fermion force has correct shape and finite values."""
        if fermion_type == "wilson":

            torch.manual_seed(789)
            U = U_fn()
            D = dirac_wilson(U, mass_parameter=mass)
            chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
            chi = chi / chi.norm()
            psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
            F = wilson_fermion_force(U, psi, D)
        else:  # clover

            torch.manual_seed(789)
            U = U_fn()
            csw = 1.5
            D = dirac_wilson_clover(U, mass_parameter=mass, csw=csw)
            chi = torch.randn(L, L, L, L, 4, NC, dtype=dtype)
            chi = chi / chi.norm()
            psi = apply_DDdag_inv(chi, D, GMRES_kwargs=GMRES_OPTS)
            F = wilson_clover_fermion_force(U, psi, D, csw)

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
