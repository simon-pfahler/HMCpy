"""
test_fermion.py -- Tests for HMCpy.fermion module.

Run with: pytest test_fermion.py -v

Tests cover:
  - Pseudofermion action computation
  - Wilson Dirac operator derivative
  - Wilson-Clover Dirac operator derivative
  - Fermion force gradient checks (autodiff vs analytic)
  - Finite difference tests
"""

import os
import sys

import pytest
import torch

# Make the package importable when running from the test directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from HMCpy.fermion import pseudofermion_action
from HMCpy.utility import gell_mann_matrices
from src.HMCpy.fermion import apply_DDdag_inv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42
L = 2  # lattice side length
NC = 3  # SU(3)
DTYPE = torch.complex128


def _numerical_fermion_force_component(
    U: torch.Tensor,
    phi: torch.Tensor,
    D_op_factory: callable,
    mu: int,
    site: tuple,
    a: int,
    eps: float = 1e-5,
    GMRES_kwargs: dict | None = None,
) -> float:
    """
    Numerical directional derivative of pseudofermion action S_pf with respect
    to U_mu(site) along the su(3) generator i*lambda_a (central finite difference).

    S_pf = phi^dag (DD^dag)^{-1} phi

    Parameters
    ----------
    U : gauge field [4, Lx, Ly, Lz, Lt, Nc, Nc]
    phi : pseudofermion field [Lx, Ly, Lz, Lt, Ns, Nc]
    D_op_factory : factory function that takes U and returns the Dirac operator D
    mu : direction index (0-3)
    site : tuple of 4 coordinates (x, y, z, t)
    a : Gell-Mann matrix index (0-7)
    eps : finite difference step size
    GMRES_kwargs : optional dict of keyword arguments for GMRES solver

    Returns
    -------
    dS/da : numerical derivative of S_pf w.r.t. the generator direction
    """
    from qcd_ml.util.solver import GMRES

    from src.HMCpy.fermion import apply_gamma5

    gen = gell_mann_matrices[a]  # lambda_a
    exp_p = torch.linalg.matrix_exp(1j * eps * gen)
    exp_m = torch.linalg.matrix_exp(-1j * eps * gen)

    idx = (mu,) + site

    U_plus = U.clone()
    U_plus[idx] = exp_p @ U[idx]

    U_minus = U.clone()
    U_minus[idx] = exp_m @ U[idx]

    # Create Dirac operators for perturbed gauge fields
    D_plus = D_op_factory(U_plus)
    D_minus = D_op_factory(U_minus)

    # Local DDdag solver with zero initial guess for better robustness
    from src.HMCpy.fermion import gamma5

    def solve_DDdag_inv(phi_in, D_in, GMRES_kwargs_in):
        def DDdag_op(psi):
            gamma5_psi = torch.einsum("ij,...jc->...ic", gamma5, psi)
            D_gamma5_psi = D_in(gamma5_psi)
            Ddag_psi = torch.einsum("ij,...jc->...ic", gamma5, D_gamma5_psi)
            return D_in(Ddag_psi)

        # Use zero initial guess for robustness
        kwargs = GMRES_kwargs_in or {}
        chi, _ = GMRES(DDdag_op, phi_in, torch.zeros_like(phi_in), **kwargs)
        return chi

    # Compute chi for perturbed fields
    chi_plus = solve_DDdag_inv(phi, D_plus, GMRES_kwargs)
    chi_minus = solve_DDdag_inv(phi, D_minus, GMRES_kwargs)

    # Compute pseudofermion action for both
    S_plus = pseudofermion_action(phi, chi_plus).item()
    S_minus = pseudofermion_action(phi, chi_minus).item()

    return (S_plus - S_minus) / (2 * eps)


def _analytic_fermion_force_component(
    F: torch.Tensor, mu: int, site: tuple, a: int
) -> float:
    """
    Analytic directional derivative from the fermion force tensor.
    """
    idx = (mu,) + site
    gen = gell_mann_matrices[a]
    val = torch.trace(gen @ F[idx])
    return -2 * val.item()


def _cold_start(L=L, Nc=NC) -> torch.Tensor:
    """All links = identity."""
    return torch.eye(Nc, dtype=DTYPE).expand(4, L, L, L, L, Nc, Nc).clone()


def _hot_start(L=L, Nc=NC, seed=SEED) -> torch.Tensor:
    """All links = random SU(3) via QR decomposition."""
    g = torch.Generator()
    g.manual_seed(seed)
    X = torch.randn(4, L, L, L, L, Nc, Nc, dtype=DTYPE, generator=g)
    Q, _ = torch.linalg.qr(X)
    det = torch.linalg.det(Q)
    phase = det / det.abs()
    phase_root = torch.exp(torch.log(phase + 1e-30j) / Nc)
    return (Q / phase_root.unsqueeze(-1).unsqueeze(-1)).detach()


# ===========================================================================
# TestFermion
# ===========================================================================


class TestFermion:
    """Tests for fermion.py: pseudofermion action, derivatives, force."""

    # -----------------------------------------------------------------------
    # Pseudofermion action
    # -----------------------------------------------------------------------

    def test_pseudofermion_action_cold_start(self):
        """
        For cold start (all links = identity) and Wilson Dirac operator,
        the action should be positive.
        """
        from qcd_ml.qcd.dirac import dirac_wilson

        U = _cold_start()
        D_op = dirac_wilson(U, mass_parameter=0.1)

        phi = torch.randn(L, L, L, L, 4, NC, dtype=DTYPE)
        chi = apply_DDdag_inv(
            phi,
            D_op,
            GMRES_kwargs={"maxiter": 1000, "eps": 1e-10, "inner_iter": 10},
        )
        S_pf = pseudofermion_action(phi, chi)

        assert S_pf.item() > 0, f"Action should be positive, got {S_pf.item()}"
        assert torch.isfinite(S_pf), "Action should be finite"

    def test_pseudofermion_action_with_chi(self):
        """
        Test that providing chi directly works and gives the same result.
        Now chi is the solution to (DD^dag) chi = phi, not D chi = phi.
        """
        from qcd_ml.qcd.dirac import dirac_wilson

        U = _hot_start()  # Use hot start to avoid GMRES convergence issues
        D_op = dirac_wilson(U, mass_parameter=0.1)

        phi = torch.randn(L, L, L, L, 4, NC, dtype=DTYPE)

        # Compute chi: solve (DD^dag) chi = phi
        chi = apply_DDdag_inv(
            phi,
            D_op,
            GMRES_kwargs={"maxiter": 1000, "eps": 1e-10, "inner_iter": 10},
        )

        # Verify: (DD^dag) chi should equal phi
        from HMCpy.fermion import apply_gamma5

        gamma5_chi = apply_gamma5(chi)
        D_gamma5_chi = D_op(gamma5_chi)
        Ddag_chi = apply_gamma5(D_gamma5_chi)
        DDdag_chi = D_op(Ddag_chi)
        residual = (DDdag_chi - phi).norm().item()
        assert (
            residual < 5e-6
        ), f"DDdag solve did not converge: residual = {residual}"

        # Compute action with chi
        S_pf = pseudofermion_action(phi, chi)

        assert S_pf.item() > 0, f"Action should be positive, got {S_pf.item()}"
        assert torch.isfinite(S_pf), "Action should be finite"

    def test_pseudofermion_action_hermitian(self):
        """
        For a Hermitian Dirac operator, the action should be real.

        Note: This test uses a non-trivial operator to avoid a bug in qcd_ml's
        GMRES solver that occurs when the solver converges in 1 iteration.
        """
        U = _cold_start()

        # Use Wilson Dirac operator from qcd_ml which is non-trivial
        from qcd_ml.qcd.dirac import dirac_wilson

        D_op = dirac_wilson(U, mass_parameter=0.1)

        phi = torch.randn(L, L, L, L, 4, NC, dtype=DTYPE)
        chi = apply_DDdag_inv(
            phi,
            D_op,
            GMRES_kwargs={"maxiter": 1000, "eps": 1e-10, "inner_iter": 10},
        )
        S_pf = pseudofermion_action(phi, chi)

        assert S_pf.isreal(), "Action should be real"
        assert torch.isfinite(S_pf), "Action should be finite"

    def test_pseudofermion_action_scales_with_phi(self):
        """
        S_pf = phi^dag chi should scale linearly with phi (since chi scales linearly with phi).

        Note: Uses Wilson Dirac operator to avoid GMRES convergence bug.
        """
        U = _cold_start()

        from qcd_ml.qcd.dirac import dirac_wilson

        D_op = dirac_wilson(U, mass_parameter=0.1)

        phi1 = torch.randn(L, L, L, L, 4, NC, dtype=DTYPE)
        phi2 = 2.0 * phi1

        chi1 = apply_DDdag_inv(
            phi1,
            D_op,
            GMRES_kwargs={"maxiter": 1000, "eps": 1e-10, "inner_iter": 10},
        )
        chi2 = apply_DDdag_inv(
            phi2,
            D_op,
            GMRES_kwargs={"maxiter": 1000, "eps": 1e-10, "inner_iter": 10},
        )

        S1 = pseudofermion_action(phi1, chi1)
        S2 = pseudofermion_action(phi2, chi2)

        # S2 = phi2^dag chi2 = (2 phi1)^dag (2 chi1) = 4 phi1^dag chi1 = 4 S1
        # So S2 should be approximately 4 * S1 (quadratic scaling in the field)
        ratio = S2.item() / S1.item()
        assert ratio == pytest.approx(
            4.0, rel=1e-5
        ), f"Action should scale quadratically: S2/S1 = {ratio:.6f}, expected 4.0"

    # -----------------------------------------------------------------------
    # Fermion force gradient check (numerical vs analytic)
    # -----------------------------------------------------------------------

    def test_fermion_force_equals_action_gradient(self):
        """
        Verify that wilson_fermion_force matches the numerical gradient
        of pseudofermion_action along all 8 Gell-Mann directions,
        at one lattice site and direction.

        Note: This test may fail if the analytic force implementation is wrong.
        The test uses cold start (identity links) for stability.
        """
        from qcd_ml.qcd.dirac import dirac_wilson

        from src.HMCpy.fermion import wilson_fermion_force

        # Use cold start for gauge field (identity links) - more stable for testing
        U = _cold_start()

        # Create the Dirac operator factory
        def D_op_factory(U_in):
            return dirac_wilson(U_in, mass_parameter=0.1)

        # Create the original Dirac operator and compute chi
        D_op = D_op_factory(U)
        # Use a fixed, non-zero phi with reasonable magnitude
        torch.manual_seed(123)
        chi = torch.randn(L, L, L, L, 4, NC, dtype=DTYPE) + 1j * torch.randn(
            L, L, L, L, 4, NC, dtype=DTYPE
        )
        chi = chi / chi.norm()  # Normalize and scale to reasonable size
        phi = D_op(chi)
        psi = apply_DDdag_inv(
            phi,
            D_op,
            GMRES_kwargs={"maxiter": 1000, "eps": 1e-10, "inner_iter": 10},
        )

        # Compute analytic fermion force
        F = wilson_fermion_force(U, psi, D_op)

        # Test at one site and direction
        mu = 0
        site = (0, 0, 0, 0)

        GMRES_kwargs = {"maxiter": 2000, "eps": 1e-8, "inner_iter": 20}

        print(f"\nAnalytic force at mu={mu} site={site}:")
        for a in range(8):
            ana = _analytic_fermion_force_component(F, mu, site, a)
            print(f"  generator {a}: analytic={ana:.8f}")

        print(f"\nNumerical derivatives at mu={mu} site={site}:")
        for a in range(8):
            num = _numerical_fermion_force_component(
                U,
                phi,
                D_op_factory,
                mu,
                site,
                a,
                eps=1e-4,
                GMRES_kwargs=GMRES_kwargs,
            )
            ana = _analytic_fermion_force_component(F, mu, site, a)

            print(f"  generator {a}: numerical={num:.8f} analytic={ana:.8f}")

            assert num == pytest.approx(ana, abs=1e-2, rel=1e-2), (
                f"Fermion force mismatch at mu={mu} site={site} generator={a}: "
                f"numerical={num:.8f} analytic={ana:.8f}"
            )
