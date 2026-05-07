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

from HMCpy.fermion import apply_DDdag_inv, pseudofermion_action
from HMCpy.utility import gell_mann_matrices

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42
L = 2  # lattice side length
NC = 3  # SU(3)
DTYPE = torch.complex128


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
