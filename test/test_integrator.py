"""
test_integrator.py -- Tests for HMCpy.integrator module.

Run with: pytest test_integrator.py -v

Tests cover:
  - Time reversibility for leapfrog and OMF4
  - Energy conservation scaling (order verification)
  - SU(3) preservation during integration
"""

import os
import sys

import pytest
import torch

# Make the package importable when running from the test directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from HMCpy.integrator import _exp_update_U, leapfrog, omf4
from HMCpy.monte_carlo import sample_momenta
from HMCpy.physics import gauge_force, hamiltonian

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42
L = 2  # lattice side length
NC = 3  # SU(3)
BETA = 6.0
DTYPE = torch.complex128


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


def _random_momenta(U: torch.Tensor, seed=SEED) -> torch.Tensor:
    """Random traceless Hermitian momenta."""
    torch.manual_seed(seed)

    return sample_momenta(U)


# ===========================================================================
# TestIntegrator
# ===========================================================================


class TestIntegrator:
    """Tests for integrator.py."""

    def _test_U_update(self, U, P, eps):
        return U + eps * P

    def _test_force(self, U):
        return -U

    def _test_hamiltonian(self, U, P):
        return (P**2) / 2 + torch.sum(U**2) / 2

    def _force(self, U):
        return gauge_force(U, BETA)

    # -----------------------------------------------------------------------
    # Time-reversibility
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "integrator_fn,name",
        [
            (leapfrog, "leapfrog"),
        ],
    )
    def test_time_reversibility_HO(self, integrator_fn, name):
        """
        Symplectic integrators are time-reversible:
            (U', P') = integrate(U, P)
            (U'', P'') = integrate(U', -P')
        must give (U'', P'') == (U, -P) up to floating-point precision.
        This test checks this for a 1D harmonic oscillator.
        """
        U0 = torch.ones(1, dtype=torch.double)
        P0 = torch.zeros(1, dtype=torch.double)
        n_steps, step_size = 4, 0.05

        U1, P1 = integrator_fn(
            U0, P0, n_steps, self._test_force, step_size, self._test_U_update
        )
        U2, P2 = integrator_fn(
            U1, -P1, n_steps, self._test_force, step_size, self._test_U_update
        )

        assert torch.allclose(U2, U0, atol=1e-9), (
            f"[{name}] time-reversibility broken for U; "
            f"max |U''- U0| = {(U2 - U0).abs().max().item():.2e}"
        )
        assert torch.allclose(P2, -P0, atol=1e-9), (
            f"[{name}] time-reversibility broken for P; "
            f"max |P'' + P0| = {(P2 + P0).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize(
        "integrator_fn,name",
        [
            (leapfrog, "leapfrog"),
            (omf4, "omf4"),
        ],
    )
    def test_time_reversibility_MD(self, integrator_fn, name):
        """
        Symplectic integrators are time-reversible:
            (U', P') = integrate(U, P)
            (U'', P'') = integrate(U', -P')
        must give (U'', P'') == (U, -P) up to floating-point precision.
        This test checks this for the MD simulation in HMC.
        """
        U0 = _hot_start()
        P0 = _random_momenta(U0)
        n_steps, step_size = 10, 1

        U1, P1 = integrator_fn(U0, P0, n_steps, self._force, step_size)
        U2, P2 = integrator_fn(U1, -P1, n_steps, self._force, step_size)

        assert torch.allclose(U2, U0, atol=1e-9), (
            f"[{name}] time-reversibility broken for U; "
            f"max |U''- U0| = {(U2 - U0).abs().max().item():.2e}"
        )
        assert torch.allclose(P2, -P0, atol=1e-9), (
            f"[{name}] time-reversibility broken for P; "
            f"max |P'' + P0| = {(P2 + P0).abs().max().item():.2e}"
        )

    # -----------------------------------------------------------------------
    # Energy conservation: delta_H must scale correctly with step size
    # -----------------------------------------------------------------------

    def _get_max_diff(self, integrator_fn, step_size, time, test_system):
        U = torch.ones(1)
        P = torch.zeros(1)
        if not test_system:
            U = _hot_start()
            P = _random_momenta(U)
        N = round(time / step_size)
        Hs = torch.zeros(N)
        for _ in range(N):
            if test_system:
                Hs[_] = self._test_hamiltonian(U, P)
                U, P = integrator_fn(
                    U, P, 1, self._test_force, step_size, self._test_U_update
                )
            else:
                Hs[_] = hamiltonian(U, P, BETA)
                U, P = integrator_fn(U, P, 1, self._force, step_size)
        return (Hs.max() - Hs.min()).abs()

    @pytest.mark.parametrize(
        "integrator_fn,name_integrator",
        [
            (leapfrog, "leapfrog"),
            (omf4, "omf4"),
        ],
    )
    @pytest.mark.parametrize(
        "test_system,name_system",
        [
            (True, "Harmonic Oscillator"),
            (False, "Molecular Dynamics"),
        ],
    )
    def test_integrator_energy_drift(
        self, integrator_fn, name_integrator, test_system, name_system
    ):
        """
        For leapfrog, delta_H = O(eps^2).
        For OMF4, delta_H = O(eps^4).
        Halving the step size must reduce delta_H by ~4x for 2nd-order methods
        and by ~16x for 4th-order methods.
        """

        fine_step_size = 1
        target_ratio = 1
        if name_integrator in {"leapfrog"}:
            target_ratio = 4
            fine_step_size = 0.05 if test_system else 0.01
        if name_integrator in {"omf4"}:
            target_ratio = 16
            fine_step_size = 0.4 if test_system else 0.1

        coarse_step_size = 2 * fine_step_size
        dH_coarse = self._get_max_diff(
            integrator_fn, coarse_step_size, 10, test_system
        )
        dH_fine = self._get_max_diff(
            integrator_fn, fine_step_size, 10, test_system
        )
        ratio = dH_coarse / dH_fine

        assert 0.67 * target_ratio < ratio < 1.5 * target_ratio, (
            f"[{name_integrator}, {name_system}] not "
            f"{target_ratio**0.5:.0f}th order: "
            f"dH(coarse)={dH_coarse:.3e}, dH(fine)={dH_fine:.3e}, "
            f"ratio={ratio:.2f} (expected ~{target_ratio})"
        )

    # -----------------------------------------------------------------------
    # Link update stays on SU(3)
    # -----------------------------------------------------------------------

    def test_exp_update_stays_on_su3(self):
        """
        exp(i * eps * P) U must remain in SU(3) (unitary, det=1) for
        su(3)-valued P.
        """
        U = _hot_start()
        P = _random_momenta(U)
        U_new = _exp_update_U(U, P, 0.1)

        eye = torch.eye(NC, dtype=DTYPE)
        UUdag = torch.matmul(U_new, U_new.conj().transpose(-1, -2))
        assert torch.allclose(
            UUdag, eye.expand_as(UUdag), atol=1e-10
        ), "Link after exp update not unitary"
        det = torch.linalg.det(U_new)
        assert torch.allclose(
            det.abs(), torch.ones_like(det.abs()), atol=1e-10
        ), "Link after exp update has |det| != 1"

    def test_leapfrog_links_stay_on_su3(self):
        """After a leapfrog trajectory all links must remain in SU(3)."""
        U = _hot_start()
        P = _random_momenta(U)
        U_new, _ = leapfrog(U, P, n_steps=5, force=self._force, step_size=0.05)

        eye = torch.eye(NC, dtype=DTYPE)
        UUdag = torch.matmul(U_new, U_new.conj().transpose(-1, -2))
        assert torch.allclose(
            UUdag, eye.expand_as(UUdag), atol=1e-8
        ), f"Links left SU(3) after leapfrog; max |UU\u2020-I| = {(UUdag - eye.expand_as(UUdag)).abs().max().item():.2e}"
