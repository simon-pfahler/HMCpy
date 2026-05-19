"""Tests for HMCpy.integrator module: time-reversibility, energy conservation, SU(3) preservation."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test.conftest import hot_start, random_momenta

from HMCpy.integrator import _exp_update_U, leapfrog, omf4
from HMCpy.physics import gauge_force, hamiltonian


class TestIntegrator:
    def _HO_force(self, U):
        return -U

    def _HO_H(self, U, P):
        return (P**2) / 2 + torch.sum(U**2) / 2

    def _MD_force(self, U, beta):
        return gauge_force(U, beta)

    def _get_max_H_diff(self, integrator, eps, time, use_HO, beta):
        if use_HO:
            U, P = torch.ones(1, dtype=torch.double), torch.zeros(
                1, dtype=torch.double
            )
            H_fn, force_fn, update_fn = (
                self._HO_H,
                self._HO_force,
                self._HO_update,
            )
        else:
            U, P = hot_start(), random_momenta(hot_start())
            H_fn = lambda U, P: hamiltonian(U, P, beta)
            force_fn, update_fn = lambda U: self._MD_force(U, beta), None
        N = round(time / eps)
        Hs = torch.zeros(N)
        for i in range(N):
            Hs[i] = H_fn(U, P)
            if update_fn:
                U, P = integrator(U, P, 1, force_fn, eps, update_fn)
            else:
                U, P = integrator(U, P, 1, force_fn, eps)
        return (Hs.max() - Hs.min()).abs()

    # --- Time-reversibility ---

    def _HO_update(self, U, P, eps):
        return U + eps * P

    @pytest.mark.parametrize("integrator,name", [(leapfrog, "leapfrog")])
    def test_time_reversibility_HO(self, integrator, name):
        """Symplectic integrator is time-reversible for harmonic oscillator."""
        U0, P0 = torch.ones(1, dtype=torch.double), torch.zeros(
            1, dtype=torch.double
        )
        U1, P1 = integrator(U0, P0, 4, self._HO_force, 0.05, self._HO_update)
        U2, P2 = integrator(U1, -P1, 4, self._HO_force, 0.05, self._HO_update)
        assert torch.allclose(U2, U0, atol=1e-9)
        assert torch.allclose(P2, -P0, atol=1e-9)

    @pytest.mark.parametrize(
        "integrator,name", [(leapfrog, "leapfrog"), (omf4, "omf4")]
    )
    def test_time_reversibility_MD(self, integrator, name, beta):
        """Symplectic integrator is time-reversible for MD simulation."""
        U0, P0 = hot_start(), random_momenta(hot_start())
        U1, P1 = integrator(U0, P0, 10, lambda U: self._MD_force(U, beta), 1)
        U2, P2 = integrator(U1, -P1, 10, lambda U: self._MD_force(U, beta), 1)
        assert torch.allclose(U2, U0, atol=1e-9)
        assert torch.allclose(P2, -P0, atol=1e-9)

    # --- Energy conservation (order verification) ---

    @pytest.mark.parametrize("integrator,order", [(leapfrog, 2), (omf4, 4)])
    @pytest.mark.parametrize("use_HO,name", [(True, "HO"), (False, "MD")])
    def test_energy_drift_order(self, integrator, order, use_HO, name, beta):
        """Energy drift scales as O(eps^order) for order-th method."""
        target = 4 if order == 2 else 16
        eps_fine = 0.05 if (order == 2 and use_HO) else (0.4 if use_HO else 0.1)
        eps_coarse = 2 * eps_fine
        dH_coarse = self._get_max_H_diff(integrator, eps_coarse, 10, use_HO, beta)
        dH_fine = self._get_max_H_diff(integrator, eps_fine, 10, use_HO, beta)
        ratio = dH_coarse / dH_fine
        assert (
            0.67 * target < ratio < 1.5 * target
        ), f"[{integrator.__name__},{name}] ratio={ratio:.2f}, expected ~{target}"

    # --- SU(3) preservation ---

    def test_exp_update_stays_on_su3(self, NC, dtype):
        """exp(i * eps * P) * U remains in SU(3)."""
        U_new = _exp_update_U(hot_start(), random_momenta(hot_start()), 0.1)
        eye = torch.eye(NC, dtype=dtype).expand_as(U_new)
        UUdag = U_new @ U_new.conj().transpose(-1, -2)
        assert torch.allclose(UUdag, eye, atol=1e-10)
        assert torch.allclose(
            torch.linalg.det(U_new).abs(),
            torch.ones_like(torch.linalg.det(U_new)).abs(),
            atol=1e-10,
        )

    def test_leapfrog_stays_on_su3(self, beta, NC, dtype):
        """Leapfrog integration preserves SU(3)."""
        U_new, _ = leapfrog(
            hot_start(), random_momenta(hot_start()), 5, lambda U: self._MD_force(U, beta), 0.05
        )
        eye = torch.eye(NC, dtype=dtype).expand_as(U_new)
        UUdag = U_new @ U_new.conj().transpose(-1, -2)
        assert torch.allclose(UUdag, eye, atol=1e-8)
