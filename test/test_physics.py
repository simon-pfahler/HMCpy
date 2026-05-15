"""Tests for HMCpy.physics module: gauge action, plaquette, force, reunitarize, energy."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test.utils import (
    BETA,
    DTYPE,
    NC,
    SEED,
    L,
    cold_start,
    hot_start,
    random_momenta,
)

from HMCpy.monte_carlo import sample_momenta
from HMCpy.physics import (
    _plaquette,
    gauge_force,
    hamiltonian,
    kinetic_energy,
    plaquette_average,
    reunitarize,
    wilson_gauge_action,
)
from HMCpy.utility import gell_mann_matrices
from test.utils import apply_gauge_transform, random_SU3


class TestPhysics:
    # --- Wilson gauge action ---

    def test_action_cold_start_zero(self):
        """Cold start (identity links) has zero gauge action."""
        assert wilson_gauge_action(cold_start(), BETA).item() == pytest.approx(
            0.0, abs=1e-12
        )

    def test_action_non_negative(self):
        """Gauge action is always >= 0."""
        assert wilson_gauge_action(hot_start(), BETA).item() >= -1e-10

    def test_action_scales_with_beta(self):
        """Action is linear in beta."""
        U = hot_start()
        S1 = wilson_gauge_action(U, BETA)
        S2 = wilson_gauge_action(U, 2 * BETA)
        assert S2.item() == pytest.approx(2 * S1.item(), rel=1e-12)

    def test_plaquette_avg_cold_is_one(self):
        """Cold start plaquette average is exactly 1."""
        assert plaquette_average(cold_start()).item() == pytest.approx(
            1.0, abs=1e-12
        )

    def test_plaquette_avg_hot_near_zero(self):
        """Hot start plaquette average is close to 0."""
        p = plaquette_average(hot_start()).item()
        assert -0.1 <= p <= 0.1

    def test_action_consistent_with_plaquette(self):
        """Action formula: S = beta * V * 6 * (1 - <plaq>)."""
        U = hot_start()
        p = plaquette_average(U)
        V, n_pairs = L**4, 6
        S_expected = BETA * V * n_pairs * (1.0 - p.item())
        S_actual = wilson_gauge_action(U, BETA).item()
        assert S_actual == pytest.approx(S_expected, rel=1e-10)

    # --- Plaquette properties ---

    @pytest.mark.parametrize(
        "mu,nv", [(m, n) for m in range(4) for n in range(m + 1, 4)]
    )
    def test_plaquette_unitary_cold(self, mu, nv):
        """Cold start plaquettes satisfy P P^dag = I."""
        P = _plaquette(cold_start(), mu, nv)
        eye = torch.eye(NC, dtype=DTYPE).expand_as(P)
        PdagP = P @ P.conj().transpose(-1, -2)
        assert torch.allclose(PdagP, eye, atol=1e-12)

    @pytest.mark.parametrize(
        "mu,nv", [(m, n) for m in range(4) for n in range(m + 1, 4)]
    )
    def test_plaquette_unitary_hot(self, mu, nv):
        """Hot start plaquettes are in SU(3): P P^dag = I, det(P) = 1."""
        P = _plaquette(hot_start(), mu, nv)
        eye = torch.eye(NC, dtype=DTYPE).expand_as(P)
        PdagP = P @ P.conj().transpose(-1, -2)
        assert torch.allclose(PdagP, eye, atol=1e-10)
        assert torch.allclose(
            torch.linalg.det(P).abs(),
            torch.ones_like(torch.linalg.det(P)).abs(),
            atol=1e-10,
        )

    @pytest.mark.parametrize(
        "mu,nv", [(m, n) for m in range(4) for n in range(m + 1, 4)]
    )
    def test_plaquette_antisymmetry(self, mu, nv):
        """P(mu,nv) = P(nv,mu)^dag."""
        U = hot_start()
        P_mn = _plaquette(U, mu, nv)
        P_nm = _plaquette(U, nv, mu)
        assert torch.allclose(P_mn, P_nm.conj().transpose(-1, -2), atol=1e-10)

    # --- Gauge force algebraic properties ---

    def test_force_hermitian(self):
        """Gauge force is Hermitian everywhere."""
        F = gauge_force(hot_start(), BETA)
        assert torch.allclose(
            F - F.conj().transpose(-1, -2), torch.zeros_like(F), atol=1e-11
        )

    def test_force_traceless(self):
        """Gauge force is traceless at every site."""
        traces = torch.einsum("...ii->...", gauge_force(hot_start(), BETA))
        assert torch.allclose(traces, torch.zeros_like(traces), atol=1e-11)

    def test_force_cold_start_zero(self):
        """Gauge force vanishes for identity links."""
        assert torch.allclose(
            gauge_force(cold_start(), BETA),
            torch.zeros_like(cold_start()),
            atol=1e-12,
        )

    # --- Gauge force gradient check ---

    def _num_force_comp(self, U, mu, site, a, eps=1e-5):
        gen = 0.5 * gell_mann_matrices[a]
        exp_p = torch.linalg.matrix_exp(1j * eps * gen)
        exp_m = torch.linalg.matrix_exp(-1j * eps * gen)
        idx = (mu,) + site
        U_p, U_m = U.clone(), U.clone()
        U_p[idx], U_m[idx] = exp_p @ U[idx], exp_m @ U[idx]
        S_p = wilson_gauge_action(U_p, BETA).item()
        S_m = wilson_gauge_action(U_m, BETA).item()
        return (S_p - S_m) / (2 * eps)

    def _ana_force_comp(self, F, mu, site, a):
        return -(
            2 * torch.trace(0.5 * gell_mann_matrices[a] @ F[(mu,) + site])
        ).real.item()

    @pytest.mark.parametrize("mu", range(4))
    def test_force_equals_gradient(self, mu):
        """Gauge force matches numerical gradient at all sites and generators."""
        U = hot_start()
        F = gauge_force(U, BETA)
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    for t in range(L):
                        site = (x, y, z, t)
                        for a in range(8):
                            num = self._num_force_comp(U, mu, site, a)
                            ana = self._ana_force_comp(F, mu, site, a)
                            assert num == pytest.approx(ana, abs=1e-7, rel=1e-5)

    def test_force_taylor_expansion(self):
        """First-order Taylor expansion: S(U+dU) ~ S(U) + Tr[dU * dS/dU]."""
        U = hot_start()
        F = gauge_force(U, BETA)
        S0 = wilson_gauge_action(U, BETA).item()

        eps, mu, site, a = 1e-5, 0, (0, 0, 0, 0), 0
        idx = (mu,) + site
        gen = 0.5 * gell_mann_matrices[a]

        U_pert = U.clone()
        U_pert[idx] = torch.linalg.matrix_exp(1j * eps * gen) @ U[idx]
        S_pert = wilson_gauge_action(U_pert, BETA).item()

        dS_num = S_pert - S0
        dS_ana = self._ana_force_comp(F, mu, site, a) * eps

        assert dS_num == pytest.approx(dS_ana, rel=1e-4)

    # --- Reunitarization ---

    def test_reunitarize_identity(self):
        """Reunitarizing identity returns identity."""
        assert torch.allclose(
            reunitarize(cold_start()), cold_start(), atol=1e-12
        )

    def test_reunitarize_restores_unitary(self):
        """Reunitarization restores U U^dag = I."""
        torch.manual_seed(SEED)
        U_noisy = hot_start() + 0.1 * torch.randn_like(hot_start())
        U_r = reunitarize(U_noisy)
        eye = torch.eye(NC, dtype=DTYPE).expand_as(U_r)
        UUdag = U_r @ U_r.conj().transpose(-1, -2)
        assert torch.allclose(UUdag, eye, atol=1e-10)

    def test_reunitarize_det_one(self):
        """Reunitarized links have det = 1."""
        torch.manual_seed(SEED)
        U_r = reunitarize(hot_start() + 0.1 * torch.randn_like(hot_start()))
        assert torch.allclose(
            torch.linalg.det(U_r),
            torch.ones_like(torch.linalg.det(U_r)),
            atol=1e-10,
        )

    def test_reunitarize_preserves_su3(self):
        """Reunitarizing valid SU(3) field is a no-op."""
        assert torch.allclose(reunitarize(hot_start()), hot_start(), atol=1e-10)

    # --- Kinetic energy ---

    def test_kinetic_energy_non_negative(self):
        """Kinetic energy is always >= 0."""
        assert kinetic_energy(random_momenta(hot_start())).item() >= -1e-12

    def test_kinetic_energy_zero_at_zero(self):
        """Kinetic energy at zero momenta is 0."""
        assert kinetic_energy(
            torch.zeros(4, L, L, L, L, NC, NC, dtype=DTYPE)
        ).item() == pytest.approx(0.0)

    def test_kinetic_energy_scales_quadratically(self):
        """T(alpha * P) = alpha^2 * T(P)."""
        P = random_momenta(hot_start())
        alpha = 2.5
        T1 = kinetic_energy(P)
        T2 = kinetic_energy(alpha * P)
        assert T2.item() == pytest.approx(alpha**2 * T1.item(), rel=1e-12)

    def test_hamiltonian_is_T_plus_S(self):
        """Hamiltonian = kinetic + potential energy."""
        U, P = hot_start(), random_momenta(hot_start())
        H = hamiltonian(U, P, BETA)
        T = kinetic_energy(P)
        S = wilson_gauge_action(U, BETA)
        assert H.item() == pytest.approx((T + S).item(), rel=1e-12)

    # --- Gauge transformation properties ---

    def test_gauge_force_transforms_correctly(self):
        r"""Gauge force transforms as F_μ(x) -> Omega(x) F_μ(x) Omega^\dag(x)."""
        torch.manual_seed(12345)
        U = hot_start()

        # Compute original gauge force
        F_original = gauge_force(U, BETA)

        # Generate random gauge transformation
        Omega = random_SU3(seed=999)
        Omega_field = Omega.expand(*U.shape[1:5], NC, NC)

        # Apply gauge transformation to U
        U_transformed = apply_gauge_transform(U, Omega_field)

        # Compute transformed gauge force
        F_transformed = gauge_force(U_transformed, BETA)

        # Expected: F_μ(x) -> Omega(x) F_μ(x) Omega^\dag(x)
        Omega_dag = Omega_field.conj().transpose(-1, -2)
        F_expected = torch.zeros_like(F_transformed)
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
                atol=1e-12,
            ), f"Gauge force at mu={mu} does not transform correctly"
