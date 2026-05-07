"""
test_physics.py -- Tests for HMCpy.physics module.

Run with: pytest test_physics.py -v

Tests cover:
  - Wilson gauge action
  - Plaquette computation
  - Gauge force
  - Reunitarization
  - Kinetic energy and Hamiltonian
"""

import os
import sys

import pytest
import torch

# Make the package importable when running from the test directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42
L = 2  # lattice side length
NC = 3  # SU(3)
BETA = 6.0
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


def _random_momenta(U: torch.Tensor, seed=SEED) -> torch.Tensor:
    """Random traceless Hermitian momenta."""
    torch.manual_seed(seed)

    return sample_momenta(U)


# ===========================================================================
# TestPhysics
# ===========================================================================


class TestPhysics:
    """Tests for physics.py: action, plaquette, force, reunitarize."""

    # -----------------------------------------------------------------------
    # Wilson gauge action -- analytic values
    # -----------------------------------------------------------------------

    def test_action_cold_start_is_zero(self):
        """
        Cold start: all plaquettes are identity, so Re Tr[1 - U_plaq] = 0
        everywhere and the action must be exactly zero.
        """
        U = _cold_start()
        S = wilson_gauge_action(U, BETA)
        assert S.item() == pytest.approx(0.0, abs=1e-12)

    def test_action_is_non_negative(self):
        """
        For any SU(3) matrix M, Re Tr[M] <= Nc, so Re Tr[1 - M] >= 0
        and S_W >= 0 always.
        """
        U = _hot_start()
        S = wilson_gauge_action(U, BETA)
        assert S.item() >= -1e-10

    def test_action_scales_with_beta(self):
        """S_W is linear in beta, so S(2*beta) = 2*S(beta)."""
        U = _hot_start()
        S1 = wilson_gauge_action(U, BETA)
        S2 = wilson_gauge_action(U, 2 * BETA)
        assert S2.item() == pytest.approx(2 * S1.item(), rel=1e-12)

    def test_action_cold_maximum_plaquette(self):
        """
        For the cold start the plaquette average must be exactly 1
        (Re Tr[I] / Nc = 1).
        """
        U = _cold_start()
        p = plaquette_average(U)
        assert p.item() == pytest.approx(1.0, abs=1e-12)

    def test_plaquette_average_hot_start_range(self):
        """
        For a random (hot) SU(3) gauge field, the mean plaquette
        Re Tr[U_plaq] / Nc should be close to 0 for each plaquette.
        The average over all plaquettes should therefore also be close to 0.
        """
        U = _hot_start()
        p = plaquette_average(U)
        assert -0.1 <= p.item() <= 0.1

    def test_action_consistent_with_plaquette(self):
        """
        S_W = (beta/Nc) * sum_{x,mu<nu} (Nc - Re Tr[P_munu])
             = beta * V * 6 * (1 - <plaq>)
        where V = L^4 and 6 = C(4,2).
        """
        U = _hot_start()
        V = L**4
        n_pairs = 6
        p = plaquette_average(U)
        S_expected = BETA * V * n_pairs * (1.0 - p.item())
        S_actual = wilson_gauge_action(U, BETA)
        assert S_actual.item() == pytest.approx(S_expected, rel=1e-10)

    # -----------------------------------------------------------------------
    # Plaquette matrix properties
    # -----------------------------------------------------------------------

    def test_plaquette_matrix_is_unitary_cold(self):
        """
        For the cold start every plaquette matrix is exactly the identity,
        so P P^dag = I at every site.
        """
        U = _cold_start()
        for mu in range(4):
            for nu in range(mu + 1, 4):
                P = _plaquette(U, mu, nu)
                eye = torch.eye(NC, dtype=DTYPE).expand_as(P)
                PdagP = torch.matmul(P.conj().transpose(-1, -2), P)
                assert torch.allclose(PdagP, eye, atol=1e-12)

    def test_plaquette_matrix_is_unitary_hot(self):
        """
        For any SU(3) field the plaquette matrix must be in SU(3):
        P P^dag = I  and  det(P) = 1.
        """
        U = _hot_start()
        eye = torch.eye(NC, dtype=DTYPE)
        for mu in range(4):
            for nu in range(mu + 1, 4):
                P = _plaquette(U, mu, nu)
                PdagP = torch.matmul(P.conj().transpose(-1, -2), P)
                assert torch.allclose(
                    PdagP,
                    eye.expand_as(PdagP),
                    atol=1e-10,
                ), f"Plaquette ({mu},{nu}) not unitary"
                det = torch.linalg.det(P)
                assert torch.allclose(
                    det.abs(),
                    torch.ones_like(det.abs()),
                    atol=1e-10,
                ), f"Plaquette ({mu},{nu}) det != 1"

    def test_plaquette_antisymmetry(self):
        """
        U_{mu,nu}(x) = U_{nu,mu}(x)^dag  (plaquette orientation reversal).
        """
        U = _hot_start()
        for mu in range(4):
            for nu in range(mu + 1, 4):
                P_mn = _plaquette(U, mu, nu)
                P_nm = _plaquette(U, nu, mu)
                assert torch.allclose(
                    P_mn,
                    P_nm.conj().transpose(-1, -2),
                    atol=1e-10,
                ), f"Plaquette antisymmetry failed for ({mu},{nu})"

    # -----------------------------------------------------------------------
    # Gauge force -- algebraic properties
    # -----------------------------------------------------------------------

    def test_force_is_hermitian(self):
        """
        F_mu(x) must satisfy  F = F^dag  (Hermitian) everywhere.
        """
        U = _hot_start()
        F = gauge_force(U, BETA)
        herm = F - F.conj().transpose(-1, -2)
        assert torch.allclose(
            herm, torch.zeros_like(herm), atol=1e-11
        ), f"Force not Hermitian; max |F-F\u2020| = {herm.abs().max().item():.2e}"

    def test_force_is_traceless(self):
        """
        F_mu(x) must be traceless: Tr[F_mu(x)] = 0 at every site.
        """
        U = _hot_start()
        F = gauge_force(U, BETA)
        traces = torch.einsum("...ii->...", F)
        assert torch.allclose(
            traces, torch.zeros_like(traces), atol=1e-11
        ), f"Force not traceless; max |Tr F| = {traces.abs().max().item():.2e}"

    def test_force_cold_start_is_zero(self):
        """
        For the cold start (all links = identity), the staple sum at each
        site is proportional to the identity, so Q = U * sigma is proportional
        to I, and the anti-Hermitian traceless projection vanishes: F = 0.
        """
        U = _cold_start()
        F = gauge_force(U, BETA)
        assert torch.allclose(
            F, torch.zeros_like(F), atol=1e-12
        ), f"Force on cold start not zero; max |F| = {F.abs().max().item():.2e}"

    # -----------------------------------------------------------------------
    # Gauge force -- finite-difference gradient check
    # -----------------------------------------------------------------------

    def _numerical_force_component(
        self, U: torch.Tensor, mu: int, site: tuple, a: int, eps: float = 1e-5
    ) -> float:
        """
        Numerical directional derivative of S_W with respect to U_mu(site)
        along the su(3) generator i*lambda_a (central finite difference).
        """
        gen = 0.5 * gell_mann_matrices[a]  # lambda_a
        exp_p = torch.linalg.matrix_exp(1j * eps * gen)
        exp_m = torch.linalg.matrix_exp(-1j * eps * gen)

        idx = (mu,) + site

        U_plus = U.clone()
        U_plus[idx] = exp_p @ U[idx]

        U_minus = U.clone()
        U_minus[idx] = exp_m @ U[idx]

        S_plus = wilson_gauge_action(U_plus, BETA).item()
        S_minus = wilson_gauge_action(U_minus, BETA).item()
        return (S_plus - S_minus) / (2 * eps)

    def _analytic_force_component(
        self, F: torch.Tensor, mu: int, site: tuple, a: int
    ) -> float:
        """
        Analytic directional derivative from the force tensor.
        """
        idx = (mu,) + site
        gen = 0.5 * gell_mann_matrices[a]
        val = 2 * torch.trace(gen @ F[idx])
        return -val.item()

    @pytest.mark.parametrize("mu", [0, 1, 2, 3])
    def test_force_equals_action_gradient(self, mu):
        """
        For each direction mu, verify that gauge_force matches the numerical
        gradient of wilson_gauge_action along all 8 Gell-Mann directions,
        at every lattice site.
        """
        U = _hot_start()
        F = gauge_force(U, BETA)

        for x in range(L):
            for y in range(L):
                for z in range(L):
                    for t in range(L):
                        site = (x, y, z, t)
                        for a in range(8):
                            num = self._numerical_force_component(
                                U, mu, site, a
                            )
                            ana = self._analytic_force_component(F, mu, site, a)
                            assert num == pytest.approx(
                                ana, abs=1e-7, rel=1e-5
                            ), (
                                f"Force mismatch at mu={mu} site={site} generator={a}: "
                                f"numerical={num:.8f} analytic={ana:.8f}"
                            )

    def test_force_single_link_perturbation(self):
        """
        Perturb one specific link by a small amount, recompute force and action,
        and verify the first-order Taylor expansion:
            S(U + dU) \u2248 S(U) + Tr[dU * dS/dU]
        """
        U = _hot_start()
        F = gauge_force(U, BETA)
        S0 = wilson_gauge_action(U, BETA)

        eps = 1e-5
        mu, site, a = 0, (0, 0, 0, 0), 0
        idx = (mu,) + site
        gen = 0.5 * gell_mann_matrices[a]
        exp_p = torch.linalg.matrix_exp(1j * eps * gen)

        U_pert = U.clone()
        U_pert[idx] = exp_p @ U[idx]
        S_pert = wilson_gauge_action(U_pert, BETA)

        dS_num = (S_pert - S0).item()
        dS_ana = self._analytic_force_component(F, mu, site, a) * eps

        assert dS_num == pytest.approx(
            dS_ana, rel=1e-4
        ), f"First-order Taylor check failed: dS_num={dS_num:.2e}, dS_ana={dS_ana:.2e}"

    # -----------------------------------------------------------------------
    # Reunitarization
    # -----------------------------------------------------------------------

    def test_reunitarize_identity(self):
        """Reunitarizing the cold start must return the identity."""
        U = _cold_start()
        U_r = reunitarize(U)
        assert torch.allclose(U_r, U, atol=1e-12)

    def test_reunitarize_restores_unitarity(self):
        """
        Perturb each link by adding a small random matrix (moving off SU(3)),
        then reunitarize and check U U^dag = I and det(U) = 1.
        """
        torch.manual_seed(SEED)
        U = _hot_start()
        noise = 0.1 * torch.randn_like(U)
        U_noisy = U + noise

        U_r = reunitarize(U_noisy)

        eye = torch.eye(NC, dtype=DTYPE)
        UUdag = torch.matmul(U_r, U_r.conj().transpose(-1, -2))
        assert torch.allclose(
            UUdag, eye.expand_as(UUdag), atol=1e-10
        ), f"Reunitarized U not unitary; max |UU\u2020-I| = {(UUdag - eye.expand_as(UUdag)).abs().max().item():.2e}"

    def test_reunitarize_determinant_is_one(self):
        """After reunitarization every link must have det = +1."""
        torch.manual_seed(SEED)
        U = _hot_start()
        noise = 0.1 * torch.randn_like(U)
        U_r = reunitarize(U + noise)

        det = torch.linalg.det(U_r)
        assert torch.allclose(
            det, torch.ones_like(det), atol=1e-10
        ), f"Reunitarized det not 1; max |det-1| = {(det - 1).abs().max().item():.2e}"

    def test_reunitarize_preserves_su3_field(self):
        """
        Reunitarizing an already-valid SU(3) field must be a no-op.
        """
        U = _hot_start()
        U_r = reunitarize(U)
        assert torch.allclose(
            U_r, U, atol=1e-10
        ), f"Reunitarize changed a valid SU(3) field; max diff = {(U_r - U).abs().max().item():.2e}"

    # -----------------------------------------------------------------------
    # Kinetic energy
    # -----------------------------------------------------------------------

    def test_kinetic_energy_non_negative(self):
        """T(P) = (1/2) Tr[P^dag P] >= 0 always."""
        U = _hot_start()
        torch.manual_seed(SEED)
        P = _random_momenta(U)
        T = kinetic_energy(P)
        assert T.item() >= -1e-12

    def test_kinetic_energy_zero_for_zero_momenta(self):
        """T(0) = 0."""
        U = _cold_start()
        P = torch.zeros_like(U)
        assert kinetic_energy(P).item() == pytest.approx(0.0, abs=1e-15)

    def test_kinetic_energy_scales_quadratically(self):
        """T(alpha * P) = alpha^2 * T(P)."""
        U = _hot_start()
        P = _random_momenta(U)
        alpha = 2.5
        T1 = kinetic_energy(P)
        T2 = kinetic_energy(alpha * P)
        assert T2.item() == pytest.approx(alpha**2 * T1.item(), rel=1e-12)

    def test_hamiltonian_is_sum_of_T_and_S(self):
        """H = T(P) + S_W(U) by definition."""
        U = _hot_start()
        P = _random_momenta(U)
        H = hamiltonian(U, P, BETA)
        T = kinetic_energy(P)
        S = wilson_gauge_action(U, BETA)
        assert H.item() == pytest.approx((T + S).item(), rel=1e-12)
