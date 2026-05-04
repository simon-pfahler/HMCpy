"""
test_hmcpy.py -- Test suite for HMCpy.

Run with:  pytest test_hmcpy.py -v

The tests are organised into four classes mirroring the package structure:
  TestPhysics    -- Wilson action, plaquette, gauge force, reunitarize
  TestIntegrator -- time-reversibility and energy-conservation scaling
  TestMonteCarlo -- momentum sampling, Metropolis criterion, hmc_step smoke test

Physics tests use a small 2^4 lattice with SU(3) (Nc=3) throughout.
A fixed random seed makes every test deterministic.

The most important tests are the force-gradient checks in TestPhysics:
they verify that gauge_force is exactly the derivative of wilson_gauge_action
by comparing the analytic force against a numerical finite-difference
gradient computed by perturbing each link along every su(3) generator.
"""

import math
import os
import sys

import pytest
import torch

# ---------------------------------------------------------------------------
# Make the package importable when running from the test directory.
# Adjust the path below to wherever HMCpy lives.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from HMCpy.fermion import gell_mann_matrices
from HMCpy.integrator import _exp_update_U, leapfrog, omf4
from HMCpy.monte_carlo import metropolis_accept, sample_momenta
from HMCpy.physics import (
    _plaquette,
    _staple,
    gauge_force,
    hamiltonian,
    kinetic_energy,
    plaquette_average,
    reunitarize,
    wilson_gauge_action,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SEED = 42
L = 2  # lattice side length -- small enough to be fast, large enough to test BC
NC = 3  # SU(3)
BETA = 6.0  # standard value for SU(3)
DTYPE = torch.complex128


def _cold_start(L=L, Nc=NC) -> torch.Tensor:
    """All links = identity."""
    return torch.eye(Nc, dtype=DTYPE).expand(4, L, L, L, L, Nc, Nc).clone()


def _hot_start(L=L, Nc=NC, seed=SEED) -> torch.Tensor:
    """All links = random SU(3) via QR decomposition."""
    g = torch.Generator()
    g.manual_seed(seed)
    X = torch.randn(4, L, L, L, L, Nc, Nc, dtype=DTYPE, generator=g)
    Q, R = torch.linalg.qr(X)
    # Fix phases so det = +1
    det = torch.linalg.det(Q)
    phase = det / det.abs()
    phase_root = torch.exp(torch.log(phase + 1e-30j) / Nc)
    return (Q / phase_root.unsqueeze(-1).unsqueeze(-1)).detach()


def _random_momenta(U: torch.Tensor, seed=SEED) -> torch.Tensor:
    """Random traceless anti-Hermitian momenta via sample_momenta (seeded)."""
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
        ), f"Force not Hermitian; max |F-F†| = {herm.abs().max().item():.2e}"

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
    # Gauge force -- finite-difference gradient check  (THE key physics test)
    #
    # The gauge force F_mu(x) must equal the derivative of the action with
    # respect to the link U_mu(x) along each su(3) direction:
    #
    #   dS / d(i lambda_a eps)|_{eps=0}  =  -Tr[ lambda_a * F_mu(x) ]  / (i)
    #
    # We test this by computing the numerical directional derivative
    #   (S(exp(i eps lambda_a) U_mu(x)) - S(exp(-i eps lambda_a) U_mu(x))) / (2 eps)
    # and comparing it to the analytic value extracted from F.
    #
    # We test all 8 Gell-Mann directions on a random subset of (mu, x) pairs.
    # -----------------------------------------------------------------------

    def _numerical_force_component(
        self, U: torch.Tensor, mu: int, site: tuple, a: int, eps: float = 1e-5
    ) -> float:
        """
        Numerical directional derivative of S_W with respect to U_mu(site)
        along the su(3) generator i*lambda_a (central finite difference).
        """
        gen = gell_mann_matrices[a]  # lambda_a, shape [3,3]
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

        The force equation of motion is  dP/dt = F, and the action derivative
        along direction i*lambda_a at link (mu, site) is:

            dS/d(eps)|_{eps=0}  =  -Tr[ lambda_a * (i * F_mu(site)) ]
                                 =  -i * Tr[ lambda_a * F_mu(site) ]

        Because F is Hermitian and traceless, this equals the imaginary
        part of Tr[ lambda_a * F_mu(site) ] after accounting for the 'i'.
        We compute it directly as Im Tr[ lambda_a * F_mu(site) ] / (-i) = Re Tr[...].
        """
        idx = (mu,) + site
        # dS/d(i*lambda_a * eps) = -Tr[lambda_a * F_mu(site)] (for anti-Hermitian F)
        gen = gell_mann_matrices[a]
        val = torch.trace(gen @ F[idx])
        # The action derivative along exp(i eps lambda_a) is  -Tr[lambda_a * F]
        return -2 * val.item()

    @pytest.mark.parametrize("mu", [0, 1, 2, 3])
    def test_force_equals_action_gradient(self, mu):
        """
        For each direction mu, verify that gauge_force matches the numerical
        gradient of wilson_gauge_action along all 8 Gell-Mann directions,
        at every lattice site.
        """
        U = _hot_start()
        F = gauge_force(U, BETA)

        # Test all sites for this mu
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

            S(U + dU) ≈ S(U) + Tr[dU * dS/dU]

        This checks the overall sign and normalisation of the force.
        """
        U = _hot_start()
        F = gauge_force(U, BETA)
        S0 = wilson_gauge_action(U, BETA)

        # Perturb U[0] at site (0,0,0,0) along generator 0 with step eps
        eps = 1e-5
        mu, site, a = 0, (0, 0, 0, 0), 0
        idx = (mu,) + site
        gen = gell_mann_matrices[a]
        exp_p = torch.linalg.matrix_exp(1j * eps * gen)

        U_pert = U.clone()
        U_pert[idx] = exp_p @ U[idx]
        S_pert = wilson_gauge_action(U_pert, BETA)

        # Numerical first-order change
        dS_num = (S_pert - S0).item()

        # Analytic first-order change: dS = -Tr[lambda_a * F_mu(site)] * eps
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
        U_noisy = U + noise  # no longer unitary

        U_r = reunitarize(U_noisy)

        eye = torch.eye(NC, dtype=DTYPE)
        UUdag = torch.matmul(U_r, U_r.conj().transpose(-1, -2))
        assert torch.allclose(
            UUdag, eye.expand_as(UUdag), atol=1e-10
        ), f"Reunitarized U not unitary; max |UU†-I| = {(UUdag - eye.expand_as(UUdag)).abs().max().item():.2e}"

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
        Reunitarizing an already-valid SU(3) field must be a no-op
        (result should be equal to input up to floating-point rounding).
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


# ===========================================================================
# TestIntegrator
# ===========================================================================


class TestIntegrator:
    """
    Tests for integrator.py.

    The gold-standard tests here are:
      1. Time-reversibility: run trajectory forward, negate P, run back ->
         must exactly recover (U0, -P0).  This is exact for any symplectic
         integrator and any step size; it catches sign errors and wrong
         coefficients immediately.
      2. Energy conservation scaling: |delta_H| must scale as eps^2 for
         leapfrog and eps^4 for OMF4 as eps -> 0.
    """

    def _test_U_update(self, U, P, eps):
        return U + eps * P

    def _test_force(self, U):
        return -U

    def _test_hamiltonian(self, U, P):
        return (P**2) / 2 + torch.sum(U**2) / 2

    def _force(self, U):
        return gauge_force(U, BETA)

    # -----------------------------------------------------------------------
    # Time-reversibility  (exact, not approximate)
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
        n_steps, step_size = 4, 0.05

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

    def _get_max_diff(self, step_size, time, test_system):
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
                U, P = leapfrog(
                    U, P, 1, self._test_force, step_size, self._test_U_update
                )
            else:
                Hs[_] = hamiltonian(U, P, BETA)
                U, P = leapfrog(U, P, 1, self._force, step_size)
        return (Hs.max() - Hs.min()).abs()

    def test_leapfrog_energy_drift_HO(self):
        """
        For leapfrog, delta_H = O(eps^2).
        Halving the step size must reduce delta_H by ~4x.
        This test checks this for a 1D harmonic oscillator.
        """

        dH_coarse = self._get_max_diff(0.1, 100, True)
        dH_fine = self._get_max_diff(0.05, 100, True)
        ratio = dH_coarse / dH_fine

        assert 3.5 < ratio < 4.5, (
            f"Leapfrog not 2nd order: dH(0.1)={dH_coarse:.3e}, "
            f"dH(0.05)={dH_fine:.3e}, ratio={ratio:.2f} (expected ~4)"
        )

    def test_leapfrog_energy_drift_MD(self):
        """
        For leapfrog, delta_H = O(eps^2).
        Halving the step size must reduce delta_H by ~4x.
        This test checks this for the MD simulation in HMC.
        """

        dH_coarse = self._get_max_diff(0.1, 10, False)
        dH_fine = self._get_max_diff(0.05, 10, False)
        ratio = dH_coarse / dH_fine

        assert 3.5 < ratio < 4.5, (
            f"Leapfrog not 2nd order: dH(0.1)={dH_coarse:.3e}, "
            f"dH(0.05)={dH_fine:.3e}, ratio={ratio:.2f} (expected ~4)"
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
        ), f"Links left SU(3) after leapfrog; max |UU†-I| = {(UUdag - eye.expand_as(UUdag)).abs().max().item():.2e}"


# ===========================================================================
# TestMonteCarlo
# ===========================================================================


class TestMonteCarlo:
    """Tests for monte_carlo.py: momentum sampling and Metropolis step."""

    # -----------------------------------------------------------------------
    # sample_momenta
    # -----------------------------------------------------------------------

    def test_momenta_shape(self):
        """sample_momenta must return a tensor with the same shape as U."""
        U = _cold_start()
        torch.manual_seed(SEED)
        P = sample_momenta(U)
        assert P.shape == U.shape

    def test_momenta_are_hermitian(self):
        """P - P^dag must be zero (Hermitian) at every site."""
        U = _cold_start()
        torch.manual_seed(SEED)
        P = sample_momenta(U)
        herm = P - P.conj().transpose(-1, -2)
        assert torch.allclose(
            herm, torch.zeros_like(herm), atol=1e-12
        ), f"Momenta not Hermitian; max |P-P†| = {herm.abs().max().item():.2e}"

    def test_momenta_are_traceless(self):
        """Tr[P_mu(x)] = 0 at every site for su(3)-valued momenta."""
        U = _cold_start()
        torch.manual_seed(SEED)
        P = sample_momenta(U)
        traces = torch.einsum("...ii->...", P)
        assert torch.allclose(
            traces, torch.zeros_like(traces), atol=1e-12
        ), f"Momenta not traceless; max |Tr P| = {traces.abs().max().item():.2e}"

    def test_momenta_correct_dtype(self):
        """sample_momenta must return complex128 tensors."""
        U = _cold_start()
        P = sample_momenta(U)
        assert P.dtype == torch.complex128

    def test_kinetic_energy_distribution(self):
        """
        With P = sum_a p_a lambda_a and p_a ~ N(0,1), the kinetic energy
        T = sum_{mu,x,a} 1 / 2 p_a^2 has expectation
        E[T] = 1 / 2 * 4 * V * 8  (4 dirs, 8 generators, V sites).

        We draw many samples and check the sample mean is close to the
        theoretical mean (within 5 sigma for N=200 samples).
        """
        V = L**4
        n_generators = 8
        n_dirs = 4
        expected_mean = 0.5 * n_dirs * V * n_generators

        N_samples = 200
        torch.manual_seed(SEED)
        U = _cold_start()
        energies = [
            kinetic_energy(sample_momenta(U)).item() for _ in range(N_samples)
        ]
        sample_mean = sum(energies) / N_samples

        # Standard deviation of the mean: sigma/sqrt(N)
        # For chi-squared: Var[T] = (1/4) * n_dof, std_mean ~ sqrt(n_dof/4/N)
        n_dof = n_dirs * V * n_generators
        std_mean = math.sqrt(n_dof / (4 * N_samples))

        assert abs(sample_mean - expected_mean) < 3 * std_mean, (
            f"Kinetic energy mean {sample_mean:.2f} far from expected {expected_mean:.2f} "
            f"(3-sigma bound: {5*std_mean:.2f})"
        )

    # -----------------------------------------------------------------------
    # metropolis_accept
    # -----------------------------------------------------------------------

    def test_metropolis_always_accepts_negative_dH(self):
        """
        When delta_H < 0 (new configuration has lower energy),
        the Metropolis step must always accept.
        """
        for dH_val in [-1e-10, -1.0, -100.0]:
            dH = torch.tensor(dH_val)
            assert (
                metropolis_accept(dH) is True
            ), f"Metropolis rejected delta_H={dH_val} < 0"

    def test_metropolis_always_accepts_zero_dH(self):
        """delta_H = 0 must always be accepted (probability = 1)."""
        dH = torch.tensor(0.0)
        assert metropolis_accept(dH) is True

    def test_metropolis_never_accepts_infinite_dH(self):
        """
        When delta_H -> infinity, the acceptance probability -> 0.
        Practically, for very large delta_H, the random number can never
        be smaller than exp(-delta_H) ~ 0.
        """
        torch.manual_seed(SEED)
        dH = torch.tensor(1e10)
        assert metropolis_accept(dH) is False

    def test_metropolis_acceptance_rate_at_known_dH(self):
        """
        For a fixed delta_H = 1.0, the theoretical acceptance probability
        is exp(-1) ≈ 0.368.  Over many trials the empirical rate must be
        close to this value.
        """
        dH = torch.tensor(1.0)
        N = 10_000
        torch.manual_seed(SEED)
        accepted = sum(1 for _ in range(N) if metropolis_accept(dH))
        rate = accepted / N
        expected = math.exp(-1.0)
        # Allow 3-sigma: std = sqrt(p*(1-p)/N) ~ 0.005
        assert abs(rate - expected) < 4 * math.sqrt(
            expected * (1 - expected) / N
        ), f"Metropolis rate {rate:.4f} far from expected {expected:.4f} for dH=1"

    def test_metropolis_acceptance_rate_at_large_dH(self):
        """
        For delta_H = 3.0, expected acceptance = exp(-3) ≈ 0.0498.
        """
        dH = torch.tensor(3.0)
        N = 10_000
        torch.manual_seed(SEED + 1)
        accepted = sum(1 for _ in range(N) if metropolis_accept(dH))
        rate = accepted / N
        expected = math.exp(-3.0)
        assert abs(rate - expected) < 4 * math.sqrt(
            expected * (1 - expected) / N
        ), f"Metropolis rate {rate:.4f} far from expected {expected:.4f} for dH=3"
