"""
test_monte_carlo.py -- Tests for HMCpy.monte_carlo module.

Run with: pytest test_monte_carlo.py -v

Tests cover:
  - Momentum sampling
  - Metropolis accept/reject criterion
"""

import math
import os
import sys

import pytest
import torch

# Make the package importable when running from the test directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from HMCpy.monte_carlo import metropolis_accept, sample_momenta
from HMCpy.physics import kinetic_energy

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SEED = 42
L = 2  # lattice side length
NC = 3  # SU(3)
DTYPE = torch.complex128


def _cold_start(L=L, Nc=NC) -> torch.Tensor:
    """All links = identity."""
    return torch.eye(Nc, dtype=DTYPE).expand(4, L, L, L, L, Nc, Nc).clone()


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
        ), f"Momenta not Hermitian; max |P-P\u2020| = {herm.abs().max().item():.2e}"

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
        is exp(-1) \u2248 0.368.  Over many trials the empirical rate must be
        close to this value.
        """
        dH = torch.tensor(1.0)
        N = 10_000
        torch.manual_seed(SEED)
        accepted = sum(1 for _ in range(N) if metropolis_accept(dH))
        rate = accepted / N
        expected = math.exp(-1.0)
        assert abs(rate - expected) < 4 * math.sqrt(
            expected * (1 - expected) / N
        ), f"Metropolis rate {rate:.4f} far from expected {expected:.4f} for dH=1"

    def test_metropolis_acceptance_rate_at_large_dH(self):
        """
        For delta_H = 3.0, expected acceptance = exp(-3) \u2248 0.0498.
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
