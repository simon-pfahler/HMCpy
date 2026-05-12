"""Tests for HMCpy.monte_carlo module: momentum sampling and Metropolis criterion."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test.utils import NC, SEED, L, cold_start

from HMCpy.monte_carlo import metropolis_accept, sample_momenta
from HMCpy.physics import kinetic_energy


class TestMonteCarlo:
    # --- Momentum sampling ---

    def test_momenta_shape(self):
        """Sampled momenta have same shape as gauge field."""
        assert sample_momenta(cold_start()).shape == cold_start().shape

    def test_momenta_hermitian(self):
        """Sampled momenta are Hermitian: P = P^dag."""
        torch.manual_seed(SEED)
        P = sample_momenta(cold_start())
        assert torch.allclose(
            P - P.conj().transpose(-1, -2), torch.zeros_like(P), atol=1e-12
        )

    def test_momenta_traceless(self):
        """Sampled momenta are traceless: Tr[P] = 0."""
        torch.manual_seed(SEED)
        traces = torch.einsum("...ii->...", sample_momenta(cold_start()))
        assert torch.allclose(traces, torch.zeros_like(traces), atol=1e-12)

    def test_momenta_dtype(self):
        """Sampled momenta are complex128."""
        assert sample_momenta(cold_start()).dtype == torch.complex128

    def test_kinetic_energy_distribution(self):
        """Kinetic energy distribution matches theoretical chi-squared."""
        torch.manual_seed(SEED)
        U = cold_start()
        V, n_gen, n_dirs = L**4, 8, 4
        expected_mean = 0.5 * n_dirs * V * n_gen
        N = 200
        energies = [kinetic_energy(sample_momenta(U)).item() for _ in range(N)]
        sample_mean = sum(energies) / N
        std_mean = math.sqrt(n_dirs * V * n_gen / (4 * N))
        assert abs(sample_mean - expected_mean) < 3 * std_mean

    # --- Metropolis accept/reject ---

    @pytest.mark.parametrize("dH", [-1e-10, -1.0, -100.0])
    def test_metropolis_accepts_negative_dH(self, dH):
        """Metropolis always accepts when delta_H < 0."""
        assert metropolis_accept(torch.tensor(dH)) is True

    def test_metropolis_accepts_zero_dH(self):
        """Metropolis always accepts when delta_H = 0."""
        assert metropolis_accept(torch.tensor(0.0)) is True

    def test_metropolis_rejects_infinite_dH(self):
        """Metropolis never accepts when delta_H -> infinity."""
        torch.manual_seed(SEED)
        assert metropolis_accept(torch.tensor(1e10)) is False

    @pytest.mark.parametrize(
        "dH,expected", [(1.0, math.exp(-1)), (3.0, math.exp(-3))]
    )
    def test_metropolis_acceptance_rate(self, dH, expected):
        """Metropolis acceptance rate matches exp(-dH)."""
        torch.manual_seed(SEED if dH == 1.0 else SEED + 1)
        N = 10_000
        accepted = sum(
            1 for _ in range(N) if metropolis_accept(torch.tensor(dH))
        )
        rate = accepted / N
        bound = 4 * math.sqrt(expected * (1 - expected) / N)
        assert abs(rate - expected) < bound
