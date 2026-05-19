"""
utility.py -- Utility functions and constants for HMCpy.

This module provides:
  - gell_mann_matrices : Gell-Mann matrices
  - su3_generators : Pre-computed 0.5 * gell_mann_matrices (SU(3) generators)
  - exp_update_U : SU(3) exponential update for gauge fields
"""

import torch

# ---------------------------------------------------------------------------
# SU(3) generators (Gell-Mann matrices)
# ---------------------------------------------------------------------------

gell_mann_matrices = torch.tensor(
    (
        ((0, 1, 0), (1, 0, 0), (0, 0, 0)),
        ((0, -1j, 0), (1j, 0, 0), (0, 0, 0)),
        ((1, 0, 0), (0, -1, 0), (0, 0, 0)),
        ((0, 0, 1), (0, 0, 0), (1, 0, 0)),
        ((0, 0, -1j), (0, 0, 0), (1j, 0, 0)),
        ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((0, 0, 0), (0, 0, -1j), (0, 1j, 0)),
        (
            (1 / (3**0.5), 0, 0),
            (0, 1 / (3**0.5), 0),
            (0, 0, -2 / (3**0.5)),
        ),
    ),
    dtype=torch.cdouble,
)

su3_generators = 0.5 * gell_mann_matrices


# ---------------------------------------------------------------------------
# SU(3) exponential link update
# ---------------------------------------------------------------------------


def exp_update_U(U: torch.Tensor, P: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Update gauge field via SU(3) exponential map: U <- exp(i * eps * P) @ U.

    P is su(3)-valued (traceless Hermitian), so exp(i * eps * P) is SU(3),
    ensuring the updated U stays on the SU(3) manifold.

    Parameters
    ----------
    U : torch.Tensor
        Gauge field [4, Lx, Ly, Lz, Lt, Nc, Nc]
    P : torch.Tensor
        Conjugate momenta (traceless Hermitian), same shape as U
    eps : float
        Step size

    Returns
    -------
    torch.Tensor
        Updated gauge field, same shape as U
    """
    return torch.matmul(torch.linalg.matrix_exp(1j * eps * P), U)
