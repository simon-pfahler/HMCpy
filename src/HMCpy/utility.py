"""
utility.py -- Utility functions and constants for HMCpy.

This module provides:
  - gell_mann_matrices : SU(3) generators (Gell-Mann matrices)
  - _exp_update_U : SU(3) exponential update for gauge fields
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


# ---------------------------------------------------------------------------
# SU(3) exponential link update
# ---------------------------------------------------------------------------


def _exp_update_U(U: torch.Tensor, P: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Update gauge field via SU(3) exponential map:

        U_mu(x) <- exp(i * eps * P_mu(x)) . U_mu(x)

    P is su(3)-valued (traceless Hermitian) so exp(i * eps * P) is SU(3),
    ensuring U stays on the SU(3) manifold.

    Parameters
    ----------
    U   : torch.Tensor, gauge field [4, Lx, Ly, Lz, Lt, Nc, Nc]
    P   : torch.Tensor, conjugate momenta (traceless Hermitian), same shape
    eps : float, step size

    Returns
    -------
    U_new : torch.Tensor, updated gauge field, same shape as U
    """
    return torch.matmul(torch.linalg.matrix_exp(1j * eps * P), U)
