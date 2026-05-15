"""
utils.py -- Shared utility functions and constants for tests.

This module provides shared test utilities to avoid code duplication across test files.

Shared constants:
  - SEED: Random seed for reproducibility (42)
  - L: Lattice side length (2)
  - NC: Number of colors for SU(Nc) (3)
  - DTYPE: Data type for tensors (torch.complex128)
  - BETA: Gauge coupling parameter (6.0)
  - MASS: Fermion mass parameter (-0.2)

Shared functions:
  - cold_start: Generate cold start gauge field (identity links)
  - hot_start: Generate hot start gauge field (random SU(3) links)
  - random_momenta: Generate random traceless Hermitian momenta
  - conjugate_gradient: CG solver for Hermitian positive-definite systems
"""

import torch

# Shared constants for all tests
SEED = 42
L = 2  # lattice side length
NC = 3  # SU(3)
DTYPE = torch.complex128
BETA = 6.0  # gauge coupling
MASS = -0.2  # fermion mass parameter


# ---------------------------------------------------------------------------
# Gauge field initialization
# ---------------------------------------------------------------------------


def cold_start(L=L, Nc=NC) -> torch.Tensor:
    """Generate a cold start gauge field with all links = identity."""
    return torch.eye(Nc, dtype=DTYPE).expand(4, L, L, L, L, Nc, Nc).clone()


def hot_start(L=L, Nc=NC, seed=SEED) -> torch.Tensor:
    """Generate a hot start gauge field with random SU(3) links via QR decomposition."""
    g = torch.Generator()
    g.manual_seed(seed)
    X = torch.randn(4, L, L, L, L, Nc, Nc, dtype=DTYPE, generator=g)
    Q, _ = torch.linalg.qr(X)
    det = torch.linalg.det(Q)
    phase = det / det.abs()
    phase_root = torch.exp(torch.log(phase + 1e-30j) / Nc)
    return (Q / phase_root.unsqueeze(-1).unsqueeze(-1)).detach()


# ---------------------------------------------------------------------------
# Momenta utilities
# ---------------------------------------------------------------------------


def random_momenta(U: torch.Tensor, seed=SEED) -> torch.Tensor:
    """Generate random traceless Hermitian momenta."""
    from HMCpy.monte_carlo import sample_momenta

    torch.manual_seed(seed)
    return sample_momenta(U)


# ---------------------------------------------------------------------------
# Solver utilities
# ---------------------------------------------------------------------------


def conjugate_gradient(
    A_op: callable,
    b: torch.Tensor,
    maxiter: int = 1000,
    tol: float = 1e-10,
) -> torch.Tensor:
    """
    Conjugate gradient solver for Hermitian positive-definite systems.
    Implemented in pure PyTorch to support autograd.

    Solves A x = b for x, where A is Hermitian positive-definite.

    Parameters
    ----------
    A_op : callable that applies A to a vector
    b : right-hand side vector
    maxiter : maximum number of iterations
    tol : convergence tolerance

    Returns
    -------
    x : solution vector
    """
    x = torch.zeros_like(b)
    r = b - A_op(x)
    p = r.clone()

    for _ in range(maxiter):
        Ap = A_op(p)
        alpha = (r.conj() * r).sum() / (p.conj() * Ap).sum().real
        x = x + alpha * p
        r_new = r - alpha * Ap

        if (r_new.conj() * r_new).sum().abs() < tol:
            return x

        beta = (r_new.conj() * r_new).sum() / (r.conj() * r).sum().real
        p = r_new + beta * p
        r = r_new

    return x
