"""
conftest.py -- Shared fixtures and utilities for pytest tests.

This module provides shared test utilities to avoid code duplication across test files.

Shared fixtures:
  - seed: Random seed for reproducibility (42)
  - L: Lattice side length (2)
  - NC: Number of colors for SU(Nc) (3)
  - dtype: Data type for tensors (torch.complex128)
  - beta: Gauge coupling parameter (6.0)
  - mass: Fermion mass parameter (-0.2)

Shared functions:
  - cold_start: Generate cold start gauge field (identity links)
  - hot_start: Generate hot start gauge field (random SU(3) links)
  - random_momenta: Generate random traceless Hermitian momenta
  - conjugate_gradient: CG solver for Hermitian positive-definite systems
  - random_SU3: Generate random SU(3) matrix
  - apply_gauge_transform: Apply gauge transformation to gauge field
"""

import os
import sys

import pytest
import torch

# Add src to path so tests can import HMCpy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    return 42


@pytest.fixture
def L():
    """Lattice side length."""
    return 2


@pytest.fixture
def NC():
    """Number of colors for SU(Nc)."""
    return 3


@pytest.fixture
def dtype():
    """Data type for tensors."""
    return torch.complex128


@pytest.fixture
def beta():
    """Gauge coupling parameter."""
    return 6.0


@pytest.fixture
def mass():
    """Fermion mass parameter."""
    return -0.2


# ---------------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------------

GMRES_OPTS = {"maxiter": 1000, "eps": 1e-10, "inner_iter": 10}


# ---------------------------------------------------------------------------
# Gauge field initialization
# ---------------------------------------------------------------------------


def cold_start(L=2, Nc=3) -> torch.Tensor:
    """Generate a cold start gauge field with all links = identity."""
    return (
        torch.eye(Nc, dtype=torch.complex128)
        .expand(4, L, L, L, L, Nc, Nc)
        .clone()
    )


def hot_start(L=2, Nc=3, seed=42) -> torch.Tensor:
    """Generate a hot start gauge field with random SU(3) links via QR decomposition."""
    g = torch.Generator()
    g.manual_seed(seed)
    X = torch.randn(4, L, L, L, L, Nc, Nc, dtype=torch.complex128, generator=g)
    Q, _ = torch.linalg.qr(X)
    det = torch.linalg.det(Q)
    phase = det / det.abs()
    phase_root = torch.exp(torch.log(phase + 1e-30j) / Nc)
    return (Q / phase_root.unsqueeze(-1).unsqueeze(-1)).detach()


# ---------------------------------------------------------------------------
# Momenta utilities
# ---------------------------------------------------------------------------


def random_momenta(U: torch.Tensor, seed=42) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Gauge transformation utilities
# ---------------------------------------------------------------------------


def random_SU3(seed: int | None = None) -> torch.Tensor:
    """
    Generate random SU(3) matrix.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    Omega : torch.Tensor, shape [3, 3]
        Random SU(3) matrix
    """
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None
    X = torch.randn(3, 3, dtype=torch.complex128, generator=g)
    Q, _ = torch.linalg.qr(X)
    det = torch.linalg.det(Q)
    phase = det / det.abs()
    phase_root = torch.exp(torch.log(phase + 1e-30j) / 3)
    return (Q / phase_root).detach()


def apply_gauge_transform(U: torch.Tensor, Omega: torch.Tensor) -> torch.Tensor:
    r"""
    Apply gauge transformation to gauge field: U_μ(x) -> Omega(x) U_μ(x) Omega^\dag(x+μ).

    Parameters
    ----------
    U : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Gauge field
    Omega : torch.Tensor, shape [Lx, Ly, Lz, Lt, Nc, Nc]
        Gauge transformation

    Returns
    -------
    U_transformed : torch.Tensor, same shape as U
        Transformed gauge field
    """
    U_transformed = torch.zeros_like(U)
    Omega_dag = Omega.adjoint()
    for mu in range(4):
        Omega_shifted = torch.roll(Omega, -1, dims=[mu])
        Omega_shifted_dag = torch.roll(Omega_dag, -1, dims=[mu])
        U_transformed[mu] = torch.einsum(
            "...ab,...bc,...cd->...ad",
            Omega,
            U[mu],
            Omega_shifted_dag,
        )
    return U_transformed
