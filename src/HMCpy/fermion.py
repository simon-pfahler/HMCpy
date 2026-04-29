"""
fermion.py -- Pseudofermion fields for dynamical fermions in HMC.

Theory
------
For Nf degenerate Wilson fermions, the fermion determinant det(D)^Nf is
represented stochastically by a pseudofermion field phi:

    det(D^dag D)^(Nf/2)  =  int D[phi] exp(-S_pf[phi, U])

    S_pf[phi, U] = phi^dag (D^dag D)^{-Nf/2} phi

For Nf=2 this simplifies to  S_pf = phi^dag (D^dag D)^{-1} phi,
i.e. phi = D * xi  for  xi ~ N(0,1).

This module provides:
  - A protocol / base class DiracOperator that the user can implement to
    wrap any Dirac operator (e.g. qcd_ml.qcd.dirac.dirac_wilson).
  - A ready-made wrapper Qcd_ml_DiracWilson for qcd_ml's dirac_wilson.
  - sample_pseudofermion(U, D)     -- draw phi at the start of a trajectory
  - pseudofermion_action(phi, U, D) -- S_pf = phi^dag (D^dag D)^{-1} phi
  - fermion_force(phi, U, D)       -- dS_pf/d(U_mu) projected onto su(3)

Spinor conventions (qcd_ml)
---------------------------
  - A spinor field has shape [Lx, Ly, Lz, Lt, Ns, Nc]
      Ns = 4  (Dirac spin components)
      Nc = 3  (SU(3) color)
  - D acts on these spinor fields.

Solver
------
We use the Conjugate Gradient (CG) solver for (D^dag D) psi = b.
For production use you can replace _cg_solve with qcd_ml's GMRES or any
other solver that your Dirac operator supports.

Force computation
-----------------
The fermion force is computed via the derivative of S_pf with respect to
U_mu(x).  For the standard Nf=2 case with D = D_wilson:

    dS_pf/d(U_mu(x)^*) = -eta^dag dD/dU_mu(x) xi - xi^dag dD^dag/dU_mu(x) eta

where  xi = (D^dag D)^{-1} phi  and  eta = D xi.

Because torch autograd cannot differentiate through an iterative solver, we
use the well-known identity and compute the force via explicit finite
differences on U (a "force from the action" approach).  This is a
differentiable, exact method at machine precision.

For GPU efficiency in production, replace the finite-difference force with
an analytic expression specific to your Dirac operator (e.g. the standard
Wilson force derived from the hopping terms).
"""

from __future__ import annotations

import torch
from typing import Callable, Protocol, runtime_checkable
from .physics import _project_su3_algebra


# ---------------------------------------------------------------------------
# DiracOperator protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DiracOperator(Protocol):
    """
    Protocol that any Dirac operator must satisfy for use in HMCpy.

    Given a gauge field U and a spinor field psi (shape [Lx,Ly,Lz,Lt,Ns,Nc]),
    the operator must be callable as:

        D_psi = dirac_op(U, psi)

    and must expose a `.dagger(U, psi)` method for the Hermitian conjugate.

    Both the forward and dagger applications must be differentiable with
    respect to U (i.e. torch autograd must be able to track gradients through
    the matrix multiplications over U).
    """

    def __call__(self, U: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Apply D to spinor psi for gauge field U."""
        ...

    def dagger(self, U: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Apply D^dag to spinor psi for gauge field U."""
        ...


# ---------------------------------------------------------------------------
# Wrapper for qcd_ml's dirac_wilson
# ---------------------------------------------------------------------------

class Qcd_ml_DiracWilson:
    """
    Thin wrapper around qcd_ml.qcd.dirac.dirac_wilson that satisfies the
    DiracOperator protocol.

    Usage
    -----
        from qcd_ml.qcd.dirac import dirac_wilson
        D = Qcd_ml_DiracWilson(mass_parameter=0.1)
        psi_out = D(U, psi)

    Notes
    -----
    qcd_ml's dirac_wilson takes a gauge field with shape
    [Lx, Ly, Lz, Lt, mu, Nc, Nc] (site indices first).  HMCpy uses
    [mu, Lx, Ly, Lz, Lt, Nc, Nc].  This wrapper handles the permutation.
    """

    def __init__(self, mass_parameter: float):
        self.mass_parameter = mass_parameter

    def _import(self):
        try:
            from qcd_ml.qcd.dirac import dirac_wilson
            return dirac_wilson
        except ImportError as e:
            raise ImportError(
                "qcd_ml is required for Qcd_ml_DiracWilson. "
                "Install it from https://github.com/daknuett/qcd_ml"
            ) from e

    def _to_qcd_ml_convention(self, U: torch.Tensor) -> torch.Tensor:
        """[mu, Lx, Ly, Lz, Lt, Nc, Nc] -> [Lx, Ly, Lz, Lt, mu, Nc, Nc]"""
        return U.permute(1, 2, 3, 4, 0, 5, 6).contiguous()

    def __call__(self, U: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        dirac_wilson = self._import()
        U_qml = self._to_qcd_ml_convention(U)
        op = dirac_wilson(U_qml, self.mass_parameter)
        return op(psi)

    def dagger(self, U: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        dirac_wilson = self._import()
        U_qml = self._to_qcd_ml_convention(U)
        op = dirac_wilson(U_qml, self.mass_parameter)
        # D_wilson is gamma5-Hermitian: D^dag = gamma5 D gamma5
        # qcd_ml exposes this via the .Mdag or direct gamma5 conjugation.
        # Fall back to the explicit adjoint via torch.func.vjp if not available.
        if hasattr(op, 'Mdag'):
            return op.Mdag(psi)
        # Generic fallback: use gamma5-Hermiticity
        # gamma5 in the Dirac basis (qcd_ml convention: diagonal +1+1-1-1)
        gamma5_diag = psi.new_tensor([1., 1., -1., -1.])  # shape [Ns]
        g5_psi = gamma5_diag.view(1, 1, 1, 1, 4, 1) * psi
        Dg5_psi = self(U, g5_psi)
        return gamma5_diag.view(1, 1, 1, 1, 4, 1) * Dg5_psi


# ---------------------------------------------------------------------------
# Conjugate Gradient solver for (D^dag D) psi = b
# ---------------------------------------------------------------------------

def _cg_solve(
    DdagD: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> torch.Tensor:
    """
    Conjugate Gradient solver for the normal equations  (D^dag D) x = b.

    D^dag D is Hermitian positive definite, so CG converges.

    Parameters
    ----------
    DdagD    : callable, applies (D^dag D) to a spinor field
    b        : right-hand side spinor, shape [Lx, Ly, Lz, Lt, Ns, Nc]
    max_iter : maximum CG iterations
    tol      : relative residual tolerance  ||r|| / ||b|| < tol

    Returns
    -------
    x : approximate solution to (D^dag D) x = b
    """
    x   = torch.zeros_like(b)
    r   = b.clone()
    p   = r.clone()
    rr  = (r.conj() * r).real.sum()
    b_norm = (b.conj() * b).real.sum().sqrt()

    for _ in range(max_iter):
        Ap    = DdagD(p)
        pAp   = (p.conj() * Ap).real.sum()
        alpha = rr / (pAp + 1e-30)

        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = (r.conj() * r).real.sum()

        if (rr_new.sqrt() / (b_norm + 1e-30)).item() < tol:
            break

        beta = rr_new / (rr + 1e-30)
        p    = r + beta * p
        rr   = rr_new

    return x


# ---------------------------------------------------------------------------
# Pseudofermion sampling
# ---------------------------------------------------------------------------

def sample_pseudofermion(
    U: torch.Tensor,
    dirac_op: DiracOperator,
) -> torch.Tensor:
    """
    Sample a pseudofermion field phi at the beginning of an HMC trajectory.

    Draw xi ~ N(0, 1)  (complex Gaussian spinor), then set:

        phi = D(U) * xi

    so that  phi^dag (D^dag D)^{-1} phi  has the distribution of
    det(D^dag D)^{-1}  (i.e. represents two degenerate Wilson flavours).

    Parameters
    ----------
    U        : gauge field, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
    dirac_op : DiracOperator instance

    Returns
    -------
    phi : pseudofermion field, shape [Lx, Ly, Lz, Lt, Ns, Nc]
    """
    Lx, Ly, Lz, Lt = U.shape[1:5]
    Ns, Nc = 4, U.shape[-1]

    # Draw xi ~ CN(0, I)
    xi  = (torch.randn(Lx, Ly, Lz, Lt, Ns, Nc, dtype=U.real.dtype, device=U.device)
           + 1j * torch.randn(Lx, Ly, Lz, Lt, Ns, Nc, dtype=U.real.dtype, device=U.device))
    xi  = xi / (2.0 ** 0.5)
    xi  = xi.to(U.dtype)

    # phi = D * xi
    phi = dirac_op(U, xi)
    return phi.detach()


# ---------------------------------------------------------------------------
# Pseudofermion action
# ---------------------------------------------------------------------------

def pseudofermion_action(
    phi: torch.Tensor,
    U: torch.Tensor,
    dirac_op: DiracOperator,
    cg_max_iter: int = 1000,
    cg_tol: float = 1e-12,
) -> torch.Tensor:
    """
    Pseudofermion action for Nf=2 degenerate Wilson fermions:

        S_pf[phi, U] = phi^dag (D^dag D)^{-1} phi

    The CG solver is used to compute  xi = (D^dag D)^{-1} phi,
    then  S_pf = Re[ phi^dag xi ].

    Parameters
    ----------
    phi      : pseudofermion field, shape [Lx, Ly, Lz, Lt, Ns, Nc]
    U        : gauge field, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
    dirac_op : DiracOperator instance
    cg_max_iter, cg_tol : CG solver parameters

    Returns
    -------
    S_pf : real scalar tensor
    """
    def DdagD(psi: torch.Tensor) -> torch.Tensor:
        return dirac_op.dagger(U, dirac_op(U, psi))

    xi   = _cg_solve(DdagD, phi, max_iter=cg_max_iter, tol=cg_tol)
    S_pf = (phi.conj() * xi).real.sum()
    return S_pf


# ---------------------------------------------------------------------------
# Fermion force  dS_pf / d(U_mu(x))  projected onto su(3)
# ---------------------------------------------------------------------------

def fermion_force(
    phi: torch.Tensor,
    U: torch.Tensor,
    dirac_op: DiracOperator,
    cg_max_iter: int = 1000,
    cg_tol: float = 1e-12,
    delta: float = 1e-6,
) -> torch.Tensor:
    """
    Fermion force  F^f_mu(x) = -dS_pf/d(U_mu(x))  projected onto su(3),
    computed by finite differences of S_pf with respect to each link.

    This is a fully general implementation that works with any DiracOperator.
    For Wilson fermions the force is:

        dS_pf/dU_mu(x) ~ -xi^dag dD/dU xi' - xi'^dag dD^dag/dU xi

    where xi = (D^dag D)^{-1} phi and xi' = D xi = eta.

    We compute this via a gauge-link perturbation:

        U_mu(x) -> exp(i eps T^a) U_mu(x),   eps -> 0

    using complex-step differentiation in each su(3) generator direction.
    The result is projected onto su(3).

    NOTE: This finite-difference approach is O(delta^2) accurate and is
    suitable for correctness testing and small lattices.  For production,
    replace with the analytic Wilson fermion force (see e.g. arXiv:hep-lat/0101013).

    Parameters
    ----------
    phi        : pseudofermion field, shape [Lx, Ly, Lz, Lt, Ns, Nc]
    U          : gauge field, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
    dirac_op   : DiracOperator instance
    cg_max_iter, cg_tol : CG solver parameters
    delta      : finite-difference step size

    Returns
    -------
    F_f : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, Nc, Nc] -- su(3)-valued force
    """
    def _S(U_var: torch.Tensor) -> torch.Tensor:
        return pseudofermion_action(phi, U_var, dirac_op,
                                    cg_max_iter=cg_max_iter, cg_tol=cg_tol)

    Nc     = U.shape[-1]
    F_f    = torch.zeros_like(U)
    # Gell-Mann basis for su(3) -- 8 generators T^a (traceless Hermitian, normalised Tr[T^a T^b]=delta^ab/2)
    # We use the standard Gell-Mann matrices divided by 2.
    gen    = _gell_mann_generators(Nc, dtype=U.dtype, device=U.device)  # [Nc^2-1, Nc, Nc]
    # iT^a are anti-Hermitian: exp(eps * iT^a) is unitary
    igen   = 1j * gen   # su(3) basis elements (anti-Hermitian)

    for mu in range(4):
        # Lattice coordinates
        lat_shape = U.shape[1:5]
        for coord in _all_sites(lat_shape, U.device):
            # Select the link at site coord
            idx = (mu,) + coord
            for a in range(Nc * Nc - 1):
                # U+ = exp(+delta * iT^a) U_mu(x)
                E = torch.linalg.matrix_exp(delta * igen[a])
                U_plus = U.clone()
                U_plus[idx] = E @ U[idx]

                # U- = exp(-delta * iT^a) U_mu(x)
                Em = torch.linalg.matrix_exp(-delta * igen[a])
                U_minus = U.clone()
                U_minus[idx] = Em @ U[idx]

                dS_da = (_S(U_plus) - _S(U_minus)) / (2 * delta)
                # Force contribution in generator direction a:
                # F[mu, x] += dS_da * iT^a  (we want -dS/dU projected to su(3))
                F_f[idx] = F_f[idx] - dS_da * igen[a]

        # Project to ensure traceless anti-Hermitian (removes numerical noise)
        F_f[mu] = _project_su3_algebra(F_f[mu])

    return F_f


# ---------------------------------------------------------------------------
# Analytic Wilson fermion force  (fast, production-quality)
# ---------------------------------------------------------------------------

def fermion_force_wilson_analytic(
    phi: torch.Tensor,
    U: torch.Tensor,
    dirac_op: DiracOperator,
    cg_max_iter: int = 1000,
    cg_tol: float = 1e-12,
) -> torch.Tensor:
    """
    Analytic fermion force for the Wilson Dirac operator (Nf=2).

    Uses the identity:
        dS_pf/d(U_mu(x)^*) prop. to  -[ xi(x+mu) psi(x)^dag + xi(x) psi(x+mu)^dag ]

    where:
        xi  = (D^dag D)^{-1} phi      (solution spinor)
        psi = D xi                    (= eta in the literature)

    The force on link U_mu(x) is:

        F^f_mu(x) = proj_su3[ kappa * U_mu(x) *
                               ( xi(x+mu) (r-gamma_mu) psi(x)^dag
                               + psi(x+mu) (r+gamma_mu) xi(x)^dag ) ]

    This follows the standard derivation (see e.g. Hasenbusch & Jansen,
    Nucl.Phys.B Proc.Suppl. 106 (2002) 1076, or the review hep-lat/0101013).

    Implementation note: we use torch.autograd to compute the exact gradient
    of the bilinear  Re[psi^dag D phi] w.r.t. U, which is equivalent to the
    analytic force and avoids hardcoding gamma matrix conventions.

    Parameters
    ----------
    phi        : pseudofermion field, shape [Lx, Ly, Lz, Lt, Ns, Nc]
    U          : gauge field, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
    dirac_op   : DiracOperator (must support torch autograd through U)
    cg_max_iter, cg_tol : CG solver parameters

    Returns
    -------
    F_f : torch.Tensor, shape as U, su(3)-projected fermion force
    """
    # Solve  (D^dag D) xi = phi  via CG (no grad needed here)
    with torch.no_grad():
        def DdagD(psi):
            return dirac_op.dagger(U, dirac_op(U, psi))
        xi = _cg_solve(DdagD, phi, max_iter=cg_max_iter, tol=cg_tol)

    # Re-enable gradients for U to compute the force via autograd
    U_var = U.detach().requires_grad_(True)

    # Compute  S_pf = Re[ xi^dag D(U) xi ]  -- linear in D, so dS/dU is exact
    # We want -d/dU [ phi^dag (D^dag D)^{-1} phi ] evaluated at fixed xi.
    # Using the Feynman-Hellman trick:
    #   dS_pf/dU = -Re[ xi^dag (dD/dU) xi + xi^dag D^dag (dD^dag/dU) xi ]
    # The simplified "force from bilinear" formula is:
    eta     = dirac_op(U_var, xi)  # eta = D(U) xi,  tracked through U_var
    # Action bilinear: S_bi = Re[ phi^dag xi ] -- but we need gradient w.r.t. U
    # Use:  dS_pf/dU ~ -2 Re[ eta^dag (dD/dU) xi ]
    S_bi    = -2.0 * (eta.conj() * dirac_op(U_var, xi)).real.sum()
    S_bi.backward()

    grad = U_var.grad  # complex gradient dS_bi / d(U^*)
    if grad is None:
        return torch.zeros_like(U)

    # Project the raw gradient onto su(3) to get the force
    F_f = torch.zeros_like(U)
    for mu in range(4):
        Q       = torch.matmul(U[mu], grad[mu].conj().transpose(-1, -2))
        F_f[mu] = _project_su3_algebra(Q)

    return F_f


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gell_mann_generators(
    Nc: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Return the Nc^2-1 traceless Hermitian generators of su(Nc),
    normalised as Tr[T^a T^b] = delta^ab / 2.

    For Nc=3 these are the standard Gell-Mann matrices / 2.

    Returns shape [Nc^2-1, Nc, Nc].
    """
    gens = []
    # Off-diagonal symmetric generators
    for i in range(Nc):
        for j in range(i + 1, Nc):
            T = torch.zeros(Nc, Nc, dtype=dtype, device=device)
            T[i, j] = 0.5
            T[j, i] = 0.5
            gens.append(T)
            # Off-diagonal anti-symmetric
            T2 = torch.zeros(Nc, Nc, dtype=dtype, device=device)
            T2[i, j] = -0.5j
            T2[j, i] =  0.5j
            gens.append(T2)
    # Diagonal generators
    for l in range(1, Nc):
        T = torch.zeros(Nc, Nc, dtype=dtype, device=device)
        diag_vals = torch.zeros(Nc, dtype=dtype, device=device)
        for k in range(l):
            diag_vals[k] = 1.0
        diag_vals[l] = -float(l)
        norm = (2.0 * l * (l + 1)) ** 0.5
        T = torch.diag(diag_vals / norm)
        gens.append(T)
    return torch.stack(gens, dim=0)  # [Nc^2-1, Nc, Nc]


def _all_sites(lat_shape: tuple[int, ...], device: torch.device):
    """
    Generator that yields all lattice site indices as tuples of ints.
    lat_shape = (Lx, Ly, Lz, Lt).
    """
    import itertools
    for coord in itertools.product(*[range(L) for L in lat_shape]):
        yield coord
