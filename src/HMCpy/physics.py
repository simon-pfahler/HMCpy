"""
physics.py -- Wilson gauge action, Hamiltonian equations, and SU(3) reunitarisation.

Conventions (compatible with qcd_ml):
  - Gauge field U has shape [mu, Lx, Ly, Lz, Lt, Nc, Nc]
      mu  : direction index (0=x, 1=y, 2=z, 3=t)
      Lx,Ly,Lz,Lt : lattice site coordinates
      Nc x Nc      : SU(3) color matrix (Nc=3)
  - Conjugate momenta P have the same shape as U; they are
    traceless Hermitian matrices (elements of su(3)).
  - All tensors are complex torch.Tensor (dtype=torch.complex128).
  - Periodic boundary conditions are used via torch.roll.
"""

import torch

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _plaquette(U: torch.Tensor, mu: int, nu: int) -> torch.Tensor:
    """
    Plaquette matrix U_{mu,nu}(x) for all sites x.

        U_{mu,nu}(x) = U_mu(x) . U_nu(x+mu_hat) . U_mu(x+nu_hat)^dag . U_nu(x)^dag

    U[mu] has shape [Lx, Ly, Lz, Lt, Nc, Nc]; lattice directions
    correspond to tensor dimensions 0..3 (x,y,z,t).

    Returns shape [Lx, Ly, Lz, Lt, Nc, Nc].
    """
    Umu = U[mu]
    Unu = U[nu]
    Unu_shift_mu = torch.roll(Unu, -1, dims=mu)  # U_nu(x + mu_hat)
    Umu_shift_nu = torch.roll(Umu, -1, dims=nu)  # U_mu(x + nu_hat)

    P = torch.matmul(Umu, Unu_shift_mu)
    P = torch.matmul(P, Umu_shift_nu.adjoint())
    P = torch.matmul(P, Unu.adjoint())
    return P


def _staple(U: torch.Tensor, mu: int, nu: int) -> torch.Tensor:
    """
    Sum of the two staples in the (mu,nu)-plane for link direction mu.

    Forward staple:  U_nu(x+mu)     . U_mu(x+nu)^dag . U_nu(x)^dag
    Backward staple: U_nu(x+mu-nu)^dag . U_mu(x-nu)^dag . U_nu(x-nu)

    Returns shape [Lx, Ly, Lz, Lt, Nc, Nc].
    """
    Umu = U[mu]
    Unu = U[nu]

    # Forward staple: U_nu(x+mu) . U_mu(x+nu)^dag . U_nu(x)^dag
    Unu_fwd = torch.roll(Unu, -1, dims=mu)
    Umu_fwd = torch.roll(Umu, -1, dims=nu)
    fwd = torch.matmul(Unu_fwd, Umu_fwd.adjoint())
    fwd = torch.matmul(fwd, Unu.adjoint())

    # Backward staple: U_nu(x+mu-nu)^dag . U_mu(x-nu)^dag . U_nu(x-nu)
    Unu_bwd = torch.roll(Unu, 1, dims=nu)
    Unu_bwd_mu = torch.roll(Unu_bwd, -1, dims=mu)
    Umu_bwd = torch.roll(Umu, 1, dims=nu)
    bwd = torch.matmul(
        Unu_bwd_mu.adjoint(), Umu_bwd.adjoint()
    )
    bwd = torch.matmul(bwd, Unu_bwd)

    return fwd + bwd


# ---------------------------------------------------------------------------
# Wilson gauge action
# ---------------------------------------------------------------------------


def wilson_gauge_action(U: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Wilson gauge action:

        S_W[U] = (beta / 3) * sum_{x, mu < nu} Re Tr[ 1 - U_{mu,nu}(x) ]

    Parameters
    ----------
    U    : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, 3, 3]
    beta : float -- bare inverse coupling (beta = 6 / g^2)

    Returns
    -------
    action : real scalar torch.Tensor
    """
    action = U.new_zeros(1, dtype=U.real.dtype)

    for mu in range(4):
        for nu in range(mu + 1, 4):
            retr = torch.einsum("...ii->...", _plaquette(U, mu, nu)).real
            action += (3 - retr).sum()

    return (beta / 3) * action


def plaquette_average(U: torch.Tensor) -> torch.Tensor:
    """
    Mean plaquette  <P> = (1 / (6 * V * 3)) * sum_{x,mu<nu} Re Tr U_{mu,nu}(x).

    Useful diagnostic: for a thermalised SU(3) configuration at beta~6 it
    should be around 0.5.

    Returns a real scalar tensor.
    """
    V = U[0].shape[0] * U[0].shape[1] * U[0].shape[2] * U[0].shape[3]
    total = U.new_zeros(1, dtype=U.real.dtype)

    for mu in range(4):
        for nu in range(mu + 1, 4):
            contrib = torch.einsum(
                "...ii->...", _plaquette(U, mu, nu)
            ).real.sum()
            total += contrib

    return total / (6 * V * 3)


# ---------------------------------------------------------------------------
# Gauge force  (molecular-dynamics equations of motion for P)
# ---------------------------------------------------------------------------


def gauge_force(U: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Gauge force that drives the momenta in the MD equations:

        dP_mu(x)/dt = F_mu(x)

    Computed as:

        F_mu(x) = beta/12 * 1j * sum_{nu != mu} (U_mu(x) Sigma_mu(x) - Sigma_mu^dag(x) U_mu(x))

    where  Sigma_mu(x) = sum_{nu != mu} (staple_forward).

    Parameters
    ----------
    U    : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
    beta : float

    Returns
    -------
    F : torch.Tensor, same shape as U (traceless Hermitian at each site)
    """
    F = torch.zeros_like(U)
    eye = torch.eye(3, dtype=F.dtype, device=F.device)

    for mu in range(4):
        sigma = torch.zeros_like(U[mu])
        for nu in range(4):
            if nu == mu:
                continue
            sigma += _staple(U, mu, nu)

        Q = torch.matmul(U[mu], sigma)

        A = Q - Q.adjoint()
        A -= (torch.einsum("...ii->...", A) / 3).unsqueeze(-1).unsqueeze(
            -1
        ) * eye
        F[mu] = A

    return (beta / 12) * 1j * F


# ---------------------------------------------------------------------------
# Kinetic energy and Hamiltonian
# ---------------------------------------------------------------------------


def kinetic_energy(P: torch.Tensor) -> torch.Tensor:
    """
    Kinetic term of the HMC Hamiltonian:

        T(P) = sum_{x,mu} Tr[ P_mu(x)^dag P_mu(x) ]

    Parameters
    ----------
    P : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]

    Returns
    -------
    T : real scalar tensor
    """
    return (P.conj() * P).real.sum()


def hamiltonian(U: torch.Tensor, P: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Full HMC Hamiltonian:

        H(U, P) = T(P) + S_W(U)

    Parameters
    ----------
    U    : gauge field, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
    P    : conjugate momenta (traceless Hermitian), same shape as U
    beta : float

    Returns
    -------
    H : real scalar tensor
    """
    return kinetic_energy(P) + wilson_gauge_action(U, beta)


# ---------------------------------------------------------------------------
# SU(3) reunitarisation
# ---------------------------------------------------------------------------


def reunitarize(U: torch.Tensor) -> torch.Tensor:
    """
    Project all link matrices back onto SU(3) via polar decomposition.

    After many MD steps, floating-point drift causes the links to drift away
    from the SU(3) manifold.  This function restores unitarity and unit
    determinant by replacing each matrix M with the unitary factor from its
    polar decomposition and then correcting the determinant:

        M = U_polar * H   (H positive semi-definite, U_polar unitary)
        U_SU3 = U_polar / det(U_polar)^(1/Nc)

    The polar factor is computed as  U_polar = M (M^dag M)^{-1/2},
    implemented via the SVD:  M = A S B^dag => U_polar = A B^dag.

    This is the standard approach used in production HMC codes.

    Parameters
    ----------
    U : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]

    Returns
    -------
    U_proj : torch.Tensor, same shape -- each link is an SU(3) matrix
    """
    # torch.linalg.svd returns (A, S, Bh) where M = A @ diag(S) @ Bh
    A, _, Bh = torch.linalg.svd(U)
    U_unitary = torch.matmul(A, Bh)  # unitary, shape as U

    # Fix determinant: divide by det^(1/Nc) to land on SU(3)
    Nc = U.shape[-1]
    det = torch.linalg.det(U_unitary)  # [...] complex scalar
    # (1/Nc)-th power of the determinant (keep phase only, |det|=1 already)
    phase = det / det.abs()  # det / |det|
    phase_root = torch.exp(
        torch.log(phase + 1e-30j) / Nc  # (1/Nc) * log(phase)
    )
    U_proj = U_unitary / phase_root.unsqueeze(-1).unsqueeze(-1)

    return U_proj
