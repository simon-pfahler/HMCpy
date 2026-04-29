"""
monte_carlo.py -- HMC accept/reject step and momentum refresh.

Supports both pure-gauge and dynamical-fermion (pseudofermion) HMC.

For pure gauge:
    H = T(P) + S_W(U)

For Nf=2 Wilson fermions (one pseudofermion field):
    H = T(P) + S_W(U) + S_pf(phi, U)
    S_pf = phi^dag (D^dag D)^{-1} phi

One HMC trajectory:
  1. Sample momenta  P ~ exp(-T(P))
  2. (If dynamical fermions) Sample phi and compute S_pf_old
  3. Run symplectic MD trajectory  (U, P) -> (U', P')
  4. Metropolis accept/reject on delta_H = H_new - H_old

Conventions match HMCpy / qcd_ml: U shape [4, Lx, Ly, Lz, Lt, Nc, Nc].
"""

import torch
from .physics import hamiltonian
from .integrator import leapfrog, omf2


# ---------------------------------------------------------------------------
# Momentum sampling
# ---------------------------------------------------------------------------

def sample_momenta(U: torch.Tensor) -> torch.Tensor:
    """
    Sample conjugate momenta P ~ exp(-T(P)), T = (1/2) Tr[P^dag P].

    Each P_mu(x) is a traceless anti-Hermitian Nc x Nc matrix drawn from
    the Gaussian measure on su(Nc).

    Parameters
    ----------
    U : torch.Tensor, shape [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Used only to determine shape, dtype, and device.

    Returns
    -------
    P : torch.Tensor, same shape as U -- traceless anti-Hermitian
    """
    shape = U.shape
    Nc    = shape[-1]
    std   = 1.0 / (2.0 ** 0.5)

    X  = std * (torch.randn(shape, dtype=U.real.dtype, device=U.device)
                + 1j * torch.randn(shape, dtype=U.real.dtype, device=U.device))
    X  = X.to(U.dtype)

    A  = (X - X.conj().transpose(-1, -2)) * 0.5          # anti-Hermitian
    tr = torch.einsum("...ii->...", A) / Nc               # trace / Nc
    eye = torch.eye(Nc, dtype=U.dtype, device=U.device)
    return A - tr.unsqueeze(-1).unsqueeze(-1) * eye        # traceless


# ---------------------------------------------------------------------------
# Metropolis accept/reject step
# ---------------------------------------------------------------------------

def metropolis_accept(
    delta_H: torch.Tensor,
    rng: torch.Generator | None = None,
) -> bool:
    """
    Metropolis criterion:  accept with probability min(1, exp(-delta_H)).

    Parameters
    ----------
    delta_H : real scalar tensor,  H_new - H_old
    rng     : optional torch.Generator for reproducibility

    Returns
    -------
    accepted : bool
    """
    if delta_H.item() <= 0.0:
        return True
    prob = torch.exp(-delta_H.cpu()).item()
    u    = torch.rand(1, generator=rng).item() if rng is not None else torch.rand(1).item()
    return u < prob


# ---------------------------------------------------------------------------
# Full HMC update  (pure gauge or dynamical fermions)
# ---------------------------------------------------------------------------

def hmc_step(
    U: torch.Tensor,
    beta: float,
    n_steps: int,
    step_size: float,
    integrator: str = "leapfrog",
    rng: torch.Generator | None = None,
    # --- dynamical fermion options ---
    dirac_op=None,
    use_analytic_force: bool = True,
    cg_max_iter: int = 1000,
    cg_tol: float = 1e-12,
) -> tuple[torch.Tensor, bool, float]:
    """
    One complete HMC update.

    Pure-gauge mode (default)
    -------------------------
    Omit `dirac_op` (or pass None).

    Dynamical-fermion mode (Nf=2 Wilson)
    -------------------------------------
    Pass a `dirac_op` (DiracOperator instance, e.g. Qcd_ml_DiracWilson).
    A pseudofermion field is sampled automatically.

    Parameters
    ----------
    U                  : current gauge field [4, Lx, Ly, Lz, Lt, Nc, Nc]
    beta               : inverse bare coupling
    n_steps            : number of MD steps per trajectory
    step_size          : MD step size epsilon
    integrator         : "leapfrog" (default) or "omf2"
    rng                : optional torch.Generator
    dirac_op           : DiracOperator instance (None = pure gauge)
    use_analytic_force : use analytic (True) or finite-difference (False)
                         fermion force.  Ignored in pure-gauge mode.
    cg_max_iter        : max CG iterations for fermion force / action
    cg_tol             : CG relative residual tolerance

    Returns
    -------
    U_out    : updated gauge field
    accepted : bool
    delta_H  : float -- energy violation (use for step-size tuning)
    """
    from .physics import kinetic_energy, wilson_gauge_action

    # ---- Momentum refresh ----
    P = sample_momenta(U)

    # ---- Pseudofermion sampling (if dynamical fermions) ----
    phi      = None
    S_pf_old = U.new_zeros(1)

    if dirac_op is not None:
        from .fermion import sample_pseudofermion, pseudofermion_action
        phi      = sample_pseudofermion(U, dirac_op)
        S_pf_old = pseudofermion_action(phi, U, dirac_op,
                                         cg_max_iter=cg_max_iter, cg_tol=cg_tol)

    # ---- Initial Hamiltonian ----
    H_old = (kinetic_energy(P)
             + wilson_gauge_action(U, beta)
             + S_pf_old)

    # ---- MD trajectory ----
    integrator_kwargs = dict(
        beta=beta,
        n_steps=n_steps,
        step_size=step_size,
        phi=phi,
        dirac_op=dirac_op,
        use_analytic_force=use_analytic_force,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
    )
    if integrator == "leapfrog":
        U_prop, P_prop = leapfrog(U, P, **integrator_kwargs)
    elif integrator == "omf2":
        U_prop, P_prop = omf2(U, P, **integrator_kwargs)
    else:
        raise ValueError(f"Unknown integrator '{integrator}'. Choose 'leapfrog' or 'omf2'.")

    # ---- Proposed Hamiltonian ----
    S_pf_new = U.new_zeros(1)
    if dirac_op is not None:
        S_pf_new = pseudofermion_action(phi, U_prop, dirac_op,
                                         cg_max_iter=cg_max_iter, cg_tol=cg_tol)

    H_new = (kinetic_energy(P_prop)
             + wilson_gauge_action(U_prop, beta)
             + S_pf_new)

    dH = H_new - H_old

    # ---- Accept / reject ----
    accepted = metropolis_accept(dH, rng=rng)
    U_out    = U_prop if accepted else U

    return U_out, accepted, dH.item()
