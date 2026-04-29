"""
HMCpy -- Hybrid Monte Carlo for lattice QCD gauge configurations.

GPU-compatible (via PyTorch), compatible with qcd_ml.

Modules
-------
physics     : Wilson action, gauge force, Hamiltonian, SU(3) reunitarisation
fermion     : Pseudofermion fields, Dirac operator protocol, fermion force
integrator  : Leapfrog and OMF2 symplectic integrators (pure gauge + dynamical)
monte_carlo : Momentum sampling and Metropolis accept/reject (pure gauge + dynamical)
"""

from .physics import (
    wilson_gauge_action,
    plaquette_average,
    gauge_force,
    kinetic_energy,
    hamiltonian,
    reunitarize,
)
from .fermion import (
    DiracOperator,
    Qcd_ml_DiracWilson,
    sample_pseudofermion,
    pseudofermion_action,
    fermion_force,
    fermion_force_wilson_analytic,
)
from .integrator import leapfrog, omf2
from .monte_carlo import sample_momenta, metropolis_accept, hmc_step

__all__ = [
    # physics
    "wilson_gauge_action",
    "plaquette_average",
    "gauge_force",
    "kinetic_energy",
    "hamiltonian",
    "reunitarize",
    # fermion
    "DiracOperator",
    "Qcd_ml_DiracWilson",
    "sample_pseudofermion",
    "pseudofermion_action",
    "fermion_force",
    "fermion_force_wilson_analytic",
    # integrator
    "leapfrog",
    "omf2",
    # monte_carlo
    "sample_momenta",
    "metropolis_accept",
    "hmc_step",
]
