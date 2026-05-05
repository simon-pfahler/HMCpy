"""
HMCpy -- Hybrid Monte Carlo for lattice QCD gauge configurations.

GPU-compatible (via PyTorch), compatible with qcd_ml.

Modules
-------
physics     : Wilson action, gauge force, Hamiltonian, SU(3) reunitarisation
fermion     : Pseudofermion fields, Dirac operator protocol, fermion force
integrator  : Leapfrog and OMF4 symplectic integrators (pure gauge + dynamical)
monte_carlo : Momentum sampling and Metropolis accept/reject (pure gauge + dynamical)
utility     : SU(3) generators and exponential update utilities
"""

from .fermion import fermion_force_wilson_analytic, pseudofermion_action
from .utility import gell_mann_matrices
from .integrator import leapfrog, omf4
from .monte_carlo import hmc_step, metropolis_accept, sample_momenta
from .physics import (
    gauge_force,
    hamiltonian,
    kinetic_energy,
    plaquette_average,
    reunitarize,
    wilson_gauge_action,
)

__all__ = [
    # physics
    "wilson_gauge_action",
    "plaquette_average",
    "gauge_force",
    "kinetic_energy",
    "hamiltonian",
    "reunitarize",
    # fermion
    "pseudofermion_action",
    "fermion_force_wilson_analytic",
    # integrator
    "leapfrog",
    "omf4",
    # utility
    "gell_mann_matrices",
    # monte_carlo
    "sample_momenta",
    "metropolis_accept",
    "hmc_step",
]
