# HMCpy - Hybrid Monte Carlo for Lattice QCD

[![Test](https://github.com/simon-pfahler/HMCpy/actions/workflows/test.yml/badge.svg)](https://github.com/simon-pfahler/HMCpy/actions/workflows/test.yml)

A PyTorch-based Hybrid Monte Carlo (HMC) implementation for SU(3) lattice gauge theory with optional dynamical Wilson/Wilson-clover fermions. GPU-accelerated via PyTorch CUDA support.

## Features

- **Pure gauge HMC** with Wilson action
- **Dynamical fermions** (Nf=2) with Wilson or Wilson-clover Dirac operator
- **GPU-accelerated**: All computations run on CUDA-enabled GPUs
- **Symplectic integrators**: Leapfrog (2nd order) and OMF4 (4th order)
- **Reunitarization**: SU(3) projection via polar decomposition
- **qcd_ml compatible**: Uses Dirac operators from [qcd_ml](https://github.com/mcgill-a2c2/qcd_ml)

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.11, PyTorch with CUDA support, and `qcd_ml`:

```bash
pip install torch qcd_ml
```

## Quick Start

Pure gauge simulation on GPU:

```python
import torch
from HMCpy import hmc_step, plaquette_average, reunitarize

# Initialize SU(3) gauge field on GPU
U = torch.eye(3, dtype=torch.cdouble).expand(4, 8, 8, 8, 8, 3, 3).to('cuda')

# HMC parameters
beta = 6.0
n_traj = 1000

for traj in range(n_traj):
    U, accepted, dH = hmc_step(
        U,
        beta=beta,
        n_steps=20,
        step_size=0.1,
        integrator="omf4",
    )
    print(f"{traj}: accepted={accepted}, dH={dH:.4f}, plaquette={plaquette_average(U).item():.6f}")
    
    if traj % 10 == 0:
        U = reunitarize(U)
```

Dynamical Wilson fermions:

```python
U, accepted, dH = hmc_step(
    U,
    beta=6.0,
    n_steps=20,
    step_size=0.05,
    dynamic=True,
    mass_parameter=-0.5,
    csw=1.0,  # Clover coefficient (set to 0 for plain Wilson)
)
```

## Structure

- `src/HMCpy/physics.py` — Wilson gauge action, force, Hamiltonian, reunitarization
- `src/HMCpy/fermion.py` — Pseudofermion action and force for Wilson/Wilson-clover
- `src/HMCpy/integrator.py` — Leapfrog and OMF4 symplectic integrators
- `src/HMCpy/monte_carlo.py` — HMC step with accept/reject and momentum refresh
- `src/HMCpy/utility.py` — SU(3) generators and exponential update

## License

MIT
