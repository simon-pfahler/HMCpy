import numpy as np
import torch

from HMCpy import hmc_step, plaquette_average, reunitarize

U = torch.eye(3, dtype=torch.cdouble).expand(4, 2, 2, 2, 2, 3, 3)

n_traj = 100000
beta = 6.0

plaquette_averages = np.zeros(n_traj)

for traj in range(n_traj):
    U, accepted, dH = hmc_step(
        U,
        beta=beta,
        n_steps=200,
        step_size=0.001,
        integrator="leapfrog",
    )

    plaquette_averages[traj] = plaquette_average(U).item()
    print(f"{traj} - {accepted}\t{dH}\t{plaquette_averages[traj]}")

    if traj % 20 == 0:
        U = reunitarize(U)
        np.savetxt(f"plaquettes_beta{beta}.dat", plaquette_averages[: traj + 1])
