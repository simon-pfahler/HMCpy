import numpy as np
import torch

from HMCpy import hmc_step, plaquette_average, reunitarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

U = torch.eye(3, dtype=torch.cdouble).expand(4, 2, 2, 2, 2, 3, 3).to(device)

n_traj = 100000
beta = 6.0
mass = -0.5
n_steps = 20
step_size = 0.1

plaquette_averages = np.zeros(n_traj)

for traj in range(n_traj):
    U, accepted, dH = hmc_step(
        U,
        beta=beta,
        n_steps=n_steps,
        step_size=step_size,
        integrator="omf4",
        dynamic=True,
        mass_parameter=mass,
        csw=1.0,
    )

    plaquette_averages[traj] = plaquette_average(U).item()
    print(f"{traj} - {accepted}\t{dH}\t{plaquette_averages[traj]}")

    if traj % 20 == 0:
        U = reunitarize(U)
        np.savetxt(
            f"plaquettes_beta{beta}_mass{mass}.dat",
            plaquette_averages[: traj + 1],
        )
