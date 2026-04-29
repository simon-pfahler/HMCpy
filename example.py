import torch

from HMCpy import hmc_step, plaquette_average, reunitarize

U = torch.eye(3, dtype=torch.cdouble).expand(4, 2, 2, 2, 2, 3, 3)

n_traj = 100

for traj in range(n_traj):
    U, accepted, dH = hmc_step(
        U,
        beta=6.0,
        n_steps=100,
        step_size=0.01,
    )

    print(f"{traj} - {accepted}, {dH}, {plaquette_average(U).item()}")

    if traj % 20 == 0:
        U = reunitarize(U)
