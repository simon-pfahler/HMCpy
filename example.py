import torch

from HMCpy import hmc_step, plaquette_average, reunitarize

U = torch.randn(4, 2, 2, 2, 2, 3, 3, dtype=torch.cdouble)

n_traj = 100

for traj in range(n_traj):
    U, accepted, dH = hmc_step(
        U,
        beta=6.0,
        n_steps=20,
        step_size=0.05,
    )

    print(f"{traj} - {plaquette_average(U)}")

    if traj % 20 == 0:
        U = reunitarize(U)
