import torch

from HMCpy import Qcd_ml_DiracWilson, hmc_step, plaquette_average, reunitarize

D = Qcd_ml_DiracWilson(mass_parameter=0.1)

U = torch.randn(4, 2, 2, 2, 2, 3, 3, dtype=torch.cdouble)

n_traj = 100

for traj in range(n_traj):
    U, accepted, dH = hmc_step(
        U,
        beta=6.0,
        n_steps=10,
        step_size=0.1,
        dirac_op=D,
        integrator="omf2",
    )

    print(plaquette_average(U))

    if traj % 20 == 0:
        U = reunitarize(U)
