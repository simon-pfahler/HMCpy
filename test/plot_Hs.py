import matplotlib.pyplot as plt
import torch

fine_step_size = 0.1
coarse_step_size = 2 * fine_step_size

fine = torch.load(f"./Hs{fine_step_size}.dat")
coarse = torch.load(f"./Hs{coarse_step_size}.dat")

plt.plot(torch.arange(0, fine.shape[0], 1), fine, label="fine")
plt.plot(torch.arange(0, fine.shape[0], 2), coarse, label="coarse")

plt.legend()
plt.show()
