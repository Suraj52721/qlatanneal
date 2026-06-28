import numpy as np
from qanneal import DenseIsing, ParallelTemperingAnnealer

n = 5
h = np.zeros(n, dtype=float)
J = np.zeros((n, n), dtype=float)
for i in range(n - 1):
    J[i, i + 1] = 0.3
    J[i + 1, i] = 0.3

ham = DenseIsing(h, J)

betas = [0.2, 0.4, 0.8, 1.2]

annealer = ParallelTemperingAnnealer(ham, betas)
result = annealer.run(sweeps_per_step=5, steps=20, swap_interval=1)

print("Best energy:", result.best_energy)
print("Swap acceptance (last):", result.swap_acceptance_trace[-1])
