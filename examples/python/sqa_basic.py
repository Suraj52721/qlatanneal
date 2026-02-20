import numpy as np
from qanneal import DenseIsing, SQASchedule, SQAAnnealer

n = 3
h = np.array([0.0, 0.0, 0.0], dtype=float)
J = np.array([
    [0.0, 0.5, -0.3],
    [0.5, 0.0, 0.2],
    [-0.3, 0.2, 0.0],
], dtype=float)

ham = DenseIsing(h, J)

betas = [0.2, 0.5, 1.0, 1.5]
gammas = [2.0, 1.5, 1.0, 0.5]
schedule = SQASchedule.from_vectors(betas, gammas)

annealer = SQAAnnealer(ham, schedule, trotter_slices=8, replicas=2)
result = annealer.run(sweeps_per_beta=10, worldline_sweeps=2)

print("Best energy:", result.best_energy)
print("Energy trace length:", len(result.energy_trace))
