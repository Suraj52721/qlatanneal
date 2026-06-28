import numpy as np
from qanneal import DenseIsing, AnnealSchedule, ReplicaAnnealer

n = 4
h = np.array([0.1, -0.2, 0.0, 0.3], dtype=float)
J = np.zeros((n, n), dtype=float)
J[0, 1] = 0.5
J[1, 0] = 0.5
J[2, 3] = -0.4
J[3, 2] = -0.4

ham = DenseIsing(h, J)
schedule = AnnealSchedule.linear(0.1, 2.0, 30)

annealer = ReplicaAnnealer(ham, schedule, replicas=4)
result = annealer.run(50)

print("Global best energy:", result.global_best_energy)
print("Average energy trace length:", len(result.average_energy_trace))
