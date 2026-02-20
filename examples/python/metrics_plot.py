import numpy as np
from qanneal import DenseIsing, AnnealSchedule, Annealer, MetricsObserver

n = 6
h = np.zeros(n, dtype=float)
J = np.zeros((n, n), dtype=float)
for i in range(n - 1):
    J[i, i + 1] = 0.2
    J[i + 1, i] = 0.2

ham = DenseIsing(h, J)
schedule = AnnealSchedule.linear(0.1, 2.0, 40)

obs = MetricsObserver()
annealer = Annealer(ham, schedule)
result = annealer.run(50, obs)

print("Best energy:", result.best_energy)
print("Recorded steps:", len(obs.energy_trace))

try:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(obs.energy_trace, label="energy")
    plt.plot(obs.magnetization_trace, label="magnetization")
    plt.xlabel("step")
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception:
    print("matplotlib not available; skipping plot")
