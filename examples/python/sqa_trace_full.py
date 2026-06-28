import os
import numpy as np
import matplotlib

# Avoid PyCharm backend crash (FigureCanvasInterAgg.tostring_rgb)
# Force a non-interactive backend in IDEs/headless.
if os.environ.get("PYCHARM_HOSTED") == "1":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qanneal import QUBO, SQASchedule, SQAAnnealer, SQAStateTraceObserver


def make_nontrivial_qubo(n, seed=7):
    rng = np.random.default_rng(seed)
    # Base random symmetric couplings
    Q = rng.normal(scale=0.2, size=(n, n))
    Q = 0.5 * (Q + Q.T)
    # Add block structure for non-triviality
    b = n // 2
    block = rng.normal(scale=0.6, size=(b, b))
    Q[:b, :b] += 0.5 * (block + block.T)
    block2 = rng.normal(scale=0.6, size=(n - b, n - b))
    Q[b:, b:] += 0.5 * (block2 + block2.T)
    # Diagonal biases
    diag = rng.uniform(-1.5, 1.5, size=n)
    Q[np.diag_indices(n)] += diag
    return Q


def main():
    # Problem size and SQA settings
    n = 20
    trotter_slices = 16
    replicas = 4
    steps = 40
    sweeps_per_beta = 10
    worldline_sweeps = 3

    # Build a non-trivial QUBO and convert to Ising
    Q = make_nontrivial_qubo(n)
    qubo = QUBO(Q)
    ising = qubo.to_ising()

    # Schedule (beta up, gamma down)
    betas = np.linspace(0.1, 4.0, steps).tolist()
    gammas = np.linspace(5.0, 0.01, steps).tolist()
    schedule = SQASchedule.from_vectors(betas, gammas)

    # Trace observer (records every sweep)
    obs = SQAStateTraceObserver()
    obs.stride = 1

    annealer = SQAAnnealer(ising, schedule, trotter_slices, replicas, backend="cpu")
    result = annealer.run(sweeps_per_beta, worldline_sweeps, observer=obs)

    print("Best energy:", result.best_energy)
    print("Records:", len(obs.state_trace))
    print("Replicas:", obs.replicas, "Slices:", obs.slices, "Spins:", obs.spins)

    # Convert traces to numpy for plotting
    states = np.array(obs.state_trace, dtype=np.int8)
    states = states.reshape(len(obs.state_trace), obs.replicas, obs.slices, obs.spins)

    avg_energy = np.array(obs.avg_energy_trace)
    replica_energy = np.array(obs.replica_energy_trace)
    beta_trace = np.array(obs.beta_trace)
    gamma_trace = np.array(obs.gamma_trace)
    step_trace = np.array(obs.step_trace)
    phase_trace = np.array(obs.phase_trace)

    # Magnetization over time (avg over replicas/slices/spins)
    magnetization = states.mean(axis=(1, 2, 3))

    # Plot 1: schedule
    plt.figure()
    plt.plot(range(steps), betas, label="beta")
    plt.plot(range(steps), gammas, label="gamma")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Anneal Schedule")
    plt.legend()

    # Plot 2: average energy per recorded sweep
    plt.figure()
    plt.plot(avg_energy, label="avg_energy")
    plt.xlabel("Record index")
    plt.ylabel("Energy")
    plt.title("Average Energy Trace (per sweep)")
    plt.legend()

    # Plot 3: replica energies
    plt.figure()
    for r in range(replica_energy.shape[1]):
        plt.plot(replica_energy[:, r], label=f"replica {r}")
    plt.xlabel("Record index")
    plt.ylabel("Energy")
    plt.title("Replica Energy Trace")
    plt.legend()

    # Plot 4: magnetization trace
    plt.figure()
    plt.plot(magnetization, label="magnetization")
    plt.xlabel("Record index")
    plt.ylabel("Magnetization")
    plt.title("Magnetization (avg over replicas/slices/spins)")
    plt.legend()

    # Plot 5: heatmap of a single replica+slice over time
    r0 = 0
    t0 = 0
    heat = states[:, r0, t0, :]
    plt.figure()
    plt.imshow(heat.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="spin")
    plt.xlabel("Record index")
    plt.ylabel("Spin index")
    plt.title(f"State Trace Heatmap (replica={r0}, slice={t0})")

    # Plot 6: phase indicator (slice vs worldline) with beta/gamma
    plt.figure()
    plt.plot(beta_trace, label="beta")
    plt.plot(gamma_trace, label="gamma")
    plt.plot(phase_trace, label="phase (0 slice, 1 worldline)", alpha=0.6)
    plt.xlabel("Record index")
    plt.title("Beta / Gamma / Phase per Record")
    plt.legend()

    # Save trace arrays for offline analysis
    np.savez(
        "sqa_trace_output.npz",
        states=states,
        avg_energy=avg_energy,
        replica_energy=replica_energy,
        magnetization=magnetization,
        beta_trace=beta_trace,
        gamma_trace=gamma_trace,
        step_trace=step_trace,
        phase_trace=phase_trace,
        sweeps=np.array(obs.sweep_trace),
        replicas=np.array(obs.replica_trace),
    )

    # In PyCharm, interactive show can fail with FigureCanvasInterAgg.
    # Save figures and only show if an interactive backend is available.
    out_dir = "sqa_trace_figs"
    os.makedirs(out_dir, exist_ok=True)
    for i, fig_num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(fig_num)
        fig.savefig(os.path.join(out_dir, f"figure_{i}.png"), dpi=150, bbox_inches="tight")

    backend = matplotlib.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        print(f"Saved figures to: {out_dir}/")


if __name__ == "__main__":
    main()
