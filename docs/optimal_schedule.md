# Optimal Adaptive J⊥ Schedule

Technical reference for the locally-adiabatic Trotter coupling schedule introduced in qanneal 0.4.0.

---

## Background

Standard SQA uses a pre-designed Γ(t) ramp — the transverse field decreases from a large quantum
value to zero according to a geometric or linear schedule determined before the run starts. This
schedule cannot adapt to the actual quantum fluctuations the system is experiencing, so it may
rush through the quantum critical point (causing diabatic transitions and poor solutions) or waste
time steps in regions far from criticality.

The **local adiabaticity condition** (Roland & Cerf 2002) gives a rigorous prescription for how
fast you can sweep a control parameter while keeping the system in the instantaneous ground state.
For a quantum annealer approaching its quantum critical point from the disordered side:

```
dΓ/dt  ∝  gap²(Γ)
```

where gap is the spectral gap above the ground state. Near a second-order quantum phase transition
the gap vanishes as a power law of the distance from the critical field Γ_c, so the schedule must
**slow exponentially near criticality**.

In the Suzuki–Trotter (SQA) representation the natural control parameter is not Γ directly but the
Trotter coupling:

```
J_⊥(β, Γ, M)  =  (1/2) ln(1 / tanh(βΓ/M))
```

J_⊥ encodes the imaginary-time bond strength between adjacent slices. Large J_⊥ → strong quantum
fluctuations; small J_⊥ → classical limit.

---

## Algorithm

### Adaptive ODE (Eq. 48 of the derivation)

```
ΔJ_⊥  =  ε̃ · χ_B^{-α}
```

where:

- **ε̃** (`eps_tilde`) — the overall adiabaticity parameter. Units: `[ΔJ_⊥ · χ_B^α]`. Smaller = more adiabatic.
- **χ_B** — bond susceptibility: `Var(B)` where `B = Σ_{k,i} s^k_i · s^{k+1}_i` (sum over imaginary-time bonds). χ_B diverges at the SQA quantum critical point J_⊥^c.
- **α** (`alpha`) — universality exponent: `α = z/(2−η) + 1/2`. Default `15/14` corresponds to the 1-D quantum Ising universality class (`z=1`, `η=1/4`). Use `α=7/4` for the 2-D quantum Ising class.

At each adaptive step:
1. Run `sweeps_per_step` + `worldline_sweeps` + `cluster_sweeps` sweeps.
2. After each full sweep, record the current bond sum B (updated incrementally — no O(nM) recomputation).
3. After all sweeps: `χ_B = Var({B samples from all sweeps × all replicas})`.
4. `J_⊥ += ε̃ · χ_B^{-α}` (clamped to `j_perp_end`).
5. Repeat until J_⊥ reaches `j_perp_end` or `num_steps` steps are exhausted.

### Why J_⊥ increases, not decreases

In qanneal J_⊥ **increases** from `j_perp_start` to `j_perp_end`. This is opposite to Γ which
decreases. The reason: `J_⊥ = 0.5·ln(1/tanh(βΓ/M))` is a monotonically **decreasing** function
of Γ. As Γ → 0 (classical limit), J_⊥ → ∞. In practice `j_perp_end` is capped at the value
corresponding to `gamma_end` of the equivalent standard schedule.

---

## χ_B Estimation

### Incremental bond tracking

B is tracked incrementally after each spin flip — O(1) update per flip instead of O(nM):

| Update type | ΔB formula |
|-------------|-----------|
| Metropolis flip of spin (replica r, slice t, site i) | `ΔB = −2·s·(s_prev + s_next)` where s_prev, s_next are the same-site spins in slices t−1, t+1, and s is the old spin value |
| Cluster flip (Swendsen–Wang along time axis) | `ΔB = −2 · Σ_{boundary bonds} seed_spin · neighbor_spin` — boundary bonds are those connecting the cluster to spins outside it |
| Worldline flip (same site, all slices flip together) | `ΔB = 0` — all imaginary-time bonds flip together so net change is zero |

### Pooled variance across sweeps and replicas

Each step produces `R × S` B-values where R = replicas, S = sweeps_per_step + cluster_sweeps.
These are pooled into a single variance estimate:

```
χ_B = E[B²] − (E[B])²
```

**Why pooling works well for R≥2 (near criticality)**:
Near J_⊥^c the autocorrelation time τ_int ≫ sweeps_per_step, so within a single replica all
S samples are nearly identical. The pooled variance collapses to the cross-replica variance:

```
Var_pooled ≈ (1/R) Σ_r (B_r − B̄)²  =  Var_cross_replica
```

This is exactly what the pre-0.4.0 single-snapshot estimator computed. Away from criticality,
τ_int ≪ sweeps_per_step so S independent samples per replica improve the estimate.

**Single-replica guard (R=1)**:
When R=1, near criticality all within-step samples are nearly identical → pooled variance → 0 →
maximum step → rushes through critical point. Guard prevents this by taking:

```
χ_B = max(pooled_var, per_bond_approx)
```

where `per_bond_approx = n·M·(1 − (B/nM)²)` is the variance of a single imaginary-time bond
averaged over the full lattice. This provides a positive lower bound on χ_B near criticality.

---

## Parameters

### `eps_tilde` (ε̃)

The single most important tuning parameter. Controls the total "speed" of the annealing.

Rule of thumb: the total J_⊥ change per run is approximately `ε̃ · num_steps · χ̄_B^{-α}` where
χ̄_B is the average susceptibility. To span a J_⊥ range of `jp_end − jp_start`:

```
ε̃_target  ≈  (jp_end − jp_start) / (num_steps · χ̄_B^{-α})
```

For most problems, χ̄_B ≈ nM/4 far from criticality (uncorrelated spins), giving a rough starting
estimate. In practice, start with `eps_tilde=0.05` and adjust based on whether J_⊥ reaches
`j_perp_end` within `num_steps`.

### `alpha` (α)

| Universality class | System | α |
|-------------------|--------|---|
| 1-D quantum Ising | Transverse-field Ising chain (default SQA) | 15/14 ≈ 1.071 |
| 2-D quantum Ising | 2-D Ising with transverse field | 7/4 = 1.75 |
| Mean-field (SK model) | Sherrington–Kirkpatrick | α = 1.0 (tentative) |

For combinatorial optimization, `15/14` is a reasonable default even when the exact universality
class is unknown.

### `num_steps` and termination

The run terminates at the **first** of:
1. J_⊥ reaches `j_perp_end` (early termination — schedule completed).
2. `num_steps` adaptive steps have been taken.

`result.j_perp_trace` records which case occurred: if the last value equals `j_perp_end`, the
schedule finished early.

---

## Choosing `j_perp_start` and `j_perp_end`

Use `optimal_j_perp_params()` to auto-compute from the problem's energy scale:

```python
from qanneal import optimal_j_perp_params

beta, jp_start, jp_end = optimal_j_perp_params(problem, mode="balanced", trotter_slices=32)
```

Under the hood this calls `auto_schedule_sqa_tuned()` to estimate `(beta_end, gamma_start, gamma_end)`
from the problem's 75th-percentile |ΔE|, then maps through `j_perp_from_beta_gamma()`:

- `beta = beta_end` of the equivalent standard schedule (cold regime)
- `jp_start = j_perp_from_beta_gamma(beta, gamma_start, M)` — quantum end
- `jp_end   = j_perp_from_beta_gamma(beta, gamma_end, M)` — classical end

You can also set them manually. Physical constraints:
- `jp_start > 0` (must start in quantum regime)
- `jp_end > jp_start` (schedule must move toward classical limit)
- `jp_start` should correspond to Γ ≈ 1–5× the coupling scale so tunneling is active
- `jp_end` should correspond to very small Γ so the system is nearly classical at the end

---

## SQA vs SQAPT vs SQAPT+SW

| Method | `cluster_sweeps` | Notes |
|--------|-----------------|-------|
| SQA | 0 | Single β. Use `replicas≥4` for good χ_B estimate. Single-replica guard active for R=1. |
| SQA+SW | >0 | Swendsen–Wang cluster updates added. Better mixing in frustrated regimes. |
| SQAPT | 0 | Multiple (β,Γ) points. All share one J_⊥. PT swaps purely classical (Trotter terms cancel). |
| SQAPT+SW | >0 | SQAPT + cluster updates. Strongest mixing, recommended for hard instances. |

For SQAPT, the `betas` and `gammas` from the ladder set each replica's fixed temperature point.
The J_⊥ schedule sweeps from `jp_start` to `jp_end` at the shared coupling level, while each
replica independently explores the problem landscape at its own thermal regime.

---

## Interpreting the J⊥ Trace

```python
import numpy as np, matplotlib.pyplot as plt

trace = result.j_perp_trace      # list of floats, length = steps taken
etrace = result.energy_trace     # energy after each adaptive step

steps = np.arange(len(trace))
step_sizes = np.diff(trace)       # ΔJ_⊥ at each step

fig, axs = plt.subplots(3, 1, figsize=(8, 9))
axs[0].plot(steps, trace);          axs[0].set_ylabel("J_⊥")
axs[1].plot(steps[:-1], step_sizes);axs[1].set_ylabel("ΔJ_⊥ (step size)")
axs[2].plot(steps, etrace);         axs[2].set_ylabel("Energy")
for ax in axs: ax.set_xlabel("Adaptive step")
plt.suptitle("Optimal schedule diagnostics"); plt.tight_layout(); plt.show()
```

**Healthy run characteristics**:
- `ΔJ_⊥` plot shows a clear dip (small steps) in the middle — this is the critical point region.
- Energy drops most steeply when J_⊥ passes through the critical region.
- J_⊥ reaches `jp_end` before `num_steps` is exhausted.

**Warning signs**:
- Flat `ΔJ_⊥` throughout → `eps_tilde` too large or `alpha` wrong; χ_B is dominated by floor.
- `ΔJ_⊥` constant (no critical-point dip) → `replicas` or `sweeps_per_step` too small for reliable χ_B.
- J_⊥ never reaches `jp_end` → increase `eps_tilde` or `num_steps`.

---

## Example: Comparing Standard vs Optimal Schedule

```python
import numpy as np
from qanneal import solve, DenseIsing

rng = np.random.default_rng(0)
n = 50
J = np.triu(rng.standard_normal((n, n)), 1)
problem = DenseIsing(np.zeros(n), J)

# Standard schedule
r_std = solve(problem, method="sqapt", replicas=8, reads=10, progress=False)

# Optimal schedule
r_opt = solve(problem, method="sqapt", replicas=8, reads=10, progress=False,
              schedule_type="optimal", optimal_eps_tilde=0.02, optimal_num_steps=300,
              cluster_sweeps=1)

print(f"Standard best energy: {r_std.best_energy:.4f}")
print(f"Optimal  best energy: {r_opt.best_energy:.4f}")
```

---

## Related

- `docs/api.md` — `run_optimal()` parameter reference
- `docs/user_guide.md` — Section 9: Optimal Adaptive Schedule
- `python/qanneal/solver.py` — `j_perp_from_beta_gamma()`, `optimal_j_perp_params()`, `solve()` dispatch
- `src/sqa_annealer.cpp` — `SQAAnnealer::run_optimal()` C++ implementation
- `src/sqa_parallel_tempering.cpp` — `SQAParallelTemperingAnnealer::run_optimal()` C++ implementation
