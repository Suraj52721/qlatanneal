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
4. Advance J_⊥ (see **Budget calibration** below — SQAPT precomputes the whole trajectory; the
   legacy fixed-`ε̃` path advances online by `J_⊥ += ε̃ · χ_B^{−α}`).
5. Repeat for all `num_steps` steps (SQAPT no longer stops early — see termination notes).

---

## Budget calibration (SQAPT) — *the* fix for the frozen schedule

> **Background / why this exists.** Before v0.4.2 the SQAPT optimal schedule used a hand-picked
> constant `ε̃` (e.g. `0.05`, or `base_eps·max(1, j_rms)` in benchmark drivers). Nothing tied `ε̃`
> to the step budget, so on problems where χ_B was large and roughly flat the per-step `ΔJ_⊥`
> was ~0.002 and J_⊥ **barely moved over the entire run** (e.g. 2.88 → 3.14 over 150 steps when it
> should have spanned 2.88 → 20.5). The schedule was effectively running at constant J_⊥ — *not
> annealing* — which silently invalidated every `sqapt-opt` vs `sqapt-std` comparison. This is the
> bug the calibration below removes.

`ε̃` is only a **scale**; the *shape* of the schedule (slow where χ_B is large, fast where it is
small) comes entirely from `χ_B(J)^{−α}`. The scale must be fixed by the step budget, not guessed.

**Pilot pre-pass.** When `eps_tilde <= 0` (the default), `run_optimal` first measures the
profile `χ_B(J)` at `calib_probes` points spanning `[j_perp_start, j_perp_end]` (`calib_sweeps`
sweeps each, on scratch states so the real run is unbiased). The same sampler the run uses
produces these probes.

**Budget-correct ε̃.** From the discrete update `J_{t+1} = J_t + ε̃·χ_B(J_t)^{−α}`, the number of
steps to cross the range is `T = (1/ε̃)·∫ χ_B(J)^{α} dJ`. Hence

```
ε̃  =  (1/T) · ∫_{J0}^{Jend} χ_B(J)^{α} dJ        (α POSITIVE; midpoint rule over the probes)
```

> ⚠️ **Note the +α and the /T.** The naive form `ε̃ = ΔJ⊥ / Σ_m χ_B(J_m)^{−α}` (χ_B to the
> *negative* power, no `/T`) does **not** consume the budget when χ_B varies — it under/overshoots
> and re-freezes. The integral form above is what makes the traversal budget-correct.

**Profile-driven trajectory (robust to profile shape).** Rather than advance J_⊥ online with the
noisy, path-dependent within-step variance (which lets the run order faster than the pilot, inflate
`χ_B^{−α}`, and leap to the end in a few steps), the SQAPT run **precomputes the entire
`J_⊥(step)` trajectory** by inverting the cumulative integral

```
G(J) = ∫_{J0}^{J} χ_B(J')^{α} dJ' ;   step t is placed at  J = G⁻¹( (t/(T−1)) · G(Jend) )
```

This guarantees, *by construction and independent of the χ_B shape*, that the schedule (a) consumes
all `num_steps`, (b) reaches `j_perp_end`, and (c) dwells where χ_B is large. The online χ_B is
still measured every step and recorded (`chi_B_online` in the debug CSV) as a diagnostic.

**Per-instance, not global.** Because χ_B's scale depends on `n`, `M`, and the coupling
distribution, `ε̃` is recalibrated for every run. A global hardcoded constant is fundamentally wrong.

**Diagnostics.** `result.calibrated_eps_tilde`, `result.j_perp_start`, `result.resolved_j_perp_end`,
`result.final_j_perp`, `result.j_perp_trace`, and `result.chi_B_trace` are populated. Pass
`debug_csv_path=...` (Python: `optimal_debug_csv=...`) for a per-step CSV — see **Pre-flight check**.



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

The overall adiabaticity scale. **Default `0.0`, which requests budget calibration** (see
*Budget calibration* above) — the recommended path for SQAPT. A **positive** value is treated as an
explicit override that skips calibration and drives the schedule online with that fixed scale
(legacy behavior; SQA `run_optimal` still requires `eps_tilde > 0` as it has no calibration).

If you must set it by hand, the total J_⊥ change per run is approximately
`ε̃ · num_steps · χ̄_B^{−α}` where χ̄_B is the average susceptibility; to span `jp_end − jp_start`:

```
ε̃_target  ≈  (jp_end − jp_start) / (num_steps · χ̄_B^{−α})
```

Prefer calibration (`eps_tilde=0.0`) over this estimate — a frozen schedule from a too-small `ε̃`
is exactly the bug calibration removes.

### `calib_probes`, `calib_sweeps` (SQAPT)

Pilot grid size and sweeps-per-probe for budget calibration. Defaults `12` and `10`. More probes
sharpen the χ_B(J) profile (and thus the trajectory) at the cost of a short pre-pass.

### `debug_csv_path` / `optimal_debug_csv` (SQAPT)

Path for the per-step diagnostic CSV
(`phase,step_index,j_perp,chi_B,delta_j,eps_tilde,alpha,floor_hit,chi_B_online`). `phase=calib`
rows are the pilot probes (negative step index); `phase=run` rows are the realized schedule.
Consumed by `scripts/plot_schedule_trajectory.py`.

### `alpha` (α)

| Universality class | System | α |
|-------------------|--------|---|
| 1-D quantum Ising | Transverse-field Ising chain (default SQA) | 15/14 ≈ 1.071 |
| 2-D quantum Ising | 2-D Ising with transverse field | 7/4 = 1.75 |
| Mean-field (SK model) | Sherrington–Kirkpatrick | α = 1.0 (tentative) |

For combinatorial optimization, `15/14` is a reasonable default even when the exact universality
class is unknown.

### `num_steps` and termination

The SQAPT optimal run **always executes all `num_steps`** (it does *not* stop early when J_⊥
reaches `j_perp_end`; once saturated it keeps sweeping at the endpoint). This keeps the total sweep
budget identical to the standard schedule — essential for a fair `opt`-vs-`std` comparison.
`result.final_j_perp` reports the J_⊥ actually reached (≈ `resolved_j_perp_end` when calibrated).

### The χ_B floor (`min_chi_B`)

`χ_B^{−α}` blows up where χ_B → 0, so a floor caps the step. **Empirically χ_B is *large* near the
quantum transition and collapses toward 0 in the *ordered* phase — it does not diverge at
criticality for this model** — so the largest steps (hence the floor's job) occur deep in the
ordered phase, not at the critical peak. When calibrating, the floor is set relative to the *max*
probed χ_B (`1e-4 · max χ_B`) rather than the old fixed `1e-4 · nM`, so it only bites where χ_B is
genuinely tiny. `floor_hit` in the debug CSV records exactly where it activated — verify it is the
ordered region, not the critical window.

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

### `j_perp_end` resolution and the silent-fallback footgun

For SQAPT, `solve(...)` now resolves `j_perp_end` (when you don't pass `optimal_j_perp_end`) to a
physically-motivated **`max(j_perp_start, 5·j_rms)`** where `j_rms = sqrt(mean J_ij²)` over the
non-zero couplings (`qanneal.solver.j_rms_from_problem`). The quantum critical J_⊥ scales like
`j_rms`, so `5·j_rms` sits well past the transition.

The C++ `run_optimal` still accepts the `j_perp_end <= 0` **sentinel**, which defaults to the
coldest replica's coupling. If that resolves *below* `j_perp_start`, the system starts past
criticality and the schedule has no range to anneal; the code installs a finite fallback ceiling
`j_perp_start + 1e6` **and now prints a `cerr` warning** (previously silent — that silence is how
the original frozen-schedule bug went undetected). **For benchmark runs, always pass an explicit
`j_perp_end` and only rely on the sentinel for genuinely undefined cases.**

> **Key physics constraint (when does opt help at all?).** The schedule only has room to act when
> `j_rms > j_perp_start (≈2.88)`. If `j_rms ≪ 2.88` (e.g. sparse 3-regular MWIS, SK at large n) the
> system is already classical at the start, χ_B ≈ 0, and `opt` cannot beat `std`. wmaxcut
> (`G(n,0.5)`, `J~U[1,10]`, `j_rms≈4–6`) is in the favorable regime. `scripts/sanity_schedule.py`
> flags the degenerate classes as `DEGENERATE` rather than `PASS`.

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
- Flat / linear `J_⊥(t)` (no dwell) → the frozen-schedule bug. With calibration this should not
  happen; if it does, check that `j_perp_end > j_perp_start` (look for the `cerr` fallback warning).
- `ΔJ_⊥` constant (no critical-point dip) → `replicas` or `sweeps_per_step` too small for reliable χ_B.
- `final_j_perp` ≪ `resolved_j_perp_end` on a non-degenerate problem → calibration failed; inspect the CSV.

---

## Pre-flight check (mandatory before any large-n HPC job)

Nothing in the pipeline used to visualize the realized `J_⊥(t)` curve, so the frozen schedule was
only discovered after a multi-hour HPC run. **Before submitting any large-n benchmark job, run a
fast sanity instance and inspect the trajectory.** A frozen/linear `J_⊥(t)` is the bug's signature
— do not proceed if you see it.

```bash
# Generates mwis_3reg / wmaxcut / sk at n=40, checks each trajectory, writes CSVs + plots.
python scripts/sanity_schedule.py --n 40 --steps 150 --plot
# Or plot a single run's debug CSV:
python scripts/plot_schedule_trajectory.py /tmp/sched_sanity/wmaxcut_n40.csv
```

A healthy trajectory visibly shows: (a) fast movement away from the critical region, (b) a
pronounced slowdown/dwell near `J_⊥^c ≈ j_rms`, (c) fast movement again past the transition, and
(d) actually reaching `j_perp_end` within the step budget. `sanity_schedule.py` prints `PASS` only
when the run consumes the full budget, reaches the end, and front-loads its steps; it prints
`DEGENERATE` for classes with no quantum range (`j_rms ≲ j_perp_start`) and exits non-zero on `FAIL`.

---

## Benchmark comparison: separating the schedule effect from cluster sweeps

The adaptive *schedule* and Swendsen–Wang *cluster sweeps* are two independent mechanisms. To
attribute a measured gain correctly, the comparison table must keep them separate:

| Comparison | Cluster sweeps | Isolates |
|------------|----------------|----------|
| `sqapt-std` vs `sqapt-opt` | off (`cluster_sweeps=0`) | **the schedule effect alone** |
| `sqapt-sw-std` vs `sqapt-sw-opt` | on (`cluster_sweeps>0`) | schedule + cluster-sweep interaction |

**Attribution rule.** Only credit "the optimal schedule" if `sqapt-opt` beats `sqapt-std`
(cluster sweeps OFF) reproducibly, now that the calibration fix is in place. If only the `-sw-`
variants improve, the gain belongs to the schedule×cluster-sweep interaction, not the schedule in
isolation, and the claim must be framed accordingly. (Pre-fix, the frozen schedule meant any
`-sw-opt` gain was almost certainly cluster sweeps supplying more decorrelated χ_B samples at a
near-constant J_⊥ — not the intended slow-near-criticality mechanism.)

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

# Optimal schedule with budget calibration (eps auto-calibrated; leave optimal_eps_tilde at 0.0)
r_opt = solve(problem, method="sqapt", replicas=8, reads=10, progress=False,
              schedule_type="optimal", optimal_num_steps=150, cluster_sweeps=1,
              optimal_debug_csv="/tmp/opt_sched.csv")

print(f"Standard best energy: {r_std.best_energy:.4f}")
print(f"Optimal  best energy: {r_opt.best_energy:.4f}")
# Inspect the realized trajectory (pre-flight):
#   python scripts/plot_schedule_trajectory.py /tmp/opt_sched.csv
```

---

## Related

- `docs/api.md` — `run_optimal()` parameter reference
- `docs/user_guide.md` — Section 9: Optimal Adaptive Schedule
- `python/qanneal/solver.py` — `j_perp_from_beta_gamma()`, `optimal_j_perp_params()`, `solve()` dispatch
- `src/sqa_annealer.cpp` — `SQAAnnealer::run_optimal()` C++ implementation
- `src/sqa_parallel_tempering.cpp` — `SQAParallelTemperingAnnealer::run_optimal()` C++ implementation
