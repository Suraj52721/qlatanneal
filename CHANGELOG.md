# Changelog

All notable changes to **qanneal** are documented here. This project follows semantic
versioning (`MAJOR.MINOR.PATCH`).

## 0.6.1

Makes the single-temperature SQA optimal schedule actually competitive.

- **β selection fix** in `optimal_j_perp_params`: `beta_end = max(4 / j_rms, 1.0)` so the cold
  phase satisfies `β · j_rms ≥ 4`. The previous `-log(p_end)/scale` collapsed to `β ≈ 0.02`
  (near-infinite temperature) for strong-coupling problems, so the classical spins never
  committed and `sqa` + `schedule_type="optimal"` returned poor solutions.
- **Two-phase SQA protocol** in `SQAAnnealer.run_optimal`: phase 1 ramps β from
  `beta_ramp_start` (0.1) to the cold `beta` at fixed J⊥ (thermal anneal) for
  `beta_ramp_fraction·num_steps` steps (default 0.3); phase 2 fixes β and runs the calibrated
  adaptive J⊥ schedule. Exposed via `solve(optimal_beta_ramp_fraction=...)`.
- `DenseIsing.couplings()` and `DenseIsing.fields()` accessors; `j_rms_from_problem()` now
  reads couplings from a `DenseIsing` (previously returned `None` and hit the fallback).

Result: SQA-opt goes from losing to winning on hard frustrated Ising problems, with the
advantage growing with problem size (see `benchmarks/schedule/`).

## 0.6.0

- Ported the budget calibration + inverse-CDF trajectory from SQAPT into
  `SQAAnnealer.run_optimal` (previously SQA required `eps_tilde > 0` and used the frozen
  online update).

## 0.5.0

Fixes the "frozen schedule" bug in the optimal adaptive J⊥ schedule.

- **Budget calibration**: `eps_tilde <= 0` (now the default in `solve`) requests a per-instance
  pilot pre-pass that sets `eps_tilde = (1/num_steps)·∫ χ_B(J)^α dJ` and drives J⊥ along the
  inverse-CDF of that integral. Guarantees the schedule traverses `[j_perp_start, j_perp_end]`
  in exactly `num_steps` and dwells where χ_B is large — the previous fixed-`ε̃` update could
  barely move J⊥ when χ_B was large.
- No early break: the run always consumes its full `num_steps` for a fair opt-vs-standard
  sweep budget.
- New result fields: `chi_B_trace`, `calibrated_eps_tilde`, `j_perp_start`,
  `resolved_j_perp_end`, `final_j_perp`. New `debug_csv_path` for per-step diagnostics.
- New `calib_probes`, `calib_sweeps` parameters; `j_rms_from_problem()` helper; `solve()`
  resolves `j_perp_end = max(j_perp_start, 5·j_rms)` when couplings are inspectable.
- Corrected physics note: χ_B is *large* near the quantum transition and collapses in the
  ordered phase (it does not diverge).

## 0.4.0

- Initial optimal adaptive J⊥ schedule: `SQAAnnealer.run_optimal`,
  `SQAParallelTemperingAnnealer.run_optimal`, `solve(..., schedule_type="optimal")`,
  `j_perp_from_beta_gamma()`, `optimal_j_perp_params()`, `SQAResult.j_perp_trace`.
