#!/usr/bin/env python3
"""Plot the realized J_perp(t) trajectory of the optimal SQAPT schedule.

Consumes the debug CSV written by SQAParallelTemperingAnnealer.run_optimal(..., debug_csv_path=...)
(equivalently solve(..., schedule_type="optimal", optimal_debug_csv=...)). The CSV columns are:

    phase,step_index,j_perp,chi_B,delta_j,eps_tilde,alpha,floor_hit

Rows with phase=="calib" are the pilot probe points (step_index < 0); phase=="run" rows are the
realized per-step schedule.

This is the pre-flight artifact that would have caught the frozen-schedule bug in minutes. A
healthy trajectory shows J_perp dwelling where chi_B is large (near criticality) and moving fast
where chi_B is small, and REACHING j_perp_end within the step budget. A flat/linear J_perp(t) is
the signature of the bug (see docs/optimal_schedule.md, "Pre-flight check").

Usage:
    python scripts/plot_schedule_trajectory.py sched.csv [-o out.png] [--show]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _read(path: Path):
    run = {"step": [], "j_perp": [], "chi_B": [], "delta_j": [], "floor_hit": []}
    calib = {"j_perp": [], "chi_B": []}
    eps_tilde = None
    alpha = None
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["phase"] == "calib":
                calib["j_perp"].append(float(row["j_perp"]))
                calib["chi_B"].append(float(row["chi_B"]))
            else:
                run["step"].append(int(row["step_index"]))
                run["j_perp"].append(float(row["j_perp"]))
                run["chi_B"].append(float(row["chi_B"]))
                run["delta_j"].append(float(row["delta_j"]))
                run["floor_hit"].append(int(float(row["floor_hit"])))
                eps_tilde = float(row["eps_tilde"])
                alpha = float(row["alpha"])
    return run, calib, eps_tilde, alpha


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="debug CSV from run_optimal(debug_csv_path=...)")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output PNG (default: <csv>.png)")
    ap.add_argument("--show", action="store_true", help="also display interactively")
    args = ap.parse_args(argv)

    if not args.csv.exists():
        ap.error(f"CSV not found: {args.csv}")

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"matplotlib required: {exc}", file=sys.stderr)
        return 2

    run, calib, eps_tilde, alpha = _read(args.csv)
    if not run["step"]:
        print("No phase=run rows in CSV.", file=sys.stderr)
        return 2

    fig, (ax_j, ax_c) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # --- J_perp(t) ---
    ax_j.plot(run["step"], run["j_perp"], "-", color="C0", lw=1.8, label="J_perp(t)")
    floored = [s for s, f in zip(run["step"], run["floor_hit"]) if f]
    floored_y = [j for j, f in zip(run["j_perp"], run["floor_hit"]) if f]
    if floored:
        ax_j.plot(floored, floored_y, ".", color="C3", ms=5,
                  label=f"chi_B floor active ({len(floored)} steps)")
    j_end = max(run["j_perp"])
    ax_j.axhline(j_end, color="gray", ls="--", lw=0.8, label=f"reached max = {j_end:.3g}")
    ax_j.set_ylabel("J_perp")
    title = "Realized optimal SQAPT schedule"
    if eps_tilde is not None:
        title += f"   (eps_tilde={eps_tilde:.3g}, alpha={alpha:.3g})"
    ax_j.set_title(title)
    ax_j.legend(loc="lower right", fontsize=8)
    ax_j.grid(alpha=0.3)

    # --- chi_B(t), with calibration profile overlaid against J on a twin axis ---
    ax_c.semilogy(run["step"], run["chi_B"], "-", color="C1", lw=1.5, label="chi_B(t) [run]")
    ax_c.set_ylabel("chi_B = Var(B)")
    ax_c.set_xlabel("step index")
    ax_c.legend(loc="upper right", fontsize=8)
    ax_c.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = args.out or args.csv.with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
