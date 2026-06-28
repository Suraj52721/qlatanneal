#!/usr/bin/env python3
"""
merge_array_results.py  —  Merge SLURM job-array outputs from hpc_sqa_launcher.py.

After running a SLURM job array (run_sqa_array.sh), each task produces:
  results/run_<task_id>.json   and   results/run_<task_id>_spins.npy

This script:
  1. Reads all matching JSON files
  2. Finds the global best solution across all tasks
  3. Writes a merged results JSON and the best spin configuration
  4. Prints a summary table

Usage
-----
  python merge_array_results.py results/run_*.json
  python merge_array_results.py --out merged_best  results/run_*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


def _load_result(json_path: str) -> dict:
    with open(json_path) as f:
        d = json.load(f)
    d["_json_path"] = json_path
    # Try to load spins
    spins_path = d.get("spins_file") or json_path.replace(".json", "_spins.npy")
    if os.path.exists(spins_path):
        d["_spins"] = np.load(spins_path)
    return d


def merge(json_paths: list, out_prefix: str) -> None:
    if not json_paths:
        print("No JSON files found. Nothing to merge.", file=sys.stderr)
        sys.exit(1)

    results = []
    for p in sorted(json_paths):
        try:
            results.append(_load_result(p))
        except Exception as e:
            print(f"  SKIP {p}: {e}", file=sys.stderr)

    if not results:
        print("All files failed to load.", file=sys.stderr)
        sys.exit(1)

    # Global best
    best = min(results, key=lambda r: float(r["best_energy"]))
    all_energies = [float(e) for r in results for e in r.get("all_energies", [r["best_energy"]])]

    # Consistent metadata from the first record
    ref = results[0]

    total_reads = sum(int(r.get("reads", 1)) for r in results)

    merged = {
        "problem_type":  ref.get("problem_type"),
        "n_spins":       ref.get("n_spins"),
        "n_edges":       ref.get("n_edges"),
        "method":        ref.get("method"),
        "mode":          ref.get("mode"),
        "seed":          ref.get("seed"),
        "n_tasks":       len(results),
        "total_reads":   total_reads,
        "best_energy":   float(best["best_energy"]),
        "best_from":     best["_json_path"],
        "energy_mean":   float(np.mean(all_energies)),
        "energy_std":    float(np.std(all_energies)),
        "energy_min":    float(np.min(all_energies)),
        "energy_max":    float(np.max(all_energies)),
        "all_energies":  all_energies,
        "best_spins_file": f"{out_prefix}_spins.npy",
    }

    # Write outputs
    json_out = f"{out_prefix}.json"
    with open(json_out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged JSON  → {json_out}")

    if "_spins" in best:
        spins_out = f"{out_prefix}_spins.npy"
        np.save(spins_out, best["_spins"])
        print(f"Best spins   → {spins_out}")

    # Summary table
    print(f"\nMerge summary  ({len(results)} tasks, {total_reads} total reads)")
    print(f"  n_spins     : {ref.get('n_spins', '?'):,}" if isinstance(ref.get('n_spins'), int) else f"  n_spins     : {ref.get('n_spins', '?')}")
    print(f"  method      : {ref.get('method', '?').upper()}")
    print(f"  best_energy : {merged['best_energy']:.6f}")
    print(f"  mean ± std  : {merged['energy_mean']:.4f} ± {merged['energy_std']:.4f}")

    # Plot energy distribution
    if HAS_PLT and len(all_energies) >= 4:
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = min(50, len(all_energies))
        ax.hist(all_energies, bins=bins,
                color="#4c8bf5", edgecolor="white", linewidth=0.5)
        ax.axvline(merged["best_energy"], color="#cc3d2b", lw=2, linestyle="--",
                   label=f"Best: {merged['best_energy']:.4f}")
        n_spins = ref.get("n_spins", "?")
        ax.set_xlabel("Energy", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(
            f"{ref.get('method', '').upper()} merged distribution  "
            f"—  n={n_spins} spins  ({len(results)} tasks)",
            fontsize=11,
        )
        ax.legend()
        fig.tight_layout()
        png_out = f"{out_prefix}_dist.png"
        fig.savefig(png_out, dpi=180)
        print(f"  energy dist → {png_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge SLURM array results from hpc_sqa_launcher.py"
    )
    parser.add_argument("files", nargs="*",
                        help="JSON result files (supports glob on most shells)")
    parser.add_argument("--out", default="merged_best",
                        help="Output prefix (default: merged_best)")
    args = parser.parse_args()

    # Expand any glob patterns not expanded by the shell (e.g. on Windows)
    paths = []
    for pattern in args.files:
        expanded = glob.glob(pattern)
        paths.extend(expanded if expanded else [pattern])

    merge(paths, args.out)


if __name__ == "__main__":
    main()
