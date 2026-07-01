#!/usr/bin/env python3
"""Final combined table + plot: SQA (v7, 30 inst) and SQAPT (v6, 10 inst) opt-vs-std.
Optimal budget-calibrated J_perp schedule, qanneal 0.6.1 (SQA) / 0.6.0 (SQAPT)."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
V7 = HERE / "results" / "v7_3395906.csv"     # SQA
V6 = HERE / "results" / "v6_3395759.csv"     # SQAPT
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)
PROBS = ["sk", "ea3d", "w3reg", "wmaxcut", "numpart"]

def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["n"] = int(r["n"]); r["success_prob"] = float(r["success_prob"])
    return rows

def sp_by(rows, method_prefix):
    """mean success prob keyed by (problem, n, 'std'/'opt') for the given method."""
    t = defaultdict(list)
    for r in rows:
        if r["solver"].startswith(method_prefix + "-"):
            sch = "opt" if r["solver"].endswith("opt") else "std"
            t[(r["problem"], r["n"], sch)].append(r["success_prob"])
    return {k: float(np.mean(v)) for k, v in t.items()}

sqa = sp_by(load(V7), "sqa")
sqapt = sp_by(load(V6), "sqapt")

def sizes(tab, prob):
    return sorted({k[1] for k in tab if k[0] == prob})

# ---- text table ----
print("=" * 78)
print("  FINAL: optimal vs standard J_perp schedule  (Delta% mean success prob)")
print("  SQA = v7 (30 inst, qanneal 0.6.1);  SQAPT = v6 (10 inst, qanneal 0.6.0)")
print("=" * 78)
print(f"{'problem':<9} {'n':>4} | {'SQA std':>8} {'SQA opt':>8} {'d%':>7} | {'PT std':>7} {'PT opt':>7} {'d%':>7}")
print("-" * 78)
def d(a, b): return (b - a) / max(a, 1e-9) * 100
for prob in PROBS:
    for n in sizes(sqa, prob):
        ss, so = sqa.get((prob, n, "std")), sqa.get((prob, n, "opt"))
        line = f"{prob:<9} {n:>4} | {ss:8.3f} {so:8.3f} {d(ss,so):+6.0f}%"
        ps, po = sqapt.get((prob, n, "std")), sqapt.get((prob, n, "opt"))
        if ps is not None:
            line += f" | {ps:7.3f} {po:7.3f} {d(ps,po):+6.0f}%"
        else:
            line += f" | {'(v6 n differ)':>23}"
        print(line)
print("=" * 78)

# ---- plot: SQA delta% vs n per problem (the headline scaling story) ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
for prob in PROBS:
    ns = sizes(sqa, prob)
    ds = [d(sqa[(prob, n, "std")], sqa[(prob, n, "opt")]) for n in ns]
    ax.plot(ns, ds, marker="o", lw=2, ms=8, label=prob)
ax.axhline(0, color="k", lw=1); ax.set_xscale("log", base=2)
ax.set_xlabel("n"); ax.set_ylabel("Δ success prob  (opt−std)/std  [%]")
ax.set_title("SQA (v7): optimal-schedule advantage GROWS with size")
ax.legend(); ax.grid(alpha=0.3, which="both")

# absolute success prob, SQA std vs opt, biggest common sizes
ax = axes[1]
x = np.arange(len(PROBS)); w = 0.35
big = {p: sizes(sqa, p)[-1] for p in PROBS}
std_v = [sqa[(p, big[p], "std")] for p in PROBS]
opt_v = [sqa[(p, big[p], "opt")] for p in PROBS]
ax.bar(x - w/2, std_v, w, label="SQA-std", color="#4C72B0", edgecolor="black")
ax.bar(x + w/2, opt_v, w, label="SQA-opt", color="#DD8452", edgecolor="black", hatch="//")
ax.set_xticks(x); ax.set_xticklabels([f"{p}\nn={big[p]}" for p in PROBS], fontsize=8)
ax.set_ylabel("success probability"); ax.set_ylim(0, 1.05)
ax.set_title("SQA success prob at largest n (opt hatched)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Optimal budget-calibrated J⊥ schedule — SQA (v7, 30 instances, qanneal 0.6.1)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "final_sqa_v7.png", dpi=140)
print("\nwrote", OUT / "final_sqa_v7.png")
