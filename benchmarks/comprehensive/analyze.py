"""
analyze.py — load the benchmark CSV and produce publication-quality plots (raw data,
side by side, no percentage-advantage framing). Runs locally (needs pandas + matplotlib).

Usage:
    python analyze.py --input results/benchmark_JOBID.csv --outdir plots/

Per problem class: best_energy, mean_energy, success_prob, TTS, wall time vs n; energy-gap
CDF and best-energy violin at the largest n. Plus two summary heatmaps (success prob, gap).
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOLVER_STYLE = {
    "sqa-std":   {"color": "#2166ac", "marker": "o", "ls": "-",  "label": "SQA std"},
    "sqa-opt":   {"color": "#d6604d", "marker": "s", "ls": "--", "label": "SQA opt"},
    "sqapt-std": {"color": "#4dac26", "marker": "^", "ls": "-",  "label": "SQAPT std"},
    "sqapt-opt": {"color": "#b8860b", "marker": "D", "ls": "--", "label": "SQAPT opt"},
}
PROBLEM_LABELS = {
    "sk": "SK Spin Glass", "ea3d": "3D EA Spin Glass", "w3reg": "3-Regular MAX-CUT",
    "wmaxcut": "Weighted MAX-CUT", "rfim2d": "2D RFIM", "planted": "Planted Partition",
    "numpart": "Number Partitioning",
}
plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_data(path):
    df = pd.read_csv(path)
    df["gs_energy"] = pd.to_numeric(df["gs_energy"], errors="coerce")
    for c in ["best_energy", "mean_energy", "energy_gap", "wall_seconds", "success", "n"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def tts(sp, confidence=0.99):
    sp = np.clip(sp, 1e-9, 1 - 1e-9)
    return np.log(1 - confidence) / np.log(1 - sp)


def _line(ax, ns, ys, errs, style):
    ax.plot(ns, ys, color=style["color"], marker=style["marker"], ls=style["ls"],
            label=style["label"], linewidth=1.8, markersize=6)
    if errs is not None:
        ax.errorbar(ns, ys, yerr=errs, fmt="none", color=style["color"], capsize=3, alpha=0.6)


def plot_metric_vs_n(df, problem, metric, ylabel, outpath, log_y=False):
    sub = df[df["problem"] == problem]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for solver, style in SOLVER_STYLE.items():
        s = sub[sub["solver"] == solver]
        if s.empty:
            continue
        ns = sorted(s["n"].unique())
        ys = [s[s["n"] == n][metric].dropna().mean() for n in ns]
        errs = [s[s["n"] == n][metric].dropna().std() /
                max(np.sqrt(len(s[s["n"] == n][metric].dropna())), 1) for n in ns]
        _line(ax, ns, ys, errs, style)
    ax.set_xlabel("Problem size $n$"); ax.set_ylabel(ylabel)
    ax.set_title(PROBLEM_LABELS.get(problem, problem))
    if log_y:
        ax.set_yscale("log")
    ax.legend(framealpha=0.9); fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_success_prob_vs_n(df, problem, outpath):
    sub = df[(df["problem"] == problem) & (df["success"] >= 0)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for solver, style in SOLVER_STYLE.items():
        s = sub[sub["solver"] == solver]
        if s.empty:
            continue
        ns = sorted(s["n"].unique())
        sps = [s[s["n"] == n]["success"].mean() for n in ns]
        errs = [np.sqrt(max(sp * (1 - sp), 0) / max(len(s[s["n"] == n]), 1))
                for n, sp in zip(ns, sps)]
        _line(ax, ns, sps, errs, style)
    ax.set_xlabel("Problem size $n$"); ax.set_ylabel("Success probability")
    ax.set_title(PROBLEM_LABELS.get(problem, problem)); ax.set_ylim(-0.05, 1.05)
    ax.legend(framealpha=0.9); fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_tts_vs_n(df, problem, outpath):
    sub = df[(df["problem"] == problem) & (df["success"] >= 0)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for solver, style in SOLVER_STYLE.items():
        s = sub[sub["solver"] == solver]
        if s.empty:
            continue
        ns = sorted(s["n"].unique())
        ttss = [tts(s[s["n"] == n]["success"].mean()) for n in ns]
        _line(ax, ns, ttss, None, style)
    ax.set_xlabel("Problem size $n$"); ax.set_ylabel("TTS (reads, 99% confidence)")
    ax.set_title(PROBLEM_LABELS.get(problem, problem)); ax.set_yscale("log")
    ax.legend(framealpha=0.9); fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_gap_cdf(df, problem, outpath):
    sub = df[df["problem"] == problem]
    if sub.empty:
        return
    sub = sub[sub["n"] == sub["n"].max()]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for solver, style in SOLVER_STYLE.items():
        s = sub[sub["solver"] == solver]["energy_gap"].dropna().values
        s = s[s > 0]
        if len(s) == 0:
            continue
        s.sort()
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=style["color"],
                ls=style["ls"], label=style["label"], linewidth=1.8)
    ax.set_xlabel(r"Relative energy gap $|E-E_{\rm ref}|/|E_{\rm ref}|$"); ax.set_ylabel("CDF")
    ax.set_title(f"{PROBLEM_LABELS.get(problem, problem)} (n={int(sub['n'].max())})")
    ax.set_xscale("log"); ax.legend(framealpha=0.9); fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_violin(df, problem, outpath):
    sub = df[df["problem"] == problem]
    if sub.empty:
        return
    largest = sub["n"].max()
    sub = sub[sub["n"] == largest]
    data, labels, colors = [], [], []
    for solver, style in SOLVER_STYLE.items():
        vals = sub[sub["solver"] == solver]["best_energy"].dropna().values
        if len(vals) == 0:
            continue
        data.append(vals); labels.append(style["label"]); colors.append(style["color"])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    parts = ax.violinplot(data, positions=range(len(labels)), showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color); pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("Best energy per read")
    ax.set_title(f"{PROBLEM_LABELS.get(problem, problem)} (n={int(largest)})")
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_heatmaps(df, outdir):
    problems = list(df["problem"].unique())
    solvers = list(SOLVER_STYLE.keys())
    sp_m = np.full((len(problems), len(solvers)), np.nan)
    gap_m = np.full((len(problems), len(solvers)), np.nan)
    for i, prob in enumerate(problems):
        sub = df[df["problem"] == prob]
        sub = sub[sub["n"] == sub["n"].max()]
        for j, solver in enumerate(solvers):
            s = sub[sub["solver"] == solver]
            spv = s[s["success"] >= 0]["success"].values
            gv = s["energy_gap"].dropna().values
            if len(spv):
                sp_m[i, j] = spv.mean()
            if len(gv):
                gap_m[i, j] = gv.mean()
    for matrix, label, fname, cmap, fmt, vmax in [
        (sp_m, "Mean success probability (largest n)", "heatmap_success.pdf", "YlGn", ".2f", 1.0),
        (gap_m, "Mean relative energy gap (largest n)", "heatmap_gap.pdf", "YlOrRd", ".3f", None),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(solvers)))
        ax.set_xticklabels([SOLVER_STYLE[s]["label"] for s in solvers], rotation=15, ha="right")
        ax.set_yticks(range(len(problems)))
        ax.set_yticklabels([PROBLEM_LABELS.get(p, p) for p in problems])
        for i in range(len(problems)):
            for j in range(len(solvers)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:{fmt}}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, label=label); ax.set_title(label)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/benchmark.csv")
    ap.add_argument("--outdir", default="plots/")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.input)
    for prob in df["problem"].unique():
        slug = str(prob)
        plot_metric_vs_n(df, prob, "best_energy", "Best energy",
                         os.path.join(args.outdir, f"{slug}_best_energy_vs_n.pdf"))
        plot_metric_vs_n(df, prob, "mean_energy", "Mean energy",
                         os.path.join(args.outdir, f"{slug}_mean_energy_vs_n.pdf"))
        plot_success_prob_vs_n(df, prob, os.path.join(args.outdir, f"{slug}_success_prob_vs_n.pdf"))
        plot_tts_vs_n(df, prob, os.path.join(args.outdir, f"{slug}_tts_vs_n.pdf"))
        plot_metric_vs_n(df, prob, "wall_seconds", "Wall time (s)",
                         os.path.join(args.outdir, f"{slug}_wall_vs_n.pdf"), log_y=True)
        plot_gap_cdf(df, prob, os.path.join(args.outdir, f"{slug}_gap_cdf.pdf"))
        plot_violin(df, prob, os.path.join(args.outdir, f"{slug}_violin.pdf"))
    plot_heatmaps(df, args.outdir)
    print(f"Plots -> {args.outdir}")
    print(f"Problems: {df['problem'].unique().tolist()}")
    print(f"Solvers:  {df['solver'].unique().tolist()}  |  rows: {len(df)}")


if __name__ == "__main__":
    main()
