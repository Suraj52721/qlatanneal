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


SUCCESS_THRESHOLD = 0.01


def load_data(path):
    """Load the benchmark CSV and add FAIR cross-solver metrics.

    The raw `energy_gap`/`success` columns (written by benchmark.py) are self-referential for
    problems without a known ground state: each solver's gap is measured against its OWN best
    read, so solver A and solver B are compared against different references -- not usable for
    cross-solver TTS/success-prob comparison. Fix: for every (problem, n, instance), find the
    best energy across ALL FOUR solvers (`best_known`) and measure every solver's gap against
    that single shared reference. For problems WITH a known gs_energy, best_known == gs_energy
    (the true optimum is always <= any solver's best, so min(best_energy) over solvers converges
    to gs_energy whenever at least one solver/read finds it -- and gs_energy is a lower bound by
    construction, so it never OVER-estimates hardness); those columns were already fair.
    """
    df = pd.read_csv(path)
    df["gs_energy"] = pd.to_numeric(df["gs_energy"], errors="coerce")
    for c in ["best_energy", "mean_energy", "energy_gap", "wall_seconds", "success", "n"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Cross-solver reference: best energy found by ANY solver for this (problem, n, instance).
    # When gs_energy is known, take the elementwise min with it too -- gs_energy is a guaranteed
    # true lower bound, so this can only tighten (never loosen) the reference versus the raw
    # empirical minimum, which matters if some instance is so hard that no solver/read ever
    # actually reaches the true optimum.
    df["best_known"] = df.groupby(["problem", "n", "instance"])["best_energy"].transform("min")
    has_gs = df["gs_energy"].notna()
    df.loc[has_gs, "best_known"] = np.minimum(df.loc[has_gs, "best_known"], df.loc[has_gs, "gs_energy"])

    df["gap_fair"] = (df["best_energy"] - df["best_known"]).abs() / (df["best_known"].abs() + 1e-12)
    df["success_fair"] = (df["gap_fair"] <= SUCCESS_THRESHOLD).astype(int)
    return df


def tts(sp, confidence=0.99, n_trials=None):
    """TTS = log(1-confidence)/log(1-sp). sp is floored before the log so a solver that
    genuinely had ZERO successes doesn't blow the axis up to ~1e9 (log(1e-9) territory) -- with
    n_trials observed reads and 0 successes, the true rate can't be pinned below ~1/n_trials
    (rule-of-three-style), so floor at 1/n_trials rather than an arbitrary tiny constant. Falls
    back to 1e-4 (TTS ~ 4.6e4, still finite/plottable) if n_trials isn't supplied."""
    floor = 1.0 / n_trials if n_trials else 1e-4
    sp = np.clip(sp, floor, 1 - 1e-9)
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
    """Fair cross-solver success probability: success_fair is 1 iff a solver's best_energy is
    within SUCCESS_THRESHOLD of best_known (the best ANY solver found for that instance), so all
    four solvers share the same reference -- unlike the raw `success` column (self-referential)."""
    sub = df[df["problem"] == problem]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for solver, style in SOLVER_STYLE.items():
        s = sub[sub["solver"] == solver]
        if s.empty:
            continue
        ns = sorted(s["n"].unique())
        sps = [s[s["n"] == n]["success_fair"].mean() for n in ns]
        errs = [np.sqrt(max(sp * (1 - sp), 0) / max(len(s[s["n"] == n]), 1))
                for n, sp in zip(ns, sps)]
        _line(ax, ns, sps, errs, style)
    ax.set_xlabel("Problem size $n$"); ax.set_ylabel("Success probability (fair, cross-solver)")
    ax.set_title(PROBLEM_LABELS.get(problem, problem)); ax.set_ylim(-0.05, 1.05)
    ax.legend(framealpha=0.9); fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_tts_vs_n(df, problem, outpath):
    sub = df[df["problem"] == problem]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for solver, style in SOLVER_STYLE.items():
        s = sub[sub["solver"] == solver]
        if s.empty:
            continue
        ns = sorted(s["n"].unique())
        ttss = [tts(s[s["n"] == n]["success_fair"].mean(), n_trials=len(s[s["n"] == n]))
                for n in ns]
        _line(ax, ns, ttss, None, style)
    ax.set_xlabel("Problem size $n$"); ax.set_ylabel("TTS (reads, 99% confidence, fair)")
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
        s = sub[sub["solver"] == solver]["gap_fair"].dropna().values
        s = s[s > 0]
        if len(s) == 0:
            continue
        s.sort()
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=style["color"],
                ls=style["ls"], label=style["label"], linewidth=1.8)
    ax.set_xlabel(r"Relative energy gap $|E-E_{\rm best\ known}|/|E_{\rm best\ known}|$ (fair)")
    ax.set_ylabel("CDF")
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
            spv = s["success_fair"].values
            gv = s["gap_fair"].dropna().values
            if len(spv):
                sp_m[i, j] = spv.mean()
            if len(gv):
                gap_m[i, j] = gv.mean()
    for matrix, label, fname, cmap, fmt, vmax in [
        (sp_m, "Mean success probability, fair (largest n)", "heatmap_success.pdf", "YlGn", ".2f", 1.0),
        (gap_m, "Mean relative energy gap, fair (largest n)", "heatmap_gap.pdf", "YlOrRd", ".3f", None),
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


def plot_sqa_vs_sqapt_walltime(df, problem, outpath):
    """SQA-std vs SQAPT-std: mean best_energy vs mean wall_seconds per instance, at the largest n.

    Distinguishes a genuine SQA advantage from a sweep-budget-allocation artifact: SQAPT spreads
    the same total sweep budget across REPLICAS_SQAPT=8 replicas, so its cold (answer-producing)
    replica gets ~1/8 the sweeps SQA's single chain gets, at ~8x the wall time. If SQAPT-std's
    cloud sits at strictly higher wall time AND worse (higher) energy than SQA-std with no
    overlap, SQA is winning on both axes simultaneously -- not explained by budget alone. If
    SQAPT's cloud would reach SQA's energy by extrapolating to even higher wall time, the
    advantage is (at least partly) a budget-allocation effect, not a fundamial SQAPT weakness.
    """
    sub = df[df["problem"] == problem]
    if sub.empty:
        return
    largest_n = sub["n"].max()
    sub = sub[sub["n"] == largest_n]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for solver in ["sqa-std", "sqapt-std"]:
        style = SOLVER_STYLE[solver]
        s = sub[sub["solver"] == solver]
        if s.empty:
            continue
        inst_means = s.groupby("instance").agg(
            mean_energy=("best_energy", "mean"), mean_wall=("wall_seconds", "mean"))
        ax.scatter(inst_means["mean_wall"], inst_means["mean_energy"], color=style["color"],
                  marker=style["marker"], label=style["label"], alpha=0.7, s=40)
        ax.axhline(inst_means["mean_energy"].mean(), color=style["color"], ls=":",
                  alpha=0.5, linewidth=1.2)

    ax.set_xlabel("Mean wall time per read (s)")
    ax.set_ylabel("Mean best energy per read")
    ax.set_title(f"{PROBLEM_LABELS.get(problem, problem)} (n={int(largest_n)})\n"
                "SQA-std vs SQAPT-std: energy vs wall time")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
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
    for prob in ["sk", "ea3d", "w3reg"]:
        if prob in df["problem"].unique():
            plot_sqa_vs_sqapt_walltime(df, prob,
                os.path.join(args.outdir, f"{prob}_walltime_sqa_vs_sqapt.pdf"))
    print(f"Plots -> {args.outdir}")
    print(f"Problems: {df['problem'].unique().tolist()}")
    print(f"Solvers:  {df['solver'].unique().tolist()}  |  rows: {len(df)}")


if __name__ == "__main__":
    main()
