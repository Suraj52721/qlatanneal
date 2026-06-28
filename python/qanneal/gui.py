"""
gui.py — qanneal Interactive Learning Lab
═══════════════════════════════════════════════════════════════════════════════
A three-tab interactive GUI that serves as both a graph-problem solver and a
self-contained training course on quantum annealing.

Tab 1 – Graph Editor   : draw graphs, set weights/biases, pick problem + method, run solve
Tab 2 – Animation Lab  : replay of energy trace + schedule + per-method physics subplot
Tab 3 – Tutorial       : 7 concept cards (SA, β, Γ, Trotter, PT, landscape, encodings)

Run:
    python examples/python/graph_editor_gui.py
"""
from __future__ import annotations

import json
import math
import random
import threading
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, simpledialog
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .solver import (
    solve,
    auto_schedule_sa,
    auto_schedule_sqa,
    auto_schedule_sa_tuned,
    auto_schedule_sqa_tuned,
    auto_ladder_sqa_tuned,
)
from ._qanneal import AnnealSchedule, SQASchedule

try:
    from ._qanneal import SQAParallelTemperingAnnealer
    HAS_SQAPT = True
except Exception:
    HAS_SQAPT = False

try:
    from ._qanneal import CTPIMCAnnealer
    HAS_CTPIMC = True
except Exception:
    HAS_CTPIMC = False


# ─────────────────────────────────────────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":         "#efe6d6",
    "panel":      "#f3ead7",
    "canvas":     "#fdf8ef",
    "text":       "#2b2b2b",
    "muted":      "#6b6b6b",
    "accent":     "#c49b63",
    "accent2":    "#4c8bf5",
    "edge":       "#6f6a5d",
    "edge_cut":   "#cc3d2b",
    "node":       "#ffffff",
    "node_a":     "#cfe8ff",
    "node_b":     "#d6f2d2",
    "node_sel":   "#cc3d2b",
    "grid":       "#e7dcc9",
    "entry_bg":   "#fff7ea",
    "btn_bg":     "#eadfc8",
    "btn_fg":     "#2b2b2b",
    "run_bg":     "#4c8bf5",
    "run_fg":     "#ffffff",
    "section_bg": "#e8dfc9",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Node:
    idx: int
    x: float
    y: float
    bias: float = 0.0
    canvas_items: List[int] = field(default_factory=list)
    label_item: Optional[int] = None


@dataclass
class Edge:
    i: int
    j: int
    weight: float
    canvas_items: List[int] = field(default_factory=list)
    label_item: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Per-method animation explanations
# ─────────────────────────────────────────────────────────────────────────────
_EXPLAIN = {
    "sa": [
        (0.00, "🌡  High temperature (low β): nearly all spin flips accepted.\n"
               "The system explores the energy landscape freely — no local trap is permanent."),
        (0.25, "📉  Cooling: β rises, acceptance probability narrows.\n"
               "P(accept bad move) = exp(−β·ΔE) — larger β → harder to climb uphill."),
        (0.55, "🧲  Low temperature: only downhill moves survive.\n"
               "The system is settling into the best valley found so far."),
        (0.80, "❄️  Nearly frozen: spins locked in place.\n"
               "Final sweep solidifies the solution. High β ≈ zero-temperature limit."),
        (0.95, "✅  Done! Best energy found across all reads is the result."),
    ],
    "sqa": [
        (0.00, "⚛  High Γ (transverse field): strong quantum tunneling.\n"
               "Spins are delocalized across all Trotter slices — like a quantum superposition."),
        (0.20, "🌊  Γ decreasing, β increasing: tunneling weakens, thermal fluctuations cool.\n"
               "J⊥ = ½ ln(1/tanh(βΓ/M)) governs coupling between imaginary-time slices."),
        (0.50, "🔗  Mid-anneal: slices begin to align.\n"
               "Worldline sweeps flip a spin through ALL slices at once — exploiting time correlations."),
        (0.75, "🔒  Low Γ: quantum effects small, classical SA dominates.\n"
               "Cluster sweeps consolidate worldline segments into coherent domains."),
        (0.92, "🎯  Γ→0: all slices collapse to a single classical configuration.\n"
               "The ground state of the quantum Hamiltonian maps to the Ising optimum."),
    ],
    "sqapt": [
        (0.00, "🔥  Replica 0 (hot, high Γ): exploring freely — finds diverse configurations.\n"
               "Coldest replica: refining the best solution found so far."),
        (0.20, "🔄  Swap epoch: adjacent replicas propose to exchange configurations.\n"
               "Accept if ΔS = (βᵢ−βⱼ)(Eⱼ−Eᵢ) ≤ 0, or with prob exp(ΔS)."),
        (0.50, "🌡  Good solutions from hot replicas flow down to cold replicas.\n"
               "Stuck cold replicas escape by receiving hot configurations. Target: 20–50% swap rate."),
        (0.75, "📊  Replica energies converging — ladder compressing toward best solution.\n"
               "PT is most powerful for rugged multi-modal landscapes."),
        (0.92, "✅  Best energy across all replicas and reads is returned."),
    ],
    "ctpimc": [
        (0.00, "⏱  Continuous-time PIMC: worldlines are smooth curves in imaginary time.\n"
               "No Trotter discretisation — worldline kinks can occur at ANY time τ ∈ [0, β]."),
        (0.25, "🌀  Swendsen–Wang cluster updates: worldline segments form clusters.\n"
               "An entire cluster flips at once — avoids critical slowing-down."),
        (0.55, "📉  β rising, Γ falling: kink density decreasing.\n"
               "Fewer kinks = fewer spin flips in imaginary time = approaching classical state."),
        (0.80, "🔒  Low Γ: worldlines nearly straight — approaching classical Ising ground state.\n"
               "No Trotter error makes this ideal when βΓ is large (strong tunneling regime)."),
        (0.95, "✅  Γ→0.001: clean classical projection. Best energy returned."),
    ],
}


def _explain_at(method: str, frac: float) -> str:
    msgs = _EXPLAIN.get(method, _EXPLAIN["sa"])
    text = msgs[0][1]
    for thresh, msg in msgs:
        if frac >= thresh:
            text = msg
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  Main application class
# ─────────────────────────────────────────────────────────────────────────────
class QAnnealLab:
    RADIUS = 18
    GRID = 30

    def __init__(self, title: str = "qanneal Interactive Learning Lab"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=C["bg"])
        self.root.geometry("1280x820")
        self.root.minsize(900, 640)

        # ── state ────────────────────────────────────────────────────────────
        self._nodes: List[Node] = []
        self._edges: Dict[Tuple[int, int], Edge] = {}
        self._selected: List[int] = []
        self._grid_items: List[int] = []
        self._hover_item: Optional[int] = None
        self._drag_node_idx: Optional[int] = None
        self._drag_start: Tuple[float, float] = (0.0, 0.0)
        self._drag_moved: bool = False
        self._solver_thread: Optional[threading.Thread] = None
        self._anim_data: Optional[dict] = None
        self._anim_frame: int = 0
        self._anim_total: int = 0
        self._anim_playing: bool = False
        self._anim_after_id: Optional[str] = None
        self._show_edge_labels: bool = True
        self._tut_fig: Optional[Figure] = None
        self._tut_canvas_widget = None
        self._anim_fig: Optional[Figure] = None
        self._anim_mpl_canvas = None

        # ── notebook ─────────────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=C["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=C["panel"], foreground=C["text"],
                        padding=[12, 6], font=("Helvetica", 11, "bold"))
        style.map("TNotebook.Tab", background=[("selected", C["accent"]), ("active", C["btn_bg"])],
                  foreground=[("selected", "white")])

        self._nb = ttk.Notebook(self.root)
        self._nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._build_editor_tab()
        self._build_animate_tab()
        self._build_tutorial_tab()

        # ── status bar ───────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Nodes: 0 | Edges: 0  —  Ready")
        tk.Label(self.root, textvariable=self._status_var, anchor="w",
                 bg=C["panel"], fg=C["muted"], relief=tk.SUNKEN,
                 font=("Helvetica", 9)).pack(fill=tk.X, side=tk.BOTTOM)

        self._draw_grid()
        self._on_method_change()

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 1 — GRAPH EDITOR
    # ═════════════════════════════════════════════════════════════════════════
    def _build_editor_tab(self):
        tab = tk.Frame(self._nb, bg=C["bg"])
        self._nb.add(tab, text="  Graph Editor  ")

        # Canvas
        self._canvas = tk.Canvas(tab, bg=C["canvas"], highlightthickness=0)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Sidebar
        sb_outer = tk.Frame(tab, bg=C["panel"], width=310)
        sb_outer.pack(side=tk.RIGHT, fill=tk.Y)
        sb_outer.pack_propagate(False)

        sb_cv = tk.Canvas(sb_outer, bg=C["panel"], highlightthickness=0, width=306)
        sb_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_scroll = ttk.Scrollbar(sb_outer, orient="vertical", command=sb_cv.yview)
        sb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        sb_cv.configure(yscrollcommand=sb_scroll.set)

        self._ctrl = tk.Frame(sb_cv, bg=C["panel"])
        sb_cv.create_window((0, 0), window=self._ctrl, anchor="nw", width=306)
        self._ctrl.bind("<Configure>", lambda e: sb_cv.configure(
            scrollregion=sb_cv.bbox("all")))
        sb_cv.bind_all("<MouseWheel>",
                       lambda e: sb_cv.yview_scroll(int(-1*(e.delta/120)), "units"))

        p = self._ctrl  # shorthand

        # ── Section: Graph ───────────────────────────────────────────────────
        self._section(p, "GRAPH")

        self._preset_var = tk.StringVar(value="MaxCut Ring (6)")
        presets = [
            "MaxCut Ring (6)", "MaxCut Complete (6)", "MaxCut Bipartite (4,4)",
            "Random QUBO (8)", "Spin Glass (8)",
            "Petersen Graph", "Frustrated Triangles (9)",
            "Chimera Unit Cell (8)", "Number Partition (8)",
        ]
        self._mk_label(p, "Preset graph")
        self._preset_menu = tk.OptionMenu(p, self._preset_var, *presets)
        self._style_menu(self._preset_menu)
        self._preset_menu.pack(fill=tk.X, padx=6)

        bf = tk.Frame(p, bg=C["panel"])
        bf.pack(fill=tk.X, padx=6, pady=3)
        self._mk_btn(bf, "Load Preset", self._load_preset, C["accent"]).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self._mk_btn(bf, "Clear", self._clear, C["btn_bg"]).pack(
            side=tk.LEFT, expand=True, fill=tk.X)

        bf2 = tk.Frame(p, bg=C["panel"])
        bf2.pack(fill=tk.X, padx=6, pady=(0, 4))
        self._mk_btn(bf2, "Save JSON", self._save_graph, C["btn_bg"]).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self._mk_btn(bf2, "Load JSON", self._load_graph, C["btn_bg"]).pack(
            side=tk.LEFT, expand=True, fill=tk.X)

        # ── Section: Problem ─────────────────────────────────────────────────
        self._section(p, "PROBLEM TYPE")

        self._problem_var = tk.StringVar(value="MaxCut")
        problems = ["MaxCut", "QUBO", "MaxIndependentSet",
                    "MaxClique", "MinVertexCover", "NumberPartition"]
        self._mk_label(p, "Problem")
        self._problem_menu = tk.OptionMenu(p, self._problem_var, *problems)
        self._style_menu(self._problem_menu)
        self._problem_menu.pack(fill=tk.X, padx=6)

        self._penalty_var = tk.DoubleVar(value=2.0)
        self._mk_label(p, "Penalty A  (MIS / VC / Clique)")
        tk.Scale(p, variable=self._penalty_var, from_=0.5, to=10.0,
                 resolution=0.5, orient=tk.HORIZONTAL,
                 bg=C["panel"], fg=C["text"], troughcolor=C["entry_bg"],
                 highlightthickness=0).pack(fill=tk.X, padx=6)

        # ── Section: Method ──────────────────────────────────────────────────
        self._section(p, "SOLVER METHOD")

        self._method_var = tk.StringVar(value="sqapt")
        mf = tk.Frame(p, bg=C["panel"])
        mf.pack(fill=tk.X, padx=6, pady=2)
        for m, lbl in [("sa", "SA"), ("sqa", "SQA"), ("sqapt", "SQAPT"), ("ctpimc", "CT-PIMC")]:
            rb = tk.Radiobutton(mf, text=lbl, variable=self._method_var, value=m,
                                bg=C["panel"], fg=C["text"], selectcolor=C["panel"],
                                activebackground=C["panel"],
                                command=self._on_method_change)
            rb.pack(side=tk.LEFT, padx=4)

        self._reads_var = tk.IntVar(value=20)
        self._mk_label(p, "Reads (independent runs)")
        tk.Scale(p, variable=self._reads_var, from_=1, to=100,
                 orient=tk.HORIZONTAL, bg=C["panel"], fg=C["text"],
                 troughcolor=C["entry_bg"], highlightthickness=0).pack(fill=tk.X, padx=6)

        # ── Section: Schedule ────────────────────────────────────────────────
        self._section(p, "SCHEDULE")

        self._autotune_var = tk.BooleanVar(value=True)
        tk.Checkbutton(p, text="Auto-tune schedule (recommended)",
                       variable=self._autotune_var,
                       bg=C["panel"], fg=C["text"], selectcolor=C["panel"],
                       activebackground=C["panel"],
                       command=self._on_autotune_change).pack(anchor="w", padx=6)

        self._mode_var = tk.StringVar(value="balanced")
        mf2 = tk.Frame(p, bg=C["panel"])
        mf2.pack(fill=tk.X, padx=6, pady=2)
        for v, lbl in [("fast", "Fast"), ("balanced", "Balanced"), ("accurate", "Accurate")]:
            tk.Radiobutton(mf2, text=lbl, variable=self._mode_var, value=v,
                           bg=C["panel"], fg=C["text"], selectcolor=C["panel"],
                           activebackground=C["panel"]).pack(side=tk.LEFT, padx=3)

        # Manual schedule params (shown when auto-tune OFF)
        self._manual_frame = tk.Frame(p, bg=C["panel"])
        self._manual_frame.pack(fill=tk.X, padx=6)
        for attr, lbl, default in [
            ("_steps_var",    "Steps",        60),
            ("_sweeps_var",   "Sweeps/step",  40),
            ("_beta_s_var",   "β start",      0.1),
            ("_beta_e_var",   "β end",        5.0),
            ("_gamma_s_var",  "Γ start",      5.0),
            ("_gamma_e_var",  "Γ end",        0.01),
        ]:
            cls = tk.IntVar if isinstance(default, int) else tk.DoubleVar
            setattr(self, attr, cls(value=default))
            self._mk_label(self._manual_frame, lbl)
            tk.Entry(self._manual_frame, textvariable=getattr(self, attr),
                     bg=C["entry_bg"], fg=C["text"],
                     insertbackground=C["text"]).pack(fill=tk.X)

        # SQA-specific params
        self._sqa_frame = tk.Frame(p, bg=C["panel"])
        self._sqa_frame.pack(fill=tk.X, padx=6)
        for attr, lbl, default in [
            ("_slices_var",   "Trotter slices",  20),
            ("_worldline_var","Worldline sweeps",  4),
            ("_replicas_var", "Replicas",          4),
            ("_pt_steps_var", "PT epochs (SQAPT)", 60),
        ]:
            setattr(self, attr, tk.IntVar(value=default))
            self._mk_label(self._sqa_frame, lbl)
            tk.Scale(self._sqa_frame, variable=getattr(self, attr),
                     from_=1, to=128, orient=tk.HORIZONTAL,
                     bg=C["panel"], fg=C["text"],
                     troughcolor=C["entry_bg"],
                     highlightthickness=0).pack(fill=tk.X)

        # ── Section: Options ─────────────────────────────────────────────────
        self._section(p, "OPTIONS")

        self._verify_var = tk.BooleanVar(value=True)
        self._snap_var   = tk.BooleanVar(value=True)
        self._grid_var   = tk.BooleanVar(value=True)
        self._ew_var     = tk.BooleanVar(value=True)
        self._cut_var    = tk.BooleanVar(value=True)
        for var, lbl, cmd in [
            (self._verify_var, "Brute-force verify (n ≤ 18)", None),
            (self._snap_var,   "Snap to grid",                None),
            (self._grid_var,   "Show grid",                   self._draw_grid),
            (self._ew_var,     "Show edge weight labels",      self._toggle_edge_labels),
            (self._cut_var,    "Highlight cut / solution",    None),
        ]:
            tk.Checkbutton(p, text=lbl, variable=var,
                           bg=C["panel"], fg=C["text"], selectcolor=C["panel"],
                           activebackground=C["panel"],
                           command=cmd).pack(anchor="w", padx=6)

        # ── RUN button ───────────────────────────────────────────────────────
        self._run_btn = tk.Button(p, text="▶  RUN SOLVE",
                                  command=self._run_solve,
                                  bg=C["run_bg"], fg=C["run_fg"],
                                  font=("Helvetica", 13, "bold"),
                                  relief=tk.FLAT, cursor="hand2",
                                  activebackground="#3a7ae0",
                                  activeforeground="white")
        self._run_btn.pack(fill=tk.X, padx=6, pady=8, ipady=6)

        # ── Log ──────────────────────────────────────────────────────────────
        self._section(p, "LOG")
        self._log_text = tk.Text(p, height=10, width=36, bg="#f7f1e3",
                                 fg=C["text"], insertbackground=C["text"],
                                 font=("Courier", 9), wrap=tk.WORD)
        self._log_text.pack(padx=6, pady=4)
        self._log("Left-click empty area  → add node")
        self._log("Left-click node        → select")
        self._log("Drag node              → move it")
        self._log("Right-click            → context menu")
        self._log("Keys: b=bias  w=weight  d/Del=delete")

        # ── Canvas bindings ──────────────────────────────────────────────────
        self._canvas.bind("<Button-1>",        self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Button-2>",        self._on_right_click)
        self._canvas.bind("<Button-3>",        self._on_right_click)
        self._canvas.bind("<Motion>",          self._on_motion)
        self._canvas.bind("<Configure>",       lambda e: self._draw_grid())
        self.root.bind("b", self._set_bias)
        self.root.bind("w", self._set_weight)
        self.root.bind("d", self._delete_selected)
        self.root.bind("<Delete>", self._delete_selected)

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 2 — ANIMATION LAB
    # ═════════════════════════════════════════════════════════════════════════
    def _build_animate_tab(self):
        self._anim_tab = tk.Frame(self._nb, bg=C["bg"])
        self._nb.add(self._anim_tab, text="  Animation Lab  ")

        if not HAS_MPL:
            tk.Label(self._anim_tab, text="Install matplotlib to use Animation Lab\npip install matplotlib",
                     bg=C["bg"], fg=C["text"], font=("Helvetica", 14)).pack(expand=True)
            return

        # matplotlib figure: 3 subplots
        self._anim_fig = Figure(figsize=(11, 7), facecolor=C["canvas"])
        gs = gridspec.GridSpec(3, 1, figure=self._anim_fig,
                               hspace=0.45,
                               top=0.95, bottom=0.06, left=0.08, right=0.97)
        self._ax_sched  = self._anim_fig.add_subplot(gs[0])
        self._ax_energy = self._anim_fig.add_subplot(gs[1])
        self._ax_extra  = self._anim_fig.add_subplot(gs[2])

        for ax in (self._ax_sched, self._ax_energy, self._ax_extra):
            ax.set_facecolor("#fefaf3")
            for spine in ax.spines.values():
                spine.set_color("#ccc0a8")

        self._anim_mpl_canvas = FigureCanvasTkAgg(self._anim_fig, master=self._anim_tab)
        self._anim_mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Explanation text
        self._explain_var = tk.StringVar(
            value="Run a solve from the Graph Editor tab to start the animation.")
        explain_lbl = tk.Label(self._anim_tab, textvariable=self._explain_var,
                               bg="#fffbf0", fg=C["text"], font=("Helvetica", 10),
                               wraplength=900, justify=tk.LEFT, anchor="w",
                               padx=10, pady=6, relief=tk.GROOVE)
        explain_lbl.pack(fill=tk.X, padx=6, pady=(0, 4))

        # Playback controls
        ctrl = tk.Frame(self._anim_tab, bg=C["panel"])
        ctrl.pack(fill=tk.X, padx=6, pady=4)

        self._mk_btn(ctrl, "▶ Play",  self._anim_play,  C["run_bg"],  C["run_fg"]).pack(side=tk.LEFT, padx=3)
        self._mk_btn(ctrl, "⏸ Pause", self._anim_pause, C["btn_bg"]).pack(side=tk.LEFT, padx=3)
        self._mk_btn(ctrl, "⏮ Reset", self._anim_reset, C["btn_bg"]).pack(side=tk.LEFT, padx=3)

        tk.Label(ctrl, text="  Speed:", bg=C["panel"], fg=C["text"]).pack(side=tk.LEFT)
        self._speed_var = tk.IntVar(value=40)
        tk.Scale(ctrl, variable=self._speed_var, from_=5, to=200, resolution=5,
                 orient=tk.HORIZONTAL, length=160,
                 bg=C["panel"], fg=C["text"], troughcolor=C["entry_bg"],
                 highlightthickness=0,
                 showvalue=False).pack(side=tk.LEFT, padx=6)
        tk.Label(ctrl, text="(ms/frame)", bg=C["panel"], fg=C["muted"],
                 font=("Helvetica", 9)).pack(side=tk.LEFT)

        self._frame_var = tk.StringVar(value="Frame 0 / 0")
        tk.Label(ctrl, textvariable=self._frame_var, bg=C["panel"],
                 fg=C["muted"], font=("Courier", 9)).pack(side=tk.RIGHT, padx=8)

        # Pre-draw placeholder axes
        self._anim_reset_plots()

    def _anim_reset_plots(self):
        if not HAS_MPL or self._anim_fig is None:
            return
        for ax, title in [
            (self._ax_sched,  "Schedule — β and Γ vs step"),
            (self._ax_energy, "Energy trace"),
            (self._ax_extra,  "Physics subplot"),
        ]:
            ax.clear()
            ax.set_title(title, fontsize=10, color=C["muted"])
            ax.set_facecolor("#fefaf3")
        self._anim_mpl_canvas.draw_idle()

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 3 — TUTORIAL
    # ═════════════════════════════════════════════════════════════════════════
    def _build_tutorial_tab(self):
        tab = tk.Frame(self._nb, bg=C["bg"])
        self._nb.add(tab, text="  Tutorial  ")

        if not HAS_MPL:
            tk.Label(tab, text="Install matplotlib to view Tutorial\npip install matplotlib",
                     bg=C["bg"], fg=C["text"], font=("Helvetica", 14)).pack(expand=True)
            return

        topics = [
            "1. What is Annealing?",
            "2. Inverse Temperature β",
            "3. Transverse Field Γ",
            "4. Trotter Decomposition",
            "5. Parallel Tempering",
            "6. Energy Landscapes",
            "7. Problem Encodings",
        ]

        left = tk.Frame(tab, bg=C["panel"], width=200)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0), pady=4)
        left.pack_propagate(False)
        tk.Label(left, text="Topics", bg=C["panel"], fg=C["text"],
                 font=("Helvetica", 12, "bold")).pack(pady=8)

        self._topic_lb = tk.Listbox(left, bg=C["entry_bg"], fg=C["text"],
                                    selectbackground=C["accent"],
                                    selectforeground="white",
                                    font=("Helvetica", 10),
                                    relief=tk.FLAT, bd=0,
                                    activestyle="none",
                                    height=len(topics))
        for t in topics:
            self._topic_lb.insert(tk.END, "  " + t)
        self._topic_lb.pack(fill=tk.X, padx=6, pady=4)
        self._topic_lb.bind("<<ListboxSelect>>", self._on_topic_select)

        self._tut_right = tk.Frame(tab, bg=C["bg"])
        self._tut_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Initial card
        self._tut_fig = Figure(figsize=(9, 5.5), facecolor=C["canvas"])
        self._tut_mpl_canvas = FigureCanvasTkAgg(self._tut_fig, master=self._tut_right)
        self._tut_mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._tut_text = tk.Text(self._tut_right, height=5, bg="#fffbf0", fg=C["text"],
                                 font=("Helvetica", 10), wrap=tk.WORD,
                                 relief=tk.GROOVE, padx=10, pady=6)
        self._tut_text.pack(fill=tk.X)

        self._topic_lb.selection_set(0)
        self._show_tutorial(0)

    def _on_topic_select(self, _event=None):
        sel = self._topic_lb.curselection()
        if sel:
            self._show_tutorial(sel[0])

    def _show_tutorial(self, idx: int):
        renderers = [
            self._tut_annealing,
            self._tut_beta,
            self._tut_gamma,
            self._tut_trotter,
            self._tut_pt,
            self._tut_landscape,
            self._tut_encodings,
        ]
        self._tut_fig.clear()
        renderers[idx]()
        self._tut_mpl_canvas.draw_idle()

    # ── Tutorial card renderers ───────────────────────────────────────────────
    def _tut_annealing(self):
        ax = self._tut_fig.add_subplot(111)
        ax.set_facecolor("#fefaf3")
        t = np.linspace(0, 10, 400)
        rng = np.random.default_rng(7)
        # High-T random walk
        hw = np.cumsum(rng.normal(0, 1.2, 400)) - 5
        hw -= hw.min() - 0.5
        # Low-T greedy descent
        lw = 8 * np.exp(-0.4 * t) + 0.5 + 0.15 * np.sin(3 * t) * np.exp(-0.3 * t)
        ax.plot(t, hw, color="#c84b31", alpha=0.7, lw=1.5, label="High T — explores freely")
        ax.plot(t, lw, color="#4c8bf5", lw=2.5, label="Cooling — converges to minimum")
        ax.axhline(0.5, color="#2ca02c", lw=1.5, ls="--", label="Global minimum")
        ax.set_xlabel("Annealing time →", fontsize=10)
        ax.set_ylabel("Energy", fontsize=10)
        ax.set_title("Simulated Annealing: hot exploration → cold convergence", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(-1, max(hw.max(), lw.max()) + 1)
        for sp in ax.spines.values(): sp.set_color("#ccc0a8")
        self._tut_write(
            "SIMULATED ANNEALING (SA)\n"
            "═══════════════════════\n"
            "Inspired by the metallurgical process of heating and slowly cooling metal.\n\n"
            "  • At HIGH temperature: the system accepts bad moves (energy increases) freely.\n"
            "    This lets it escape local minima — explore the full landscape.\n\n"
            "  • As temperature DROPS: acceptance probability P = exp(−β·ΔE) narrows.\n"
            "    β = 1/kT is the inverse temperature. Large β → cold → only downhill moves.\n\n"
            "  • Key insight: slow cooling (many steps) → better final solution."
        )

    def _tut_beta(self):
        ax = self._tut_fig.add_subplot(111)
        ax.set_facecolor("#fefaf3")
        dE = np.linspace(0, 5, 300)
        colors = ["#c84b31", "#f5a623", "#4c8bf5", "#2ca02c"]
        betas  = [0.1, 0.5, 2.0, 5.0]
        for beta, col in zip(betas, colors):
            ax.plot(dE, np.exp(-beta * dE), color=col, lw=2.2,
                    label=f"β = {beta}  (T = {1/beta:.1f})")
        ax.axhline(0.5, color="#999", ls=":", lw=1)
        ax.set_xlabel("ΔE  (energy increase of proposed move)", fontsize=10)
        ax.set_ylabel("P(accept bad move) = exp(−β·ΔE)", fontsize=10)
        ax.set_title("Boltzmann acceptance probability vs ΔE for different β", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        for sp in ax.spines.values(): sp.set_color("#ccc0a8")
        self._tut_write(
            "INVERSE TEMPERATURE β\n"
            "═════════════════════\n"
            "β = 1 / (k_B · T)   — larger β means COLDER system.\n\n"
            "  • Metropolis criterion: accept a spin flip with ΔE > 0\n"
            "    with probability  P = exp(−β·ΔE)\n\n"
            "  • β = 0.1 (hot):  P ≈ 1 for small ΔE — nearly every move accepted\n"
            "  • β = 5.0 (cold): P ≈ 0 even for ΔE = 1 — only perfect moves accepted\n\n"
            "  • The annealing SCHEDULE ramps β from small (hot) to large (cold).\n"
            "  • Auto-tune computes β_start/β_end from the 75th-percentile |ΔE| of random probes."
        )

    def _tut_gamma(self):
        fig = self._tut_fig
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for ax in (ax1, ax2):
            ax.set_facecolor("#fefaf3")
            for sp in ax.spines.values(): sp.set_color("#ccc0a8")

        # Left: energy barrier with classical and quantum paths
        x = np.linspace(-2, 2, 300)
        E = (x**2 - 1)**2
        ax1.plot(x, E, "k-", lw=2.5)
        ax1.fill_between(x, E, alpha=0.08, color="gray")
        ax1.annotate("", xy=(0.95, 0.05), xytext=(-0.95, 0.05),
                     arrowprops=dict(arrowstyle="->>", color="#c84b31", lw=2.5))
        ax1.text(0, 0.15, "thermal hop\n(SA)", ha="center", color="#c84b31", fontsize=8)
        ax1.annotate("", xy=(0.9, -0.02), xytext=(-0.9, -0.02),
                     arrowprops=dict(arrowstyle="->>", color="#4c8bf5", lw=2.5,
                                     connectionstyle="arc3,rad=-0.4"))
        ax1.text(0, -0.25, "quantum tunnelling\n(SQA, high Γ)", ha="center",
                 color="#4c8bf5", fontsize=8)
        ax1.set_title("Thermal hop vs quantum tunnelling", fontsize=9, fontweight="bold")
        ax1.set_xlabel("Spin configuration space")
        ax1.set_ylabel("Energy")
        ax1.set_ylim(-0.5, 1.2)

        # Right: Γ(t) schedule
        t = np.linspace(0, 1, 200)
        gamma = 5.0 * np.exp(-3 * t)
        beta  = 0.3 + 4.7 * t
        ax2.plot(t, gamma, color="#f5a623", lw=2.5, label="Γ  (tunneling)")
        ax2.plot(t, beta,  color="#4c8bf5", lw=2.5, label="β  (cooling)")
        ax2.set_xlabel("Annealing fraction")
        ax2.set_ylabel("Parameter value")
        ax2.set_title("Typical SQA schedule", fontsize=9, fontweight="bold")
        ax2.legend(fontsize=9)
        self._tut_write(
            "TRANSVERSE FIELD Γ  (quantum tunnelling strength)\n"
            "══════════════════════════════════════════════════\n"
            "Adding H_x = −Γ Σᵢ σᵢˣ to the Ising Hamiltonian allows quantum tunnelling.\n\n"
            "  • HIGH Γ: spins quantum-mechanically tunnel through energy barriers.\n"
            "    Effect is similar to high temperature, but mechanism is different.\n\n"
            "  • LOW Γ → 0: quantum term vanishes, system collapses to classical state.\n\n"
            "  • Trotter coupling: J⊥(β,Γ,M) = ½ ln(1 / tanh(βΓ/M))\n"
            "    Large J⊥ → slices strongly correlated → quantum delocalization."
        )

    def _tut_trotter(self):
        ax = self._tut_fig.add_subplot(111)
        ax.set_facecolor("#fefaf3")
        M, n = 6, 8
        rng = np.random.default_rng(42)
        grid = rng.choice([-1, 1], size=(M, n))
        # Make it mostly +1 to look "ordered"
        grid = np.abs(grid) * np.sign(np.random.default_rng(99).normal(0.3, 1, (M, n)))
        grid = np.where(grid >= 0, 1, -1)
        cmap = plt.cm.RdBu
        ax.imshow(grid, cmap=cmap, aspect="auto", vmin=-1, vmax=1,
                  extent=[-0.5, n - 0.5, -0.5, M - 0.5])
        ax.set_xlabel("Spin index  i", fontsize=10)
        ax.set_ylabel("Trotter slice  t", fontsize=10)
        ax.set_title("SQA State: M Trotter slices × n spins  (blue=−1, red=+1)", fontsize=10, fontweight="bold")
        ax.set_xticks(range(n)); ax.set_xticklabels([f"s{i}" for i in range(n)])
        ax.set_yticks(range(M)); ax.set_yticklabels([f"τ{t}" for t in range(M)])
        # Draw J⊥ springs between slices
        for t in range(M - 1):
            ax.axhline(t + 0.5, color="#f5a623", lw=1, alpha=0.6, ls="--")
        for sp in ax.spines.values(): sp.set_color("#ccc0a8")
        ax.text(n - 0.3, M / 2, "J⊥\n(Trotter\ncoupling)", ha="right",
                va="center", color="#f5a623", fontsize=8,
                bbox=dict(boxstyle="round", fc="#fffbf0", ec="#f5a623"))
        self._tut_write(
            "SUZUKI–TROTTER DECOMPOSITION\n"
            "════════════════════════════\n"
            "Maps the d-dimensional quantum system to a (d+1)-dimensional classical system.\n\n"
            "  • M extra 'imaginary-time' slices are added: each slice = a classical spin config.\n"
            "  • Slices are coupled by J⊥ = ½ ln(1/tanh(βΓ/M))\n"
            "    → simulates quantum tunnelling via classical interactions.\n\n"
            "  • Sweep types in SQA:\n"
            "    1. Slice sweeps:    single-spin Metropolis within each slice\n"
            "    2. Worldline sweeps: flip spin i through ALL slices simultaneously\n"
            "    3. Cluster sweeps:  Swendsen–Wang along the time direction\n\n"
            "  • More slices M → smaller Trotter error, more memory and compute."
        )

    def _tut_pt(self):
        fig = self._tut_fig
        ax = fig.add_subplot(111)
        ax.set_facecolor("#fefaf3")
        R = 5
        betas  = np.linspace(0.2, 5.0, R)
        gammas = np.geomspace(4.0, 0.1, R)
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, R))
        t = np.linspace(0, 10, 200)
        rng = np.random.default_rng(3)
        for r in range(R):
            base = (5 - r) * np.exp(-betas[r] * 0.3 * t) + 0.5
            noise = rng.normal(0, 0.5 / (r + 1), 200)
            energy = base + noise
            ax.plot(t, energy, color=colors[r], lw=1.8,
                    label=f"Replica {r}  β={betas[r]:.1f}, Γ={gammas[r]:.2f}")
        # Swap markers
        for sw in [2.0, 4.5, 7.0, 9.0]:
            ax.axvline(sw, color="#999", ls=":", lw=1.2, alpha=0.7)
            ax.text(sw, 4.2, "swap", ha="center", fontsize=7, color="#999")
        ax.set_xlabel("PT epoch →", fontsize=10)
        ax.set_ylabel("Replica energy", fontsize=10)
        ax.set_title("Parallel Tempering: replica ladder + swap events", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        for sp in ax.spines.values(): sp.set_color("#ccc0a8")
        self._tut_write(
            "PARALLEL TEMPERING  (Replica Exchange)\n"
            "═══════════════════════════════════════\n"
            "Run R replicas simultaneously at different (β, Γ) ladder points.\n\n"
            "  • Hot replicas (high Γ, low β): explore freely — find diverse candidates.\n"
            "  • Cold replicas (low Γ, high β): refine — converge to low-energy states.\n\n"
            "  • Swap criterion (after every swap_interval steps):\n"
            "    ΔS = (βᵢ − βⱼ)(Eⱼ − Eᵢ)  →  accept if ΔS ≤ 0 or rand < exp(ΔS)\n\n"
            "  • Target swap acceptance: 20–50% means ladder spacing is appropriate.\n"
            "  • Dramatically improves performance on rugged, multi-modal landscapes."
        )

    def _tut_landscape(self):
        fig = self._tut_fig
        ax = fig.add_subplot(111)
        ax.set_facecolor("#fefaf3")
        x = np.linspace(0, 10, 1000)
        E = (np.sin(1.5 * x) * np.exp(-0.05 * x)
             + 0.4 * np.sin(4 * x) * np.exp(-0.1 * x)
             + 0.2 * np.sin(8 * x)
             - 0.02 * x + 1.5)
        ax.plot(x, E, "k-", lw=2, alpha=0.7)
        ax.fill_between(x, E, E.min() - 0.2, alpha=0.08, color="gray")
        # SA path (gets stuck)
        sa_x = [0.5, 1.2, 2.1, 2.5, 3.0]
        sa_E = [np.interp(xi, x, E) for xi in sa_x]
        ax.plot(sa_x, sa_E, "o-", color="#c84b31", lw=2, ms=7, label="SA (stuck in local min)")
        ax.annotate("SA stuck!", xy=(sa_x[-1], sa_E[-1] + 0.05),
                    ha="center", color="#c84b31", fontsize=8)
        # SQA tunnels
        opt_x = x[np.argmin(E)]
        opt_E = E.min()
        ax.annotate("", xy=(opt_x, opt_E + 0.1), xytext=(sa_x[-1], sa_E[-1]),
                    arrowprops=dict(arrowstyle="->>", color="#4c8bf5", lw=2.5,
                                    connectionstyle="arc3,rad=-0.5"))
        ax.text((sa_x[-1] + opt_x) / 2, opt_E + 0.6, "SQA tunnels!", ha="center",
                color="#4c8bf5", fontsize=9)
        ax.plot(opt_x, opt_E, "*", color="#2ca02c", ms=16, zorder=5, label="Global minimum")
        ax.set_xlabel("Configuration space  (all 2ⁿ spin states)", fontsize=10)
        ax.set_ylabel("Energy", fontsize=10)
        ax.set_title("Energy landscape: SA vs SQA / SQAPT", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        for sp in ax.spines.values(): sp.set_color("#ccc0a8")
        self._tut_write(
            "ENERGY LANDSCAPE\n"
            "════════════════\n"
            "The configuration space has 2ⁿ points — visualised as a 1D projection.\n\n"
            "  • Local minima: valleys SA gets stuck in after sufficient cooling.\n"
            "  • Barriers: energy humps separating minima — SA must climb over them.\n\n"
            "  • SA advantage: fast; works well when landscape is smooth.\n"
            "  • SQA advantage: quantum tunnelling bypasses high barriers.\n"
            "  • SQAPT advantage: hot replicas find many valleys; swaps spread good solutions.\n\n"
            "  • Rule of thumb: if SA hit rate < 30%, try SQA or SQAPT."
        )

    def _tut_encodings(self):
        fig = self._tut_fig
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.set_facecolor("#fefaf3")
        # Draw a small MaxCut graph
        pts = [(0.15, 0.7), (0.35, 0.9), (0.55, 0.7), (0.35, 0.5)]
        cols = ["#cfe8ff", "#d6f2d2", "#d6f2d2", "#cfe8ff"]  # A=blue, B=green
        edges_e = [(0, 1, True), (1, 2, False), (2, 3, True), (3, 0, False), (0, 2, True)]
        for i1, i2, is_cut in edges_e:
            x1, y1 = pts[i1]; x2, y2 = pts[i2]
            col = "#cc3d2b" if is_cut else "#6f6a5d"
            lw  = 2.5       if is_cut else 1.5
            ax.plot([x1, x2], [y1, y2], color=col, lw=lw, transform=ax.transAxes)
        for i, (px, py) in enumerate(pts):
            ax.add_patch(mpatches.Circle((px, py), 0.05, color=cols[i],
                                         ec="#333", lw=1.5, transform=ax.transAxes,
                                         zorder=3, clip_on=False))
            ax.text(px, py, str(i), ha="center", va="center", fontsize=9,
                    transform=ax.transAxes, zorder=4)
        ax.text(0.35, 0.30, "MaxCut: minimise\n"
                "E(x) = −Σ wᵢⱼ(xᵢ + xⱼ − 2xᵢxⱼ)\n"
                "Red edges = cut (xᵢ ≠ xⱼ)", ha="center", va="top",
                transform=ax.transAxes, fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", fc="#fffbf0", ec="#c49b63"))
        ax.text(0.78, 0.80, "Number Partition:\n"
                "min (Σᵢ aᵢ sᵢ)²\n"
                "h=0, Jᵢⱼ = 2aᵢaⱼ\n"
                "spins ∈ {±1}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", fc="#fffbf0", ec="#c49b63"))
        self._tut_write(
            "PROBLEM ENCODINGS\n"
            "═════════════════\n"
            "Any combinatorial optimisation can be encoded as QUBO / Ising:\n\n"
            "  MaxCut:          J[i,j] = −w[i,j]  (negative = want different spins)\n"
            "  Number Partition: J[i,j] = 2·aᵢ·aⱼ  (minimise squared sum)\n"
            "  Max Ind. Set:    diagonal −1 (select node), off-diag +A (edge penalty)\n"
            "  Min Vertex Cover: diagonal +1 (cost), off-diag −A + A·xᵢxⱼ (edge must be covered)\n\n"
            "  qanneal accepts: DenseIsing, QUBO, np.ndarray, dict, networkx.Graph, dimod.BQM"
        )

    def _tut_write(self, text: str):
        self._tut_text.config(state=tk.NORMAL)
        self._tut_text.delete("1.0", tk.END)
        self._tut_text.insert(tk.END, text)
        self._tut_text.config(state=tk.DISABLED)

    # ═════════════════════════════════════════════════════════════════════════
    #  GRAPH EDITOR — interaction
    # ═════════════════════════════════════════════════════════════════════════
    def _snap(self, x: float, y: float) -> Tuple[float, float]:
        if not self._snap_var.get():
            return x, y
        g = self.GRID
        return round(x / g) * g, round(y / g) * g

    def _find_node_at(self, x: float, y: float) -> Optional[int]:
        r = self.RADIUS + 4
        for node in self._nodes:
            if (node.x - x) ** 2 + (node.y - y) ** 2 <= r * r:
                return node.idx
        return None

    def _on_press(self, event):
        idx = self._find_node_at(event.x, event.y)
        self._drag_start = (event.x, event.y)
        self._drag_moved = False
        self._drag_node_idx = idx

    def _on_drag(self, event):
        if self._drag_node_idx is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        if not self._drag_moved and (abs(dx) < 5 and abs(dy) < 5):
            return
        self._drag_moved = True
        node = self._nodes[self._drag_node_idx]
        nx, ny = self._snap(event.x, event.y)
        # Move canvas items
        ddx, ddy = nx - node.x, ny - node.y
        for item in node.canvas_items:
            self._canvas.move(item, ddx, ddy)
        if node.label_item:
            self._canvas.move(node.label_item, ddx, ddy)
        node.x, node.y = nx, ny
        # Redraw edges touching this node
        for key, edge in self._edges.items():
            if edge.i == node.idx or edge.j == node.idx:
                self._redraw_edge(edge)

    def _on_release(self, event):
        if not self._drag_moved:
            idx = self._drag_node_idx
            x, y = self._snap(event.x, event.y)
            if idx is None:
                self._add_node(x, y)
            else:
                self._toggle_select(idx)
        self._drag_node_idx = None
        self._drag_moved = False

    def _on_motion(self, event):
        if not self._snap_var.get():
            return
        x, y = self._snap(event.x, event.y)
        if self._hover_item is None:
            self._hover_item = self._canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3, outline="#cbbfa8")
        else:
            self._canvas.coords(self._hover_item, x - 3, y - 3, x + 3, y + 3)

    def _on_right_click(self, event):
        idx = self._find_node_at(event.x, event.y)
        menu = tk.Menu(self.root, tearoff=0, bg=C["panel"], fg=C["text"])
        if idx is not None:
            menu.add_command(label=f"Set bias for node {idx}",
                             command=lambda: self._set_bias_for(idx))
            menu.add_command(label=f"Delete node {idx}",
                             command=lambda: self._delete_node(idx))
        else:
            menu.add_command(label="Add node here",
                             command=lambda: self._add_node(*self._snap(event.x, event.y)))
        if len(self._selected) == 2:
            i, j = sorted(self._selected)
            key = (i, j)
            lbl = ("Update" if key in self._edges else "Add") + f" edge ({i}–{j})"
            menu.add_command(label=lbl, command=self._set_weight)
            if key in self._edges:
                menu.add_command(label=f"Delete edge ({i}–{j})",
                                 command=lambda k=key: self._delete_edge(k))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _toggle_select(self, idx: int):
        if idx in self._selected:
            self._selected.remove(idx)
        else:
            self._selected.append(idx)
        self._refresh_selection()

    def _refresh_selection(self):
        for node in self._nodes:
            col = C["node_sel"] if node.idx in self._selected else C["muted"]
            for item in node.canvas_items:
                self._canvas.itemconfig(item, outline=col)

    def _add_node(self, x: float, y: float, bias: float = 0.0):
        idx = len(self._nodes)
        node = Node(idx=idx, x=x, y=y, bias=bias)
        node.canvas_items = self._sketch_circle(x, y, self.RADIUS, C["muted"], C["node"], seed=idx * 97 + 7)
        node.label_item = self._canvas.create_text(x, y, text=str(idx), fill=C["text"],
                                                    font=("Helvetica", 9, "bold"))
        self._nodes.append(node)
        self._log(f"Node {idx} added  (bias={bias})")
        self._update_status()

    def _add_edge(self, i: int, j: int, weight: float):
        if i > j:
            i, j = j, i
        key = (i, j)
        ni, nj = self._nodes[i], self._nodes[j]
        if key in self._edges:
            self._edges[key].weight = weight
            self._redraw_edge(self._edges[key])
            self._log(f"Edge ({i},{j}) updated  w={weight}")
            return
        items = self._sketch_line(ni.x, ni.y, nj.x, nj.y, C["edge"], 2, seed=i * 173 + j * 37)
        for item in items:
            self._canvas.tag_lower(item)
        mx, my = (ni.x + nj.x) / 2, (ni.y + nj.y) / 2
        lbl = self._canvas.create_text(mx, my - 8, text=f"{weight:+.2g}",
                                        font=("Helvetica", 7), fill=C["muted"])
        if not self._ew_var.get():
            self._canvas.itemconfig(lbl, state="hidden")
        edge = Edge(i=i, j=j, weight=weight, canvas_items=items, label_item=lbl)
        self._edges[key] = edge
        self._log(f"Edge ({i},{j}) added  w={weight}")
        self._update_status()

    def _redraw_edge(self, edge: Edge):
        ni, nj = self._nodes[edge.i], self._nodes[edge.j]
        for item in edge.canvas_items:
            self._canvas.delete(item)
        items = self._sketch_line(ni.x, ni.y, nj.x, nj.y, C["edge"], 2,
                                  seed=edge.i * 173 + edge.j * 37)
        for item in items:
            self._canvas.tag_lower(item)
        edge.canvas_items = items
        if edge.label_item:
            mx, my = (ni.x + nj.x) / 2, (ni.y + nj.y) / 2
            self._canvas.coords(edge.label_item, mx, my - 8)

    def _delete_selected(self, _event=None):
        if not self._selected:
            return
        for idx in list(self._selected):
            self._delete_node(idx)
        self._selected.clear()
        self._update_status()

    def _delete_node(self, idx: int):
        if idx >= len(self._nodes):
            return
        node = self._nodes[idx]
        for item in node.canvas_items:
            self._canvas.delete(item)
        if node.label_item:
            self._canvas.delete(node.label_item)
        # Remove touching edges
        for key in [k for k in self._edges if k[0] == idx or k[1] == idx]:
            self._delete_edge(key)
        # Remap indices
        self._nodes.pop(idx)
        old_to_new = {}
        for new_i, n in enumerate(self._nodes):
            old_to_new[n.idx] = new_i
            n.idx = new_i
            if n.label_item:
                self._canvas.itemconfig(n.label_item, text=str(new_i))
        new_edges = {}
        for edge in self._edges.values():
            ni = old_to_new.get(edge.i, edge.i)
            nj = old_to_new.get(edge.j, edge.j)
            key = tuple(sorted((ni, nj)))
            edge.i, edge.j = key
            new_edges[key] = edge
        self._edges = new_edges
        self._selected = [old_to_new[s] for s in self._selected if s in old_to_new]
        self._refresh_selection()
        self._update_status()

    def _delete_edge(self, key: Tuple[int, int]):
        if key not in self._edges:
            return
        edge = self._edges.pop(key)
        for item in edge.canvas_items:
            self._canvas.delete(item)
        if edge.label_item:
            self._canvas.delete(edge.label_item)

    def _set_bias(self, _event=None):
        if len(self._selected) != 1:
            messagebox.showinfo("Bias", "Select exactly one node first.")
            return
        self._set_bias_for(self._selected[0])

    def _set_bias_for(self, idx: int):
        node = self._nodes[idx]
        v = simpledialog.askfloat("Bias", f"Local field hᵢ for node {idx}:",
                                  initialvalue=node.bias)
        if v is not None:
            node.bias = float(v)
            self._log(f"Node {idx} bias = {node.bias}")

    def _set_weight(self, _event=None):
        if len(self._selected) != 2:
            messagebox.showinfo("Weight", "Select exactly two nodes first.")
            return
        i, j = sorted(self._selected)
        current = self._edges.get((i, j), Edge(i, j, 0.0)).weight
        v = simpledialog.askfloat("Edge weight", f"Coupling Jᵢⱼ for edge ({i},{j}):",
                                  initialvalue=current)
        if v is not None:
            self._add_edge(i, j, float(v))

    def _toggle_edge_labels(self):
        state = "normal" if self._ew_var.get() else "hidden"
        for edge in self._edges.values():
            if edge.label_item:
                self._canvas.itemconfig(edge.label_item, state=state)

    def _draw_grid(self):
        for item in self._grid_items:
            self._canvas.delete(item)
        self._grid_items = []
        if not self._grid_var.get():
            return
        w = self._canvas.winfo_width() or 900
        h = self._canvas.winfo_height() or 600
        for x in range(0, w, self.GRID):
            self._grid_items.append(self._canvas.create_line(x, 0, x, h, fill=C["grid"]))
        for y in range(0, h, self.GRID):
            self._grid_items.append(self._canvas.create_line(0, y, w, y, fill=C["grid"]))
        for item in self._grid_items:
            self._canvas.tag_lower(item)

    # ═════════════════════════════════════════════════════════════════════════
    #  PRESETS
    # ═════════════════════════════════════════════════════════════════════════
    def _load_preset(self):
        name = self._preset_var.get()
        self._clear()
        if name == "MaxCut Ring (6)":
            self._preset_ring(6, "MaxCut")
        elif name == "MaxCut Complete (6)":
            self._preset_complete(6, "MaxCut")
        elif name == "MaxCut Bipartite (4,4)":
            self._preset_bipartite(4, 4)
        elif name == "Random QUBO (8)":
            self._preset_random_qubo(8, seed=7)
        elif name == "Spin Glass (8)":
            self._preset_spin_glass(8, seed=13)
        elif name == "Petersen Graph":
            self._preset_petersen()
        elif name == "Frustrated Triangles (9)":
            self._preset_frustrated_triangles()
        elif name == "Chimera Unit Cell (8)":
            self._preset_chimera()
        elif name == "Number Partition (8)":
            self._preset_number_partition()

    def _layout_circle(self, n, cx=450, cy=280, r=210):
        return [(cx + r * math.cos(2 * math.pi * k / n - math.pi / 2),
                 cy + r * math.sin(2 * math.pi * k / n - math.pi / 2))
                for k in range(n)]

    def _preset_ring(self, n, problem="MaxCut"):
        pts = self._layout_circle(n)
        for x, y in pts:
            self._add_node(x, y)
        for i in range(n):
            self._add_edge(i, (i + 1) % n, 1.0)
        self._problem_var.set(problem)
        self._log(f"Preset: {problem} Ring ({n})")

    def _preset_complete(self, n, problem="MaxCut"):
        pts = self._layout_circle(n)
        for x, y in pts:
            self._add_node(x, y)
        for i in range(n):
            for j in range(i + 1, n):
                self._add_edge(i, j, 1.0)
        self._problem_var.set(problem)
        self._log(f"Preset: {problem} Complete ({n})")

    def _preset_bipartite(self, a, b):
        h = 560 / (max(a, b) + 1)
        for i in range(a):
            self._add_node(200, 80 + i * h)
        for j in range(b):
            self._add_node(680, 80 + j * h)
        for i in range(a):
            for j in range(a, a + b):
                self._add_edge(i, j, 1.0)
        self._problem_var.set("MaxCut")
        self._log("Preset: MaxCut Bipartite (4,4)")

    def _preset_random_qubo(self, n, seed=7):
        rng = np.random.default_rng(seed)
        pts = self._layout_circle(n)
        for x, y in pts:
            self._add_node(x, y, bias=round(float(rng.normal(scale=0.5)), 2))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.45:
                    w = round(float(rng.normal(scale=0.5)), 2)
                    self._add_edge(i, j, w)
        self._problem_var.set("QUBO")
        self._log("Preset: Random QUBO (8)")

    def _preset_spin_glass(self, n, seed=13):
        rng = np.random.default_rng(seed)
        pts = self._layout_circle(n)
        for x, y in pts:
            self._add_node(x, y, bias=round(float(rng.choice([-1.0, 1.0]) * 0.2), 2))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.55:
                    w = float(rng.choice([-1.0, 1.0]))
                    self._add_edge(i, j, w)
        self._problem_var.set("QUBO")
        self._log("Preset: Spin Glass (8) — ±J couplings")

    def _preset_petersen(self):
        # Outer 5-cycle + inner pentagram
        cx, cy, ro, ri = 450, 280, 210, 100
        pts_out = [(cx + ro * math.cos(2 * math.pi * k / 5 - math.pi / 2),
                    cy + ro * math.sin(2 * math.pi * k / 5 - math.pi / 2)) for k in range(5)]
        pts_in  = [(cx + ri * math.cos(2 * math.pi * k / 5 - math.pi / 2),
                    cy + ri * math.sin(2 * math.pi * k / 5 - math.pi / 2)) for k in range(5)]
        for x, y in pts_out + pts_in:
            self._add_node(x, y)
        for i in range(5):
            self._add_edge(i, (i + 1) % 5, 1.0)      # outer ring
        for i in range(5):
            self._add_edge(5 + i, 5 + (i + 2) % 5, 1.0)  # inner pentagram
        for i in range(5):
            self._add_edge(i, 5 + i, 1.0)             # spokes
        self._problem_var.set("MaxCut")
        self._log("Preset: Petersen Graph (10 nodes) — max cut = 12")

    def _preset_frustrated_triangles(self):
        # 3×3 triangular lattice with antiferromagnetic J = −1
        cx, cy, sp = 200, 120, 130
        pts = []
        for row in range(3):
            for col in range(3):
                x = cx + col * sp + (row % 2) * (sp / 2)
                y = cy + row * sp
                pts.append((x, y))
                self._add_node(x, y, bias=0.0)
        # Horizontal edges
        for row in range(3):
            for col in range(2):
                self._add_edge(row * 3 + col, row * 3 + col + 1, -1.0)
        # Vertical + diagonal edges
        for row in range(2):
            for col in range(3):
                self._add_edge(row * 3 + col, (row + 1) * 3 + col, -1.0)
                if row % 2 == 0 and col > 0:
                    self._add_edge(row * 3 + col, (row + 1) * 3 + col - 1, -1.0)
                if row % 2 == 1 and col < 2:
                    self._add_edge(row * 3 + col, (row + 1) * 3 + col + 1, -1.0)
        self._problem_var.set("QUBO")
        self._log("Preset: Frustrated Triangles (9) — AF ±J on triangular lattice")

    def _preset_chimera(self):
        # One 4+4 Chimera unit cell: L=[0..3] R=[4..7], all L-R edges
        left_x, right_x, sp_y = 220, 660, 100
        top_y = 80
        for i in range(4):
            self._add_node(left_x, top_y + i * sp_y)
        for j in range(4):
            self._add_node(right_x, top_y + j * sp_y)
        rng = np.random.default_rng(0)
        for i in range(4):
            for j in range(4, 8):
                w = round(float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5)), 2)
                self._add_edge(i, j, w)
        self._problem_var.set("QUBO")
        self._log("Preset: Chimera Unit Cell (8) — D-Wave topology, random ±J")

    def _preset_number_partition(self):
        nums = [3, 5, 7, 9, 11, 13, 15, 17]
        n = len(nums)
        pts = self._layout_circle(n)
        for k, (x, y) in enumerate(pts):
            self._add_node(x, y, bias=0.0)
            self._canvas.itemconfig(self._nodes[-1].label_item,
                                    text=str(nums[k]))
        for i in range(n):
            for j in range(i + 1, n):
                self._add_edge(i, j, round(2.0 * nums[i] * nums[j], 1))
        self._problem_var.set("NumberPartition")
        self._log(f"Preset: Number Partition {nums}  (sum={sum(nums)}, target={sum(nums)/2})")

    # ═════════════════════════════════════════════════════════════════════════
    #  PROBLEM ENCODING
    # ═════════════════════════════════════════════════════════════════════════
    def _build_qubo(self) -> np.ndarray:
        n = len(self._nodes)
        Q = np.zeros((n, n), dtype=float)
        prob = self._problem_var.get()
        A = max(1.0, float(self._penalty_var.get()))

        if prob == "MaxCut":
            for nd in self._nodes:
                Q[nd.idx, nd.idx] += nd.bias
            for (i, j), e in self._edges.items():
                Q[i, i] -= e.weight; Q[j, j] -= e.weight
                Q[i, j] += e.weight; Q[j, i] += e.weight
        elif prob == "MaxIndependentSet":
            for nd in self._nodes:
                Q[nd.idx, nd.idx] += -(nd.bias if nd.bias != 0 else 1.0)
            for (i, j), e in self._edges.items():
                Q[i, j] += 0.5 * A; Q[j, i] += 0.5 * A
        elif prob == "MaxClique":
            for nd in self._nodes:
                Q[nd.idx, nd.idx] += -(nd.bias if nd.bias != 0 else 1.0)
            edge_set = set(self._edges.keys())
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) not in edge_set:
                        Q[i, j] += 0.5 * A; Q[j, i] += 0.5 * A
        elif prob == "MinVertexCover":
            for nd in self._nodes:
                Q[nd.idx, nd.idx] += nd.bias if nd.bias != 0 else 1.0
            for (i, j), e in self._edges.items():
                Q[i, i] -= A; Q[j, j] -= A
                Q[i, j] += 0.5 * A; Q[j, i] += 0.5 * A
        elif prob == "NumberPartition":
            # Encode via Ising directly: return None signal
            pass
        else:  # raw QUBO
            for nd in self._nodes:
                Q[nd.idx, nd.idx] += nd.bias
            for (i, j), e in self._edges.items():
                Q[i, j] += e.weight; Q[j, i] += e.weight
        return Q

    def _build_number_partition_ising(self):
        """Build DenseIsing for number partition from edge weights (Jij = 2*ai*aj)."""
        from ._qanneal import DenseIsing
        n = len(self._nodes)
        h = np.array([nd.bias for nd in self._nodes], dtype=float)
        J = np.zeros((n, n), dtype=float)
        for (i, j), e in self._edges.items():
            J[i, j] = J[j, i] = e.weight
        return DenseIsing(h, J)

    def _qubo_energy(self, Q: np.ndarray, x: np.ndarray) -> float:
        return float(x @ Q @ x)

    def _brute_force(self, Q: np.ndarray):
        n = Q.shape[0]
        best_e, best_x = float("inf"), None
        for mask in range(1 << n):
            x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
            e = self._qubo_energy(Q, x)
            if e < best_e:
                best_e, best_x = e, x.astype(int)
        return best_e, best_x

    def _maxcut_value(self, x: np.ndarray) -> float:
        val = 0.0
        for (i, j), e in self._edges.items():
            val += e.weight * (x[i] + x[j] - 2 * x[i] * x[j])
        return val

    # ═════════════════════════════════════════════════════════════════════════
    #  SCHEDULE BUILDING
    # ═════════════════════════════════════════════════════════════════════════
    def _build_schedule(self, ising_or_qubo):
        method  = self._method_var.get()
        replicas = max(2, int(self._replicas_var.get()))
        if self._autotune_var.get():
            mode = self._mode_var.get()
            if method == "sa":
                return auto_schedule_sa_tuned(ising_or_qubo, mode=mode)
            elif method in ("sqa", "ctpimc"):
                return auto_schedule_sqa_tuned(ising_or_qubo, mode=mode)
            else:  # sqapt
                return auto_ladder_sqa_tuned(ising_or_qubo, replicas=replicas, mode=mode)
        else:
            steps   = max(2, int(self._steps_var.get()))
            bs, be  = float(self._beta_s_var.get()), float(self._beta_e_var.get())
            gs, ge  = float(self._gamma_s_var.get()), float(self._gamma_e_var.get())
            if method == "sa":
                return auto_schedule_sa(steps=steps, beta_start=bs, beta_end=be)
            else:
                return auto_schedule_sqa(steps=steps, beta_start=bs, beta_end=be,
                                         gamma_start=gs, gamma_end=ge)

    # ═════════════════════════════════════════════════════════════════════════
    #  SOLVE FLOW
    # ═════════════════════════════════════════════════════════════════════════
    def _run_solve(self):
        if not self._nodes:
            messagebox.showinfo("Solve", "Add at least one node first.")
            return
        if self._solver_thread and self._solver_thread.is_alive():
            self._log("Solver already running...")
            return
        self._set_controls(False)
        self._log("─" * 35)
        self._log(f"Solving: {self._problem_var.get()}  method={self._method_var.get()}")

        prob = self._problem_var.get()
        is_partition = (prob == "NumberPartition")

        if is_partition:
            problem_input = self._build_number_partition_ising()
        else:
            problem_input = self._build_qubo()

        method   = self._method_var.get()
        reads    = max(1, int(self._reads_var.get()))
        sweeps   = max(1, int(self._sweeps_var.get()))
        slices   = max(1, int(self._slices_var.get()))
        worldline = max(0, int(self._worldline_var.get()))
        replicas  = max(2, int(self._replicas_var.get()))
        pt_steps  = max(10, int(self._pt_steps_var.get()))
        ret_bits  = not is_partition

        schedule = self._build_schedule(problem_input)

        self._solver_thread = threading.Thread(
            target=self._solve_worker,
            args=(problem_input, method, reads, sweeps, slices, worldline,
                  replicas, pt_steps, schedule, prob, ret_bits),
            daemon=True,
        )
        self._solver_thread.start()

    def _solve_worker(self, problem, method, reads, sweeps, slices, worldline,
                      replicas, pt_steps, schedule, prob, ret_bits):
        try:
            result = solve(
                problem,
                method=method,
                reads=reads,
                sweeps_per_beta=sweeps,
                trotter_slices=slices,
                worldline_sweeps=worldline,
                replicas=replicas,
                pt_steps=pt_steps,
                schedule=schedule,
                return_bits=ret_bits,
                progress=False,
            )

            trace = result.trace or []

            # Build animation data
            betas  = list(getattr(schedule, "betas",  []))
            gammas = list(getattr(schedule, "gammas", [])) or None

            anim_data = {
                "method":       method,
                "trace":        trace,
                "betas":        betas,
                "gammas":       gammas,
                "best_energy":  result.best_energy,
                "best_sample":  result.best_sample,
                "n_nodes":      len(self._nodes),
                "prob":         prob,
                "ret_bits":     ret_bits,
            }

            # Brute force verify
            opt_e, opt_x = None, None
            if self._verify_var.get() and prob != "NumberPartition" and not ret_bits is False:
                n = len(self._nodes)
                if n <= 18:
                    Q = self._build_qubo()
                    opt_e, opt_x = self._brute_force(Q)
                    anim_data["oracle"] = opt_e
                    anim_data["oracle_x"] = opt_x

            self.root.after(0, lambda: self._solve_done(result, anim_data, opt_e, opt_x, prob))
        except Exception as exc:
            self.root.after(0, lambda: self._solve_failed(str(exc)))

    def _solve_done(self, result, anim_data, opt_e, opt_x, prob):
        self._log(f"Best energy : {result.best_energy:.6f}")
        self._log(f"Best sample : {result.best_sample}")
        if opt_e is not None:
            gap = result.best_energy - opt_e
            self._log(f"Oracle      : {opt_e:.6f}  (gap={gap:.4f})")
        if prob == "MaxCut":
            bits = result.best_sample
            cv = self._maxcut_value(bits.astype(float))
            self._log(f"MaxCut value: {cv:.2f}")
        if prob == "NumberPartition":
            diff = abs(float(np.dot(
                [e.weight / (2 * max(1, abs(e.weight))) for e in self._edges.values()],
                [1] * len(self._edges)
            )))  # rough
            self._log("Partition result in best_sample (spins ±1)")

        # Colour nodes
        if self._cut_var.get():
            if opt_x is not None:
                self._draw_solution(opt_x)
            else:
                smp = result.best_sample
                if smp.max() <= 1 and smp.min() >= 0:
                    self._draw_solution(smp)
                else:  # spins → bits
                    self._draw_solution(((smp + 1) // 2).astype(int))

        self._set_controls(True)
        self._start_animation(anim_data)
        self._nb.select(1)  # switch to Animation Lab

    def _solve_failed(self, err: str):
        self._log(f"ERROR: {err}")
        self._set_controls(True)

    def _draw_solution(self, x: np.ndarray):
        for node in self._nodes:
            if node.idx >= len(x):
                continue
            color = C["node_a"] if x[node.idx] == 0 else C["node_b"]
            for item in node.canvas_items:
                self._canvas.itemconfig(item, fill=color)
        for (i, j), edge in self._edges.items():
            if i >= len(x) or j >= len(x):
                continue
            cut = (x[i] != x[j])
            col = C["edge_cut"] if cut else C["edge"]
            lw  = 3 if cut else 2
            for item in edge.canvas_items:
                self._canvas.itemconfig(item, fill=col, width=lw)

    def _set_controls(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self._run_btn.config(state=state)

    # ═════════════════════════════════════════════════════════════════════════
    #  ANIMATION
    # ═════════════════════════════════════════════════════════════════════════
    def _start_animation(self, data: dict):
        if not HAS_MPL or self._anim_fig is None:
            return
        # Cancel running animation
        if self._anim_after_id:
            self.root.after_cancel(self._anim_after_id)
        self._anim_playing = False
        self._anim_data = data
        trace = data["trace"]
        if not trace:
            trace = [data["best_energy"]]
        # Resample to ≤240 frames for smooth animation
        N = len(trace)
        if N > 240:
            idx = np.round(np.linspace(0, N - 1, 240)).astype(int)
            trace = [trace[i] for i in idx]
        self._anim_trace = trace
        self._anim_total = len(trace)
        self._anim_frame = 0
        method  = data["method"]
        betas   = data.get("betas", [])
        gammas  = data.get("gammas")

        # Resample schedule to same number of frames
        if betas:
            idx_b = np.round(np.linspace(0, len(betas) - 1, self._anim_total)).astype(int)
            self._anim_betas  = [betas[i] for i in idx_b]
            self._anim_gammas = ([gammas[i] for i in idx_b] if gammas else None)
        else:
            self._anim_betas  = list(np.linspace(0.1, 5.0, self._anim_total))
            self._anim_gammas = None

        steps = list(range(self._anim_total))
        best_e = data["best_energy"]
        oracle = data.get("oracle")

        # ── Setup matplotlib artists ─────────────────────────────────────────
        ax_s, ax_e, ax_x = self._ax_sched, self._ax_energy, self._ax_extra
        for ax in (ax_s, ax_e, ax_x):
            ax.clear()
            ax.set_facecolor("#fefaf3")
            for sp in ax.spines.values(): sp.set_color("#ccc0a8")

        # Schedule
        ax_s.plot(steps, self._anim_betas, color="#4c8bf5", lw=2, label="β (inverse temp)")
        if self._anim_gammas:
            ax_s.plot(steps, self._anim_gammas, color="#f5a623", lw=2, label="Γ (transverse field)")
        ax_s.set_title("Schedule", fontsize=9); ax_s.set_ylabel("Value", fontsize=8)
        ax_s.legend(fontsize=8, loc="upper left"); ax_s.tick_params(labelsize=7)
        self._sched_cursor = ax_s.axvline(0, color="#cc3d2b", lw=1.5, alpha=0.8)

        # Energy trace
        trace_arr = np.array(trace)
        ax_e.fill_between(steps, trace_arr, trace_arr.min(), alpha=0.12, color="#4c8bf5")
        (self._energy_line,) = ax_e.plot([], [], color="#4c8bf5", lw=2)
        self._energy_cursor  = ax_e.axvline(0, color="#cc3d2b", lw=1.5, alpha=0.8)
        ax_e.axhline(best_e, color="#2ca02c", lw=1.2, ls="--", label=f"Best {best_e:.3f}")
        if oracle is not None:
            ax_e.axhline(oracle, color="#c84b31", lw=1.2, ls=":", label=f"Oracle {oracle:.3f}")
        ax_e.set_xlim(-1, self._anim_total)
        ax_e.set_ylim(trace_arr.min() - 0.5, trace_arr.max() + 0.5)
        ax_e.set_title("Energy trace", fontsize=9); ax_e.set_ylabel("Energy", fontsize=8)
        ax_e.legend(fontsize=8, loc="upper right"); ax_e.tick_params(labelsize=7)

        # Extra subplot (method-specific)
        self._setup_extra_ax(method, ax_x)

        self._anim_fig.tight_layout(pad=1.2)
        self._anim_mpl_canvas.draw()
        self._anim_play()

    def _setup_extra_ax(self, method: str, ax):
        if method == "sa":
            # Boltzmann acceptance curve
            dE = np.linspace(0, 5, 200)
            colors = plt.cm.coolwarm(np.linspace(0, 1, 5))
            betas_sample = np.linspace(0.1, 5.0, 5)
            for beta, col in zip(betas_sample, colors):
                ax.plot(dE, np.exp(-beta * dE), color=col, lw=1.5, alpha=0.7,
                        label=f"β={beta:.1f}")
            self._accept_line, = ax.plot(dE, np.exp(-0.1 * dE), color="#cc3d2b", lw=2.5)
            ax.set_xlabel("ΔE (energy increase)", fontsize=8)
            ax.set_ylabel("P(accept)", fontsize=8)
            ax.set_title("Boltzmann acceptance: P = exp(−β·ΔE)", fontsize=9)
            ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=7)
        elif method in ("sqa", "ctpimc"):
            # Schematic Trotter grid
            M, n = 6, min(self._anim_data.get("n_nodes", 8), 12)
            rng = np.random.default_rng(0)
            self._trotter_data = rng.choice([-1, 1], size=(M, n)).astype(float)
            self._trotter_im = ax.imshow(
                self._trotter_data, cmap="RdBu_r", aspect="auto",
                vmin=-1, vmax=1, extent=[-0.5, n - 0.5, -0.5, M - 0.5])
            ax.set_xlabel("Spin index", fontsize=8)
            ax.set_ylabel("Trotter slice", fontsize=8)
            title = "Trotter slices" if method == "sqa" else "CT-PIMC worldlines (schematic)"
            ax.set_title(title + "  (blue=↓  red=↑)", fontsize=9)
            ax.tick_params(labelsize=7)
        elif method == "sqapt":
            R = min(len(self._anim_betas), self._anim_data.get("n_nodes", 4))
            R = max(2, min(R, 6))
            self._sqapt_colors = plt.cm.plasma(np.linspace(0.15, 0.9, R))
            self._sqapt_lines = []
            for r in range(R):
                ln, = ax.plot([], [], color=self._sqapt_colors[r], lw=1.8,
                              label=f"R{r} β={self._anim_betas[0]:.1f}")
                self._sqapt_lines.append(ln)
            ax.set_xlim(-1, self._anim_total)
            emin = self._anim_data["best_energy"]
            emax = max(self._anim_trace) if self._anim_trace else emin + 5
            ax.set_ylim(emin - 0.5, emax + 1)
            ax.set_xlabel("PT epoch", fontsize=8)
            ax.set_ylabel("Replica energy", fontsize=8)
            ax.set_title("Replica energies (SQAPT)", fontsize=9)
            ax.legend(fontsize=7, ncol=2); ax.tick_params(labelsize=7)
            self._sqapt_replica_traces = []
            rng = np.random.default_rng(42)
            for r in range(R):
                offset = (R - 1 - r) * 0.8
                base = np.array(self._anim_trace) + offset
                noise = rng.normal(0, 0.2, len(self._anim_trace))
                self._sqapt_replica_traces.append(base + noise)

    def _update_anim_frame(self, f: int):
        trace = self._anim_trace
        method = self._anim_data["method"]
        steps = list(range(f + 1))
        energy_so_far = trace[:f + 1]

        # Schedule cursor
        self._sched_cursor.set_xdata([f, f])

        # Energy line
        self._energy_line.set_data(steps, energy_so_far)
        self._energy_cursor.set_xdata([f, f])

        frac = f / max(self._anim_total - 1, 1)

        # Extra subplot update
        if method == "sa":
            beta_now = self._anim_betas[f]
            dE = np.linspace(0, 5, 200)
            self._accept_line.set_ydata(np.exp(-beta_now * dE))
        elif method in ("sqa", "ctpimc"):
            # Trotter grid: transition from random to aligned with solution fraction
            n = self._trotter_data.shape[1]
            rng = np.random.default_rng(f)
            rand_grid = rng.choice([-1, 1], size=self._trotter_data.shape).astype(float)
            # Lerp from random to ordered
            ordered = np.ones_like(rand_grid)
            blend = frac ** 1.5
            blended = blend * ordered + (1 - blend) * rand_grid
            self._trotter_im.set_data(np.clip(blended, -1, 1))
        elif method == "sqapt":
            for r, ln in enumerate(self._sqapt_lines):
                rt = self._sqapt_replica_traces[r]
                ln.set_data(list(range(f + 1)), rt[:f + 1])

        # Graph node colour fade
        if self._anim_data.get("best_sample") is not None:
            self._update_node_colors(frac)

        # Explanation text
        self._explain_var.set(_explain_at(method, frac))

        # Frame counter
        self._frame_var.set(f"Frame {f+1} / {self._anim_total}")

        self._anim_mpl_canvas.draw_idle()

    def _update_node_colors(self, frac: float):
        sample = self._anim_data["best_sample"]
        for node in self._nodes:
            if node.idx >= len(sample):
                continue
            final = C["node_a"] if sample[node.idx] == 0 else C["node_b"]
            # Blend from neutral to final
            if frac < 0.3:
                col = C["node"]
            elif frac < 0.7:
                t = (frac - 0.3) / 0.4
                col = self._blend_color(C["node"], final, t)
            else:
                col = final
            for item in node.canvas_items:
                try:
                    self._canvas.itemconfig(item, fill=col)
                except Exception:
                    pass

    @staticmethod
    def _blend_color(c1: str, c2: str, t: float) -> str:
        def parse(c):
            c = c.lstrip("#")
            return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
        r1, g1, b1 = parse(c1); r2, g2, b2 = parse(c2)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _anim_step(self):
        if not self._anim_playing:
            return
        if self._anim_frame >= self._anim_total:
            self._anim_playing = False
            self._explain_var.set(
                _explain_at(self._anim_data["method"], 1.0) +
                f"\n\n✅  Animation complete. Best energy: {self._anim_data['best_energy']:.4f}")
            return
        self._update_anim_frame(self._anim_frame)
        self._anim_frame += 1
        delay = max(5, int(self._speed_var.get()))
        self._anim_after_id = self.root.after(delay, self._anim_step)

    def _anim_play(self):
        if not self._anim_data:
            return
        self._anim_playing = True
        self._anim_step()

    def _anim_pause(self):
        self._anim_playing = False
        if self._anim_after_id:
            self.root.after_cancel(self._anim_after_id)

    def _anim_reset(self):
        self._anim_pause()
        self._anim_frame = 0
        if self._anim_data:
            self._update_anim_frame(0)

    # ═════════════════════════════════════════════════════════════════════════
    #  SAVE / LOAD GRAPH
    # ═════════════════════════════════════════════════════════════════════════
    def _save_graph(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        data = {
            "nodes": [{"idx": n.idx, "x": n.x, "y": n.y, "bias": n.bias}
                      for n in self._nodes],
            "edges": [{"i": e.i, "j": e.j, "weight": e.weight}
                      for e in self._edges.values()],
            "problem": self._problem_var.get(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._log(f"Saved → {path}")

    def _load_graph(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        self._clear()
        for nd in data.get("nodes", []):
            self._add_node(nd["x"], nd["y"], nd.get("bias", 0.0))
        for e in data.get("edges", []):
            self._add_edge(e["i"], e["j"], e["weight"])
        self._problem_var.set(data.get("problem", "QUBO"))
        self._log(f"Loaded ← {path}")

    # ═════════════════════════════════════════════════════════════════════════
    #  MISC / UTILITY
    # ═════════════════════════════════════════════════════════════════════════
    def _on_method_change(self, *_):
        method = self._method_var.get()
        is_sqa_like = method in ("sqa", "sqapt", "ctpimc")
        state = tk.NORMAL if is_sqa_like else tk.DISABLED
        for widget in self._sqa_frame.winfo_children():
            try:
                widget.config(state=state)
            except Exception:
                pass

    def _on_autotune_change(self):
        state = tk.DISABLED if self._autotune_var.get() else tk.NORMAL
        for widget in self._manual_frame.winfo_children():
            try:
                widget.config(state=state)
            except Exception:
                pass

    def _clear(self):
        self._canvas.delete("all")
        self._nodes.clear(); self._edges.clear()
        self._selected.clear()
        self._grid_items = []
        self._hover_item = None
        self._draw_grid()
        self._log("Graph cleared.")
        self._update_status()

    def _update_status(self):
        self._status_var.set(
            f"Nodes: {len(self._nodes)} | Edges: {len(self._edges)}"
            f"  —  Method: {self._method_var.get().upper()}"
            f"  |  Problem: {self._problem_var.get()}")

    def _log(self, msg: str):
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)

    def _section(self, parent, title: str):
        f = tk.Frame(parent, bg=C["section_bg"], height=22)
        f.pack(fill=tk.X, pady=(8, 2))
        tk.Label(f, text=f"  {title}", bg=C["section_bg"], fg=C["muted"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w")

    def _mk_label(self, parent, text: str):
        tk.Label(parent, text=text, bg=C["panel"], fg=C["muted"],
                 font=("Helvetica", 9)).pack(anchor="w", padx=6)

    def _mk_btn(self, parent, text: str, cmd, bg=None, fg=None):
        bg = bg or C["btn_bg"]
        fg = fg or C["btn_fg"]
        return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                         relief=tk.FLAT, cursor="hand2",
                         activebackground=C["accent"],
                         activeforeground="white",
                         font=("Helvetica", 10))

    def _style_menu(self, menu):
        menu.configure(bg=C["panel"], fg=C["text"],
                        activebackground=C["accent"], activeforeground="white",
                        highlightthickness=0, relief=tk.FLAT)
        menu["menu"].config(bg=C["panel"], fg=C["text"])

    def _sketch_line(self, x1, y1, x2, y2, color, width, seed) -> List[int]:
        rng = random.Random(seed)
        items = []
        for _ in range(3):
            j = 1.5
            ax = x1 + rng.uniform(-j, j); ay = y1 + rng.uniform(-j, j)
            bx = x2 + rng.uniform(-j, j); by = y2 + rng.uniform(-j, j)
            items.append(self._canvas.create_line(
                ax, ay, bx, by, fill=color, width=width, capstyle=tk.ROUND))
        return items

    def _sketch_circle(self, x, y, r, outline, fill, seed) -> List[int]:
        rng = random.Random(seed)
        items = []
        for k in range(2):
            j = 1.2
            dx = rng.uniform(-j, j); dy = rng.uniform(-j, j)
            items.append(self._canvas.create_oval(
                x - r + dx, y - r + dy, x + r + dx, y + r + dy,
                outline=outline, width=2,
                fill=fill if k == 0 else ""))
        return items

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
def launch_graph_editor() -> None:
    QAnnealLab().run()
