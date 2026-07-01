"""Task 4.3 — j_rms and j_perp_end resolution for the optimal SQAPT schedule.

Guards against the silent-fallback footgun that hid the frozen-schedule bug: when
j_perp_end resolves to <= j_perp_start the C++ installs an effectively-unbounded ceiling
(and now warns). These tests pin down, for three representative problem classes:

  * mwis_3reg : sparse 3-regular couplings, small j_rms (<< j_perp_start ~ 2.88)
                -> the *degenerate* class where the schedule has little room and the
                   sentinel fallback ceiling is expected to trigger.
  * wmaxcut   : dense G(n,0.5), J ~ U[1,10], large j_rms (>> j_perp_start)
                -> the well-posed class; 5*j_rms is honored verbatim.
  * sk        : J ~ N(0, 1/sqrt(n)) all-to-all, intermediate j_rms.

j_rms here is sqrt(mean(J_ij^2)) over the *non-zero* upper-triangle couplings, matching
qanneal.solver.j_rms_from_problem.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Use the in-tree build (python/qanneal/_qanneal*.so) without requiring an install.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "python"))

solver = pytest.importorskip("qanneal.solver")
_q = pytest.importorskip("qanneal._qanneal")
DenseIsing = _q.DenseIsing
SQAPT = _q.SQAParallelTemperingAnnealer

J_PERP_START_TROTTER = solver.j_perp_from_beta_gamma  # helper alias


def _expected_jrms(J: np.ndarray) -> float:
    iu = np.triu_indices(J.shape[0], k=1)
    off = J[iu]
    nz = off[off != 0.0]
    return math.sqrt(float(np.mean(nz * nz)))


def make_mwis_3reg(n: int = 30, seed: int = 0) -> np.ndarray:
    """Sparse 3-regular-ish symmetric couplings; small j_rms."""
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    for i in range(n):
        # connect to ~3 distinct partners
        for _ in range(3):
            j = int(rng.integers(0, n))
            if j != i:
                w = 0.5  # MWIS penalty-scale couplings are modest
                J[i, j] = w
                J[j, i] = w
    return J


def make_wmaxcut(n: int = 40, seed: int = 1) -> np.ndarray:
    """Dense G(n,0.5) with J ~ U[1,10]; large j_rms."""
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                w = rng.uniform(1.0, 10.0)
                J[i, j] = w
                J[j, i] = w
    return J


def make_sk(n: int = 40, seed: int = 2) -> np.ndarray:
    """Sherrington-Kirkpatrick: all-to-all J ~ N(0, 1/sqrt(n))."""
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    s = 1.0 / math.sqrt(n)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(rng.normal(0.0, s))
            J[i, j] = w
            J[j, i] = w
    return J


@pytest.mark.parametrize("maker", [make_mwis_3reg, make_wmaxcut, make_sk])
def test_j_rms_matches_analytic(maker):
    J = maker()
    got = solver.j_rms_from_problem(J)
    assert got is not None
    assert got == pytest.approx(_expected_jrms(J), rel=1e-9)


def _resolved_end(J: np.ndarray, j_perp_end: float) -> float:
    """Run a 1-step optimal SQAPT and read back the resolved j_perp_end."""
    n = J.shape[0]
    ham = DenseIsing(np.zeros(n), J)
    betas = [0.2, 0.5, 1.0]
    gammas = [2.0, 1.0, 0.5]
    ann = SQAPT(ham, betas, gammas, 16)
    ann.set_seed(7)
    res = ann.run_optimal(num_steps=2, sweeps_per_step=2, worldline_sweeps=0,
                          eps_tilde=0.0, j_perp_end=j_perp_end,
                          calib_probes=4, calib_sweeps=2)
    return res.resolved_j_perp_end, res.j_perp_start


def test_wmaxcut_jperp_end_honored():
    """Well-posed class: 5*j_rms > start, so the explicit target is used verbatim."""
    J = make_wmaxcut()
    jrms = solver.j_rms_from_problem(J)
    target = 5.0 * jrms
    resolved, start = _resolved_end(J, target)
    assert target > start  # well-posed: real room to anneal
    assert resolved == pytest.approx(target, rel=1e-9)


def test_sk_jperp_end_honored():
    J = make_sk()
    jrms = solver.j_rms_from_problem(J)
    # SK j_rms ~ 1/sqrt(n)-scale is tiny; use an explicit target above start for this test.
    _, start = _resolved_end(J, 1.0)
    target = max(5.0 * jrms, start + 2.0)
    resolved, start2 = _resolved_end(J, target)
    assert resolved == pytest.approx(target, rel=1e-9)


def test_mwis_3reg_sentinel_triggers_fallback():
    """Degenerate class: passing the 0.0 sentinel with a small coupling scale resolves a
    target below j_perp_start, so the C++ installs the unbounded fallback ceiling (and warns).
    This is the footgun the audit flags — we assert it is detectable, not silent."""
    J = make_mwis_3reg()
    resolved, start = _resolved_end(J, 0.0)  # 0.0 -> sentinel path
    # Fallback ceiling is start + 1e6 -> resolved is enormously larger than start.
    assert resolved > start + 1e5
