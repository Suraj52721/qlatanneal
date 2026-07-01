#!/usr/bin/env python3
"""Five structurally distinct HARD Ising problems for the optimal-schedule benchmark.

All are scaled so the coupling RMS j_rms sits ABOVE the SQA/SQAPT start point j_perp_start(~2.88),
i.e. the hottest replica begins in the QUANTUM phase — the only regime where the adaptive J_perp
schedule has a transition to navigate. Each generator returns (h, J, offset, n_actual) with a
symmetric zero-diagonal Ising coupling matrix J and h = 0.

  wmaxcut : weighted MAX-CUT on dense G(n,0.5), J_ij ~ U[1,10]      (dense, j_rms ~ 6)
  sk      : Sherrington-Kirkpatrick, fully-connected J_ij ~ N(0,5)  (mean-field glass, j_rms ~ 5)
  ea3d    : Edwards-Anderson 3D cubic lattice, NN bonds ~ N(0,5)    (short-range glass, j_rms ~ 5)
  w3reg   : weighted MAX-CUT on a random 3-regular graph, ~U[1,10]  (sparse bounded-degree, j_rms ~ 6)
  numpart : number partitioning, wide integers a ~ U{1..1e6}        (rank-1 needle landscape, hard phase)
"""
from __future__ import annotations
import math
import numpy as np


def j_rms(J: np.ndarray) -> float:
    """RMS of the present (non-zero) upper-triangle couplings."""
    iu = np.triu_indices(J.shape[0], k=1)
    off = J[iu]
    nz = off[off != 0.0]
    return math.sqrt(float(np.mean(nz * nz))) if nz.size else 0.0


def make_wmaxcut(n: int, seed: int):
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    present = rng.random(iu[0].size) < 0.5
    J[iu] = rng.uniform(1.0, 10.0, size=iu[0].size) * present
    J = J + J.T
    return np.zeros(n), J, 0.0, n


def make_sk(n: int, seed: int, sigma: float = 5.0):
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    J[iu] = rng.normal(0.0, sigma, size=iu[0].size)
    J = J + J.T
    return np.zeros(n), J, 0.0, n


def make_ea3d(n: int, seed: int, sigma: float = 5.0):
    """3D cubic lattice L^3 (L=round(n^(1/3))), periodic, nearest-neighbour Gaussian bonds."""
    L = max(2, round(n ** (1.0 / 3.0)))
    N = L * L * L
    rng = np.random.default_rng(seed)
    J = np.zeros((N, N))
    def idx(x, y, z): return ((x % L) * L + (y % L)) * L + (z % L)
    for x in range(L):
        for y in range(L):
            for z in range(L):
                a = idx(x, y, z)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    b = idx(x + dx, y + dy, z + dz)
                    if a != b and J[a, b] == 0.0:
                        w = float(rng.normal(0.0, sigma))
                        J[a, b] = J[b, a] = w
    return np.zeros(N), J, 0.0, N


def make_w3reg(n: int, seed: int, degree: int = 3):
    """Weighted MAX-CUT on a random d-regular graph (configuration model, simple-graph retry)."""
    rng = np.random.default_rng(seed)
    if (n * degree) % 2 != 0:
        n += 1
    edges = None
    cand = set()
    for _ in range(1000):
        stubs = np.repeat(np.arange(n), degree)
        rng.shuffle(stubs)
        cand = set()
        ok = True
        for k in range(0, len(stubs), 2):
            a, b = int(stubs[k]), int(stubs[k + 1])
            if a == b or (a, b) in cand or (b, a) in cand:
                ok = False
                break
            cand.add((a, b))
        if ok:
            edges = cand
            break
    if edges is None:
        edges = cand
    J = np.zeros((n, n))
    for a, b in edges:
        w = float(rng.uniform(1.0, 10.0))
        J[a, b] = J[b, a] = w
    return np.zeros(n), J, 0.0, n


def make_numpart(n: int, seed: int, bits_max: int = 10 ** 6, target_jrms: float = 5.0):
    """Number partitioning with WIDE integers a_i ~ U{1..10^6} (the computationally HARD phase,
    where near-perfect partitions are exponentially rare). Ising: minimise (sum a_i s_i)^2 ->
    J = a a^T (off-diagonal), offset = 0.5*sum a_i^2.

    Raw couplings a_i a_j ~ 10^12 lie far outside the transverse-field (Trotter) range, so no
    quantum transition would fall inside the accessible J_perp window. We RESCALE J and the offset
    by a global constant to j_rms = target_jrms: an exact energy rescaling that leaves the argmin
    (and the full combinatorial hardness) unchanged while bringing the coupling scale into the
    quantum regime the schedule needs. Use small n (25-30): that is where ~20-bit numbers sit in
    the hard phase (bits/n ~ 0.7-0.8)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, bits_max + 1, size=n).astype(float)
    J = np.outer(a, a)
    np.fill_diagonal(J, 0.0)
    offset = 0.5 * float(np.sum(a * a))
    jr = j_rms(J)
    if jr > 0.0:
        s = target_jrms / jr
        J *= s
        offset *= s
    return np.zeros(n), J, offset, n


GENERATORS = {
    "wmaxcut": make_wmaxcut,
    "sk": make_sk,
    "ea3d": make_ea3d,
    "w3reg": make_w3reg,
    "numpart": make_numpart,
}


def make_problem(name: str, n: int, seed: int):
    return GENERATORS[name](n, seed)
