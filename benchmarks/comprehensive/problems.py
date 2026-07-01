"""
problems.py — 7 problem generators for the comprehensive benchmark.

Each returns (J, h, gs_energy_or_None):
  J is n x n symmetric numpy float64 (Ising couplings; energy uses sum_{i<j} J_ij s_i s_j),
  h is length-n numpy float64 (local fields),
  gs_energy_or_None is the exact ground-state energy (float) or None if unknown.

Energy convention (matches qanneal DenseIsing):
    E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
Any exact gs_energy below is computed with this same sum_{i<j} convention.
"""
import numpy as np


def _ising_energy(J, h, s):
    """E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j, matching DenseIsing."""
    return float(h @ s + 0.5 * (s @ J @ s))   # 0.5 s^T J s == sum_{i<j} for symmetric, 0-diag


def sk_spin_glass(n, rng):
    """Sherrington-Kirkpatrick spin glass. J_ij ~ N(0, 1/sqrt(n)); no fields. GS unknown."""
    J = rng.standard_normal((n, n)) / np.sqrt(n)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    return J, np.zeros(n), None


def ea3d_spin_glass(n_side, rng):
    """Edwards-Anderson 3D lattice (periodic), J_ij ~ U(-1,1) on nearest-neighbour bonds."""
    n = n_side ** 3
    J = np.zeros((n, n))
    def idx(x, y, z): return x * n_side * n_side + y * n_side + z
    for x in range(n_side):
        for y in range(n_side):
            for z in range(n_side):
                i = idx(x, y, z)
                for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    j = idx((x + dx) % n_side, (y + dy) % n_side, (z + dz) % n_side)
                    if i == j:
                        continue
                    c = rng.uniform(-1, 1)
                    J[i, j] += c
                    J[j, i] += c
    return J, np.zeros(n), None


def _random_regular_edges(n, d, rng, max_tries=2000):
    """Configuration-model random d-regular simple graph (networkx-free). Returns edge set.
    Retries until a simple graph is produced; verifies connectivity via union-find."""
    def connected(edges):
        parent = list(range(n))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for a, b in edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        return len({find(i) for i in range(n)}) == 1
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(n), d)
        rng.shuffle(stubs)
        edges, ok = set(), True
        for k in range(0, len(stubs), 2):
            a, b = int(stubs[k]), int(stubs[k + 1])
            if a == b or (a, b) in edges or (b, a) in edges:
                ok = False
                break
            edges.add((a, b))
        if ok and connected(edges):
            return edges
    raise RuntimeError(f"could not build connected {d}-regular graph on {n} nodes")


def random_3regular_maxcut(n, rng):
    """MAX-CUT on a connected random 3-regular graph. J_ij = -1 on edges. n must be even. GS unknown."""
    if n % 2 != 0:
        raise ValueError("3-regular graph requires even n")
    edges = _random_regular_edges(n, 3, rng)
    J = np.zeros((n, n))
    for i, j in edges:
        J[i, j] -= 1.0
        J[j, i] -= 1.0
    return J, np.zeros(n), None


def weighted_maxcut(n, rng):
    """Weighted MAX-CUT on Erdos-Renyi G(n,0.5), weights U[1,10]. J_ij = -w_ij. GS unknown."""
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                w = rng.uniform(1, 10)
                J[i, j] -= w
                J[j, i] -= w
    return J, np.zeros(n), None


def rfim_2d(n_side, h_strength, rng):
    """Random-Field Ising Model, 2D periodic square lattice. J_ij = +1; h_i ~ U(-h,+h)."""
    n = n_side ** 2
    J = np.zeros((n, n))
    for i in range(n_side):
        for j in range(n_side):
            idx = i * n_side + j
            right = i * n_side + (j + 1) % n_side
            down = ((i + 1) % n_side) * n_side + j
            J[idx, right] += 1.0; J[right, idx] += 1.0
            J[idx, down] += 1.0; J[down, idx] += 1.0
    h = rng.uniform(-h_strength, h_strength, size=n)
    return J, h, None


def planted_partition(n, rng, p_in=0.7, p_out=0.3, J_in=-2.0, J_out=1.0):
    """Planted bisection. Within-community bonds J_in (prob p_in); cross J_out (prob p_out).
    Signs follow qanneal's minimise-E = +sum J s s convention: J_in < 0 (ferromagnetic within
    community, same spin lowers energy) and J_out > 0 (antiferromagnetic across communities,
    opposite spins lower energy). The planted bisection then satisfies every bond simultaneously,
    so it IS the ground state; gs_energy is computed exactly (sum_{i<j} convention)."""
    assert n % 2 == 0
    community = np.array([0] * (n // 2) + [1] * (n // 2))
    gs_spin = np.where(community == 0, 1, -1).astype(np.int64)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same = community[i] == community[j]
            if same and rng.random() < p_in:
                J[i, j] = J[j, i] = J_in
            elif (not same) and rng.random() < p_out:
                J[i, j] = J[j, i] = J_out
    h = np.zeros(n)
    return J, h, _ising_energy(J, h, gs_spin.astype(float))


def number_partitioning(n, rng, max_val=10 ** 6):
    """Number partitioning: minimise (sum_i a_i s_i)^2. Ising J = a a^T (0 diagonal), h = 0.
    Integers a_i ~ U{1..1e6} (hard phase at small n). J rescaled so the coupling scale (j_rms)
    is in the quantum regime; the rescaling is a global energy factor and does not change the
    optimum. gs computed by brute force for n<=20, else None (sum_{i<j} convention)."""
    a = rng.integers(1, max_val + 1, size=n).astype(float)
    a = a / np.sqrt(np.sum(a ** 2)) * n          # normalize coupling scale into the regime
    J = np.outer(a, a)
    np.fill_diagonal(J, 0.0)
    h = np.zeros(n)
    gs_energy = None
    if n <= 20:
        best = np.inf
        for bits in range(2 ** n):
            s = np.array([(1.0 if (bits >> k) & 1 else -1.0) for k in range(n)])
            best = min(best, _ising_energy(J, h, s))
        gs_energy = float(best)
    return J, h, gs_energy
