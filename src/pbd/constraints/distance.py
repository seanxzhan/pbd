"""Stretch (distance) constraint, paper eqs. (10)/(11).

C(p_i, p_j) = |p_i - p_j| - rest
"""
from __future__ import annotations

import numpy as np

from pbd.coloring import greedy_color
from pbd.constraints.base import ConstraintGroup
from pbd.mesh import Mesh


def _project_edges(
    P: np.ndarray,
    W: np.ndarray,
    edges: np.ndarray,
    rest: np.ndarray,
) -> np.ndarray:
    """Vectorized distance projection for a subset of edges (eqs. 10/11).

    Returns ΔP: (N, 3). Used by both the full Jacobi batch and the per-
    color Gauss–Seidel sweeps; within a vertex-disjoint color class the
    ``np.add.at`` calls reduce to plain assignments but we leave them as
    ``add.at`` for safety + uniformity.
    """
    out = np.zeros_like(P)
    if edges.shape[0] == 0:
        return out
    i = edges[:, 0]
    j = edges[:, 1]
    diff = P[i] - P[j]
    L = np.linalg.norm(diff, axis=1)
    ok = L > 1e-12
    n = np.zeros_like(diff)
    n[ok] = diff[ok] / L[ok, None]
    C = L - rest

    w1 = W[i]
    w2 = W[j]
    wsum = w1 + w2
    active = ok & (wsum > 0.0)
    s = np.zeros_like(C)
    s[active] = C[active] / wsum[active]

    d1 = -(s * w1)[:, None] * n
    d2 = +(s * w2)[:, None] * n
    np.add.at(out, i, d1)
    np.add.at(out, j, d2)
    return out


class Stretch(ConstraintGroup):
    def __init__(self, edges: np.ndarray, rest: np.ndarray, k: float = 1.0):
        edges = np.ascontiguousarray(edges, dtype=np.int64)
        rest = np.ascontiguousarray(rest, dtype=np.float64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges must be (M, 2); got {edges.shape}")
        if rest.shape != (edges.shape[0],):
            raise ValueError(f"rest shape {rest.shape} != ({edges.shape[0]},)")
        self.idx = edges
        self.rest = rest
        self.k = float(k)
        self._colors: np.ndarray | None = None  # lazy

    @classmethod
    def from_mesh(cls, mesh: Mesh, k: float = 1.0) -> "Stretch":
        e = mesh.edges
        d = mesh.V[e[:, 0]] - mesh.V[e[:, 1]]
        rest = np.linalg.norm(d, axis=1)
        return cls(e, rest, k=k)

    # ------------------------------------------------------------- Jacobi

    def project_batch(self, P: np.ndarray, W: np.ndarray) -> np.ndarray:
        return _project_edges(P, W, self.idx, self.rest)

    # ------------------------------------------------------- Gauss–Seidel

    def _ensure_colors(self) -> np.ndarray:
        if self._colors is None:
            self._colors = greedy_color(self.idx)
        return self._colors

    @property
    def n_colors(self) -> int:
        c = self._ensure_colors()
        return int(c.max() + 1) if c.size else 1

    def project_color(self, P: np.ndarray, W: np.ndarray, c: int) -> np.ndarray:
        colors = self._ensure_colors()
        mask = colors == c
        return _project_edges(P, W, self.idx[mask], self.rest[mask])
