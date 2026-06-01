"""Dihedral bending constraint, paper §4.1 + Appendix A.

For each interior edge (p1, p2) shared by two triangles (p1, p2, p3) and
(p1, p2, p4):

    C(p1, p2, p3, p4) = arccos(n1 · n2) - phi_0
    n1 = (p2 - p1) × (p3 - p1) / |...|
    n2 = (p2 - p1) × (p4 - p1) / |...|

Gradients via Appendix A (eqs. 25-28); final correction eq. (29).
"""
from __future__ import annotations

import numpy as np

from pbd.coloring import greedy_color
from pbd.constraints.base import ConstraintGroup
from pbd.mesh import Mesh


def _initial_dihedrals(V: np.ndarray, quads: np.ndarray) -> np.ndarray:
    if quads.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    p1 = V[quads[:, 0]]; p2 = V[quads[:, 1]]
    p3 = V[quads[:, 2]]; p4 = V[quads[:, 3]]
    e2 = p2 - p1; e3 = p3 - p1; e4 = p4 - p1
    c23 = np.cross(e2, e3); l23 = np.linalg.norm(c23, axis=1, keepdims=True)
    c24 = np.cross(e2, e4); l24 = np.linalg.norm(c24, axis=1, keepdims=True)
    n1 = c23 / np.maximum(l23, 1e-20)
    n2 = c24 / np.maximum(l24, 1e-20)
    d = np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0)
    return np.arccos(d)


def _project_dihedrals(
    P: np.ndarray,
    W: np.ndarray,
    quads: np.ndarray,
    rest: np.ndarray,
) -> np.ndarray:
    """Vectorized bend projection for a subset of dihedral 4-tuples."""
    out = np.zeros_like(P)
    if quads.shape[0] == 0:
        return out

    i1 = quads[:, 0]; i2 = quads[:, 1]
    i3 = quads[:, 2]; i4 = quads[:, 3]

    p1 = P[i1]; p2 = P[i2]; p3 = P[i3]; p4 = P[i4]
    e2 = p2 - p1; e3 = p3 - p1; e4 = p4 - p1

    c23 = np.cross(e2, e3); l23 = np.linalg.norm(c23, axis=1)
    c24 = np.cross(e2, e4); l24 = np.linalg.norm(c24, axis=1)

    eps = 1e-12
    valid = (l23 > eps) & (l24 > eps)
    inv23 = np.where(valid, 1.0 / np.maximum(l23, eps), 0.0)
    inv24 = np.where(valid, 1.0 / np.maximum(l24, eps), 0.0)

    n1 = c23 * inv23[:, None]
    n2 = c24 * inv24[:, None]

    d = np.sum(n1 * n2, axis=1)
    d_cl = np.clip(d, -1.0 + eps, 1.0 - eps)

    C = np.arccos(d_cl) - rest

    d_col = d_cl[:, None]
    q3 = (np.cross(e2, n2) + np.cross(n1, e2) * d_col) * inv23[:, None]
    q4 = (np.cross(e2, n1) + np.cross(n2, e2) * d_col) * inv24[:, None]
    q2 = -((np.cross(e3, n2) + np.cross(n1, e3) * d_col) * inv23[:, None]
           + (np.cross(e4, n1) + np.cross(n2, e4) * d_col) * inv24[:, None])
    q1 = -q2 - q3 - q4

    w1 = W[i1]; w2 = W[i2]; w3 = W[i3]; w4 = W[i4]
    denom = (w1 * np.sum(q1 * q1, axis=1)
             + w2 * np.sum(q2 * q2, axis=1)
             + w3 * np.sum(q3 * q3, axis=1)
             + w4 * np.sum(q4 * q4, axis=1))

    active = valid & (denom > eps)
    sin_factor = np.sqrt(np.maximum(0.0, 1.0 - d_cl * d_cl))
    s = np.zeros_like(C)
    s[active] = sin_factor[active] * C[active] / denom[active]

    np.add.at(out, i1, -(s * w1)[:, None] * q1)
    np.add.at(out, i2, -(s * w2)[:, None] * q2)
    np.add.at(out, i3, -(s * w3)[:, None] * q3)
    np.add.at(out, i4, -(s * w4)[:, None] * q4)
    return out


class Bend(ConstraintGroup):
    def __init__(self, quads: np.ndarray, rest_phi: np.ndarray, k: float = 1.0):
        quads = np.ascontiguousarray(quads, dtype=np.int64)
        rest_phi = np.ascontiguousarray(rest_phi, dtype=np.float64)
        if quads.ndim != 2 or quads.shape[1] != 4:
            raise ValueError(f"quads must be (M, 4); got {quads.shape}")
        if rest_phi.shape != (quads.shape[0],):
            raise ValueError(f"rest_phi shape {rest_phi.shape} != ({quads.shape[0]},)")
        self.idx = quads
        self.rest = rest_phi
        self.k = float(k)
        self._colors: np.ndarray | None = None  # lazy

    @classmethod
    def from_mesh(cls, mesh: Mesh, k: float = 1.0) -> "Bend":
        rest = _initial_dihedrals(mesh.V, mesh.bend_quads)
        return cls(mesh.bend_quads, rest, k=k)

    # ------------------------------------------------------------- Jacobi

    def project_batch(self, P: np.ndarray, W: np.ndarray) -> np.ndarray:
        return _project_dihedrals(P, W, self.idx, self.rest)

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
        return _project_dihedrals(P, W, self.idx[mask], self.rest[mask])
