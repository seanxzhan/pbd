"""Closed-mesh volume / pressure constraint, paper §4.4 (eqs. 14, 15).

Single equality constraint over all N vertices:

    C(p_1, ..., p_N) = Σ_t (p_{t1} × p_{t2}) · p_{t3}  -  k_pressure · V_0

Per-vertex gradient (eq. 15) is the sum, over each triangle the vertex
belongs to, of a per-corner cross product.
"""
from __future__ import annotations

import numpy as np

from pbd.constraints.base import ConstraintGroup
from pbd.mesh import Mesh


def _volume_sum(V: np.ndarray, F: np.ndarray) -> float:
    p1 = V[F[:, 0]]; p2 = V[F[:, 1]]; p3 = V[F[:, 2]]
    return float(np.sum(np.cross(p1, p2) * p3))


class Volume(ConstraintGroup):
    def __init__(
        self,
        F: np.ndarray,
        V0: float,
        k_pressure: float = 1.0,
        k: float = 1.0,
    ):
        self.F = np.ascontiguousarray(F, dtype=np.int64)
        self.V0 = float(V0)
        self.k_pressure = float(k_pressure)
        self.k = float(k)

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
        k_pressure: float = 1.0,
        k: float = 1.0,
    ) -> "Volume":
        if not mesh.is_closed:
            raise ValueError("Volume constraint requires a closed mesh")
        V0 = _volume_sum(mesh.V, mesh.F)
        return cls(mesh.F, V0, k_pressure=k_pressure, k=k)

    def project_batch(self, P: np.ndarray, W: np.ndarray) -> np.ndarray:
        F = self.F
        p1 = P[F[:, 0]]; p2 = P[F[:, 1]]; p3 = P[F[:, 2]]

        # C = Σ triple-products − k_pressure · V_0
        C = float(np.sum(np.cross(p1, p2) * p3)) - self.k_pressure * self.V0

        # Per-corner gradient contribution (eq. 15)
        g1 = np.cross(p2, p3)
        g2 = np.cross(p3, p1)
        g3 = np.cross(p1, p2)
        grad = np.zeros_like(P)
        np.add.at(grad, F[:, 0], g1)
        np.add.at(grad, F[:, 1], g2)
        np.add.at(grad, F[:, 2], g3)

        denom = float(np.sum(W * np.sum(grad * grad, axis=1)))
        if denom < 1e-20 or abs(C) < 1e-20:
            return np.zeros_like(P)

        s = C / denom
        return -s * W[:, None] * grad
