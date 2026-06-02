"""Static colliders (plane, sphere) + per-step collision constraint group.

Paper §6: collisions are encoded as inequality constraints C(p) ≥ 0 that
are *generated each step* (not static topology-derived like Stretch). At
the start of each step we check predicted positions against colliders;
penetrating particles get a one-shot constraint that pushes them onto the
surface (C = 0). After the step, the velocities of those particles are
post-processed with restitution and Coulomb friction (eq. 16).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbd.coloring import greedy_color
from pbd.constraints.base import ConstraintGroup


# ---------------------------------------------------------------- colliders


@dataclass
class Plane:
    """Half-space {x : n · x ≥ d}, with outward normal n (unit) and offset d.

    A particle at position p is inside the allowed region iff
    ``n · p - d ≥ 0``.
    """
    normal: np.ndarray
    offset: float

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=np.float64)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            raise ValueError("plane normal must be nonzero")
        self.normal = n / nn
        self.offset = float(self.offset)

    def signed_distance(self, P: np.ndarray) -> np.ndarray:
        """(N,) signed distance; ≥ 0 outside, < 0 inside the wall."""
        return P @ self.normal - self.offset

    def surface_normals(self, P: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.normal, P.shape).copy()


@dataclass
class Sphere:
    """Solid sphere obstacle: outside region {x : |x - c| ≥ r}."""
    center: np.ndarray
    radius: float

    def __post_init__(self):
        self.center = np.ascontiguousarray(self.center, dtype=np.float64)
        if self.center.shape != (3,):
            raise ValueError(f"sphere center must be (3,); got {self.center.shape}")
        if self.radius <= 0:
            raise ValueError("sphere radius must be > 0")
        self.radius = float(self.radius)

    def signed_distance(self, P: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(P - self.center, axis=1)
        return d - self.radius

    def surface_normals(self, P: np.ndarray) -> np.ndarray:
        d = P - self.center
        L = np.linalg.norm(d, axis=1, keepdims=True)
        # If a particle is exactly at the centre, fall back to +x; this is
        # degenerate but shouldn't NaN.
        ok = L > 1e-12
        n = np.zeros_like(d)
        n[ok[:, 0]] = d[ok[:, 0]] / L[ok[:, 0]]
        n[~ok[:, 0]] = np.array([1.0, 0.0, 0.0])
        return n


# ----------------------------------------------------- collision constraint


class CollisionGroup(ConstraintGroup):
    """One-shot inequality constraints generated at predict time.

    Each constraint pushes a single particle ``i`` onto a target half-space
    defined by an outward unit normal ``n_i`` and a target signed distance
    ``d_i``: enforce ``n_i · (p_i - p0_i) ≥ d_i`` where ``p0_i`` is the
    surface anchor (precomputed at generation time so the constraint
    becomes a simple plane projection during iteration).

    Stored fields:
        idx     : (M,) int64 particle indices
        normals : (M, 3) float64 unit outward normals
        offsets : (M,) float64 — the value ``offset_i = n_i · p0_i`` so
                  the constraint reads C_i(p) = n_i · p - offset_i.
    """

    def __init__(
        self,
        idx: np.ndarray,
        normals: np.ndarray,
        offsets: np.ndarray,
        k: float = 1.0,
    ):
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        normals = np.ascontiguousarray(normals, dtype=np.float64)
        offsets = np.ascontiguousarray(offsets, dtype=np.float64)
        if idx.ndim != 1:
            raise ValueError(f"idx must be (M,); got {idx.shape}")
        if normals.shape != (idx.shape[0], 3):
            raise ValueError(f"normals shape {normals.shape} != ({idx.shape[0]}, 3)")
        if offsets.shape != idx.shape:
            raise ValueError(f"offsets shape {offsets.shape} != {idx.shape}")
        self.idx = idx
        self.normals = normals
        self.offsets = offsets
        self.k = float(k)
        self._colors: np.ndarray | None = None  # lazy

    def _project(
        self,
        P: np.ndarray,
        W: np.ndarray,
        i: np.ndarray,
        n: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros_like(P)
        if i.shape[0] == 0:
            return out
        # C(p) = n · p - offset; only project when C < 0 (still penetrating).
        C = np.einsum("ij,ij->i", P[i], n) - offsets
        active = (C < 0.0) & (W[i] > 0.0)
        if not active.any():
            return out
        # Single-vertex constraint with |∇C| = 1: λ = C / (w·1) gives
        # Δp = -λ · w · n = -C · n. The mass cancels, so the projection
        # always lands exactly on the surface in one shot regardless of W.
        np.add.at(out, i[active], (-C[active])[:, None] * n[active])
        return out

    def project_batch(self, P: np.ndarray, W: np.ndarray) -> np.ndarray:
        return self._project(P, W, self.idx, self.normals, self.offsets)

    # ------------------------------------------------------- Gauss–Seidel

    def _ensure_colors(self) -> np.ndarray:
        if self._colors is None:
            # idx is (M,); reshape to (M, 1) so greedy_color can run.
            self._colors = greedy_color(self.idx.reshape(-1, 1))
        return self._colors

    @property
    def n_colors(self) -> int:
        c = self._ensure_colors()
        return int(c.max() + 1) if c.size else 1

    def project_color(self, P: np.ndarray, W: np.ndarray, c: int) -> np.ndarray:
        colors = self._ensure_colors()
        mask = colors == c
        return self._project(P, W, self.idx[mask], self.normals[mask], self.offsets[mask])


# ------------------------------------------- per-step constraint generation


def generate_collision_constraints(
    P: np.ndarray,
    colliders,
    free_mask: np.ndarray,
    k: float = 1.0,
    skin: float = 0.0,
) -> CollisionGroup:
    """Build a CollisionGroup from the predicted positions.

    Parameters
    ----------
    P : (N, 3) float64
        Predicted positions after the gravity/predict step.
    colliders : iterable of Plane / Sphere.
    free_mask : (N,) bool
        Only generate constraints for particles with W > 0; pinned ones
        cannot move and would just create degenerate constraints.
    skin : float
        Contact-detection tolerance. A vertex with ``sdf < skin`` (i.e.
        either inside the surface or within ``skin`` of it) gets a
        constraint generated. Constraints with ``sdf > 0`` are "armed
        but inactive" (the projection only fires when C < 0), giving
        contact hysteresis: a vertex that just grazes the surface keeps
        its constraint next step instead of toggling on/off as it drifts
        across the sdf=0 boundary. Recommended ~1e-3 for cloth on smooth
        obstacles.
    """
    idx_list, n_list, off_list = [], [], []
    for c in colliders:
        sdf = c.signed_distance(P)                    # (N,)
        hit = (sdf < skin) & free_mask                 # bool mask
        if not hit.any():
            continue
        Phit = P[hit]
        n = c.surface_normals(Phit)                    # (M_hit, 3) unit outward
        # Anchor each constraint at the *surface point* directly below the
        # particle (along the outward normal). Then n · p_anchor = offset.
        # For a plane: anchor is p - sdf · n; offset = n · p - sdf = c.offset.
        # For a sphere: anchor is centre + r · n; offset = n · (centre + r·n) = n·centre + r.
        # In either case the surface offset equals  n · p − sdf(p).
        offsets = np.einsum("ij,ij->i", Phit, n) - sdf[hit]
        idx_list.append(np.where(hit)[0])
        n_list.append(n)
        off_list.append(offsets)

    if not idx_list:
        return CollisionGroup(
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            k=k,
        )

    return CollisionGroup(
        np.concatenate(idx_list),
        np.concatenate(n_list, axis=0),
        np.concatenate(off_list),
        k=k,
    )
