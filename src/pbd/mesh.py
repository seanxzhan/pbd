"""Triangle mesh with topology extraction and validation.

Topology pieces we need downstream:
  - unique undirected edges (E, 2)         → stretch constraints
  - dihedral 4-tuples (B, 4)               → bend constraints
  - boundary mask                          → skip bend on boundary edges
  - is_closed flag                         → gate volume constraint
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class NonManifoldError(ValueError):
    """Raised when a mesh has an edge shared by more than two faces."""


@dataclass
class Mesh:
    V: np.ndarray            # (N, 3) float64 — vertex positions
    F: np.ndarray            # (M, 3) int64 — triangle vertex indices

    edges: np.ndarray        # (E, 2) int64 — unique undirected edges, sorted
    bend_quads: np.ndarray   # (B, 4) int64 — (p1, p2, p3, p4) per interior edge
    boundary_mask: np.ndarray  # (E,) bool — true on boundary (1-incident) edges

    @property
    def n_verts(self) -> int:
        return int(self.V.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.F.shape[0])

    @property
    def is_closed(self) -> bool:
        return not bool(self.boundary_mask.any())

    def face_areas(self) -> np.ndarray:
        v0 = self.V[self.F[:, 0]]
        v1 = self.V[self.F[:, 1]]
        v2 = self.V[self.F[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    def vertex_masses(self, density: float) -> np.ndarray:
        """Lump 1/3 of each adjacent face's mass to each vertex (paper §4.1).

        ``density`` is mass per unit area (kg/m²).
        """
        face_mass = self.face_areas() * density
        per_corner = np.repeat(face_mass / 3.0, 3)
        m = np.zeros(self.n_verts, dtype=np.float64)
        np.add.at(m, self.F.ravel(), per_corner)
        return m


def build_mesh(V: np.ndarray, F: np.ndarray) -> Mesh:
    """Construct a Mesh from vertex / face arrays, extracting topology.

    Raises
    ------
    NonManifoldError
        If any edge is shared by more than two faces.
    """
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V must be (N, 3); got {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F must be (M, 3); got {F.shape}")

    raw = np.sort(F[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    edges, inv = np.unique(raw, axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=edges.shape[0])

    if (counts > 2).any():
        bad = np.where(counts > 2)[0]
        raise NonManifoldError(
            f"Non-manifold mesh: {bad.size} edge(s) shared by >2 faces "
            f"(first: vertices {edges[bad[0]].tolist()}, "
            f"shared by {int(counts[bad[0]])} faces)"
        )

    boundary_mask = counts == 1
    interior_mask = counts == 2
    interior_ids = np.where(interior_mask)[0]

    bend_quads = _extract_bend_quads(F, edges, inv, counts, interior_ids)

    return Mesh(
        V=V,
        F=F,
        edges=edges,
        bend_quads=bend_quads,
        boundary_mask=boundary_mask,
    )


def _extract_bend_quads(
    F: np.ndarray,
    edges: np.ndarray,
    inv: np.ndarray,
    counts: np.ndarray,
    interior_ids: np.ndarray,
) -> np.ndarray:
    """For each interior edge, return (p1, p2, p3, p4):
    p1, p2 are the edge endpoints; p3 and p4 are the third vertices of the
    two adjacent faces. Vectorized — no Python loop over edges.
    """
    if interior_ids.size == 0:
        return np.empty((0, 4), dtype=np.int64)

    # Group corner positions by edge id. inv has length 3*M; sorting it
    # places all corners belonging to the same edge consecutively.
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))  # (E,)
    face_per_corner = np.repeat(np.arange(F.shape[0], dtype=np.int64), 3)

    s = starts[interior_ids]                      # (B,)
    c0 = order[s]
    c1 = order[s + 1]
    f0 = face_per_corner[c0]
    f1 = face_per_corner[c1]

    a = edges[interior_ids, 0]                    # (B,)
    b = edges[interior_ids, 1]
    tri0 = F[f0]                                  # (B, 3)
    tri1 = F[f1]
    # Each row has exactly one entry not equal to a or b → that's the third vertex.
    mask0 = (tri0 != a[:, None]) & (tri0 != b[:, None])
    mask1 = (tri1 != a[:, None]) & (tri1 != b[:, None])
    p3 = tri0[mask0]
    p4 = tri1[mask1]

    return np.column_stack([a, b, p3, p4]).astype(np.int64, copy=False)
