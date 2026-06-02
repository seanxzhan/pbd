"""Minimal OBJ loader. Numpy only, no trimesh dep."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an OBJ file as (V, F).

    Returns
    -------
    V : (N, 3) float64
    F : (M, 3) int64

    Only `v` and `f` lines are parsed. Polygons with >3 vertices are
    triangulated as a fan from the first vertex. Texture and normal
    references on `f` lines (e.g. ``1/2/3``) are ignored.
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open() as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            head = tok[0]
            if head == "v":
                verts.append([float(x) for x in tok[1:4]])
            elif head == "f":
                idx = [int(t.split("/", 1)[0]) - 1 for t in tok[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return (
        np.asarray(verts, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
    )


def fix_winding(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Re-orient face winding so it is consistent and globally outward.

    Two passes:

    1. **Consistency.** Walk the face-adjacency graph (edges → incident
       faces). When two adjacent faces traverse their shared edge in
       the SAME direction, one of them is flipped. After this pass
       every pair of adjacent faces traverses any shared edge in
       opposite directions — the standard manifold-orientability
       condition.

    2. **Global outward orientation.** Compute the signed volume of
       the surface via the divergence-theorem identity ``V_signed =
       (1/6) Σ V0 · (V1 × V2)``. If negative, flip every face. This
       step assumes a closed mesh; on an open mesh the signed volume
       depends on the choice of origin, and step 2 may give a wrong
       global flip.

    Returns a new ``(M, 3)`` face array; ``V`` is unchanged.

    Notes
    -----
    * Disconnected components: each component is made internally
      consistent. Step 2 acts on the whole mesh — if components have
      opposite orientations, the global flip resolves only the
      dominant one.
    * Non-manifold edges (>2 incident faces): each manifold patch is
      handled correctly; the answer near the non-manifold edge is
      undefined but the function does not raise.
    """
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V must be (Vn, 3); got {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F must be (Fn, 3); got {F.shape}")

    n_faces = F.shape[0]
    if n_faces == 0:
        return F.copy()

    # Each face has 3 directed edges (i->j) at positions 0, 1, 2.
    raw = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)  # (Fn, 3, 2)
    # +1 if the directed edge runs from low to high vertex id ("forward"
    # in canonical undirected-edge order), -1 otherwise.
    direction = np.where(raw[:, :, 0] < raw[:, :, 1], 1, -1).astype(np.int8)
    # Undirected edge id per (face, position).
    sorted_edges = np.sort(raw, axis=2)
    _, inv = np.unique(sorted_edges.reshape(-1, 2), axis=0, return_inverse=True)
    inv = inv.reshape(n_faces, 3)

    n_edges = int(inv.max()) + 1
    edge_faces: list[list[tuple[int, int]]] = [[] for _ in range(n_edges)]
    for f in range(n_faces):
        for k in range(3):
            edge_faces[int(inv[f, k])].append((f, k))

    # sign[f]: +1 = wind same as seed, -1 = wind flipped from seed,
    # 0 = unvisited. The BFS propagates the consistency constraint
    # "two adjacent faces traverse the shared edge in opposite
    # directions" — i.e. their signed directions sum to 0.
    sign = np.zeros(n_faces, dtype=np.int8)

    for seed in range(n_faces):
        if sign[seed] != 0:
            continue
        sign[seed] = 1
        stack = [seed]
        while stack:
            f = stack.pop()
            for k in range(3):
                edge_id = int(inv[f, k])
                f_dir = int(direction[f, k]) * int(sign[f])
                for nf, nk in edge_faces[edge_id]:
                    if nf == f or sign[nf] != 0:
                        continue
                    nf_dir_unsigned = int(direction[nf, nk])
                    # If the unsigned direction matches our signed
                    # direction, nf is wound the same way as f on this
                    # edge — that's an inconsistency, so flip nf.
                    sign[nf] = -1 if f_dir == nf_dir_unsigned else 1
                    stack.append(nf)

    F_out = F.copy()
    flip_mask = sign == -1
    if flip_mask.any():
        F_out[flip_mask] = F_out[flip_mask][:, [0, 2, 1]]

    # Step 2: outward orientation via signed volume.
    V0 = V[F_out[:, 0]]
    V1 = V[F_out[:, 1]]
    V2 = V[F_out[:, 2]]
    signed_vol = float(np.einsum("fi,fi->f", V0, np.cross(V1, V2)).sum() / 6.0)
    if signed_vol < 0.0:
        F_out = F_out[:, [0, 2, 1]]

    return F_out


def convex_hull(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the convex hull of a 3-D point cloud, as ``(V_hull, F_hull)``.

    The hull is a closed, manifold, outward-wound triangle mesh — exactly
    the shape ``TriangleMesh`` expects. Use this to derive a collision
    proxy for a non-convex obstacle whose own surface would mislead the
    closest-point fallback (see ``TriangleMesh`` docstring).

    Implementation note
    -------------------
    Delegates to pymeshlab's ``generate_convex_hull`` filter (Quickhull
    under the hood). The hull's face winding is then run through
    ``fix_winding`` so we don't depend on pymeshlab's orientation choice.

    Parameters
    ----------
    V : (N, 3) array-like
        Input point cloud. Hull is computed over the entire set; isolated
        interior points are ignored automatically.

    Returns
    -------
    V_hull : (Nh, 3) float64
    F_hull : (Mh, 3) int64
    """
    import pymeshlab as pml

    V = np.ascontiguousarray(V, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V must be (N, 3); got {V.shape}")
    if V.shape[0] < 4:
        raise ValueError(f"convex hull needs >= 4 points; got {V.shape[0]}")

    ms = pml.MeshSet()
    ms.add_mesh(pml.Mesh(V))
    ms.generate_convex_hull()
    hull = ms.current_mesh()
    V_hull = np.ascontiguousarray(hull.vertex_matrix(), dtype=np.float64)
    F_hull = np.ascontiguousarray(hull.face_matrix(), dtype=np.int64)
    F_hull = fix_winding(V_hull, F_hull)
    return V_hull, F_hull
