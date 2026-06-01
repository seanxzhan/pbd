"""Greedy graph coloring of constraint groups.

Two constraints "conflict" iff they share a vertex. A coloring assigns each
constraint a color such that constraints in the same color class are
vertex-disjoint — within a color class the projection batch can be applied
without races, and across color classes the previous color's correction is
visible to the next, giving true Gauss-Seidel information propagation.

For typical cloth meshes a greedy first-fit coloring uses 5–7 colors.
"""
from __future__ import annotations

import numpy as np


def greedy_color(idx: np.ndarray) -> np.ndarray:
    """Greedy first-fit coloring of constraints.

    Parameters
    ----------
    idx : (M, n) int array
        Vertex indices of each constraint (n = 2 for stretch edges, 4 for
        bend dihedrals, 1 for collisions). For 1-vertex constraints pass
        ``idx.reshape(-1, 1)``.

    Returns
    -------
    colors : (M,) int32
        Color label per constraint. ``colors.max() + 1`` is the number of
        color classes; ``colors == c`` selects all constraints of color c.
    """
    if idx.ndim != 2:
        raise ValueError(f"idx must be 2-D (M, n); got shape {idx.shape}")
    M, n = idx.shape
    if M == 0:
        return np.zeros(0, dtype=np.int32)

    colors = np.full(M, -1, dtype=np.int32)
    # vert_colors[v] = set of colors used by constraints touching vertex v
    vert_colors: dict[int, set[int]] = {}

    for ci in range(M):
        used: set[int] = set()
        for k in range(n):
            v = int(idx[ci, k])
            cs = vert_colors.get(v)
            if cs is not None:
                used.update(cs)
        # Lowest color not in `used`.
        c = 0
        while c in used:
            c += 1
        colors[ci] = c
        for k in range(n):
            v = int(idx[ci, k])
            s = vert_colors.get(v)
            if s is None:
                vert_colors[v] = {c}
            else:
                s.add(c)
    return colors
