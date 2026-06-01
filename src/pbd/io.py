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
