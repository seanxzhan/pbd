"""Mesh topology + OBJ I/O sanity tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pbd import build_mesh, load_obj, NonManifoldError


# ---------------------------------------------------------------- helpers

def _two_tri_strip():
    """Two triangles sharing edge (1, 2):
        2---3
        |\\ |
        | \\|
        0---1
    Triangles: (0,1,2) and (1,3,2). Shared edge: (1, 2).
    """
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    F = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    return V, F


def _tetrahedron():
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    F = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ], dtype=np.int64)
    return V, F


# ---------------------------------------------------------------- tests

def test_two_triangle_strip_topology():
    V, F = _two_tri_strip()
    m = build_mesh(V, F)

    assert m.n_verts == 4
    assert m.n_faces == 2
    # 5 unique edges: (0,1), (0,2), (1,2), (1,3), (2,3)
    assert m.edges.shape == (5, 2)
    expected_edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    np.testing.assert_array_equal(m.edges, expected_edges)

    # Boundary: every edge except (1,2)
    assert m.boundary_mask.sum() == 4
    assert not m.boundary_mask[2]  # the (1,2) edge
    assert not m.is_closed

    # One bend quad on edge (1,2): third vertex of face 0 is 0, of face 1 is 3
    assert m.bend_quads.shape == (1, 4)
    a, b, p3, p4 = m.bend_quads[0]
    assert (a, b) == (1, 2)
    assert {p3, p4} == {0, 3}


def test_tetrahedron_is_closed():
    V, F = _tetrahedron()
    m = build_mesh(V, F)
    assert m.is_closed
    assert m.boundary_mask.sum() == 0
    # Tetrahedron has 6 edges and 4 dihedral pairs (one per edge — every edge is interior).
    assert m.edges.shape == (6, 2)
    assert m.bend_quads.shape == (6, 4)


def test_non_manifold_raises():
    V = np.zeros((5, 3))
    # Three triangles all sharing edge (0, 1) — non-manifold.
    F = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    with pytest.raises(NonManifoldError):
        build_mesh(V, F)


def test_vertex_masses_sum_to_total():
    V, F = _two_tri_strip()
    m = build_mesh(V, F)
    density = 2.5
    vm = m.vertex_masses(density)
    total_area = m.face_areas().sum()
    assert vm.shape == (4,)
    np.testing.assert_allclose(vm.sum(), total_area * density)


def test_obj_roundtrip():
    V, F = _two_tri_strip()
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 2 4 3\n"
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
        fh.write(obj)
        path = Path(fh.name)
    try:
        V2, F2 = load_obj(path)
        np.testing.assert_allclose(V2, V)
        np.testing.assert_array_equal(F2, F)
    finally:
        path.unlink()


def test_obj_with_texture_normals_skipped():
    """f line with `v/vt/vn` notation: only the vertex index is read."""
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/1 3/3/1\n"
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
        fh.write(obj)
        path = Path(fh.name)
    try:
        V, F = load_obj(path)
        np.testing.assert_array_equal(F, [[0, 1, 2]])
    finally:
        path.unlink()


if __name__ == "__main__":
    test_two_triangle_strip_topology()
    test_tetrahedron_is_closed()
    test_non_manifold_raises()
    test_vertex_masses_sum_to_total()
    test_obj_roundtrip()
    test_obj_with_texture_normals_skipped()
    print("Phase A: mesh tests passed.")
