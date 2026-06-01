"""Closed-mesh volume / pressure constraint — verification #14."""
from __future__ import annotations

import numpy as np

from pbd import System, Volume, build_mesh
from pbd.constraints.volume import _volume_sum


def _octahedron():
    V = np.array([
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0],
    ])
    F = np.array([
        [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],   # top hat (CCW from +z)
        [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3],   # bottom (CCW from -z)
    ], dtype=np.int64)
    return build_mesh(V, F)


def test_volume_zero_correction_at_rest():
    m = _octahedron()
    assert m.is_closed
    sys = System.from_mesh(m, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Volume.from_mesh(m, k_pressure=1.0, k=1.0))
    X_before = sys.X.copy()
    sys.step(dt=1e-3, iters=10)
    np.testing.assert_allclose(sys.X, X_before, atol=1e-9)


def test_volume_inflates_when_k_pressure_above_one():
    """k_pressure = 1.5, k=1 ⇒ volume drives toward 1.5 V0 within a few iters."""
    m = _octahedron()
    sys = System.from_mesh(m, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Volume.from_mesh(m, k_pressure=1.5, k=1.0))

    V0_sum = _volume_sum(sys.X, m.F)
    sys.step(dt=1e-3, iters=30)
    V_final = _volume_sum(sys.X, m.F)
    np.testing.assert_allclose(V_final, 1.5 * V0_sum, rtol=1e-3)


def test_open_mesh_rejected():
    import pytest

    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ])
    F = np.array([[0, 1, 2]], dtype=np.int64)
    m = build_mesh(V, F)
    assert not m.is_closed
    with pytest.raises(ValueError):
        Volume.from_mesh(m)


def test_volume_pinned_vertex_unchanged():
    m = _octahedron()
    sys = System.from_mesh(m, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Volume.from_mesh(m, k_pressure=1.5, k=1.0))
    sys.pin([0])
    x0 = sys.X[0].copy()
    sys.step(dt=1e-3, iters=20)
    np.testing.assert_array_equal(sys.X[0], x0)
