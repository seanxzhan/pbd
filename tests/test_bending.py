"""Dihedral bend constraint behaviour — verification #12 and corollaries."""
from __future__ import annotations

import numpy as np

from pbd import Bend, System, build_mesh, Stretch


def _flat_strip():
    """Two coplanar triangles sharing edge (0, 1):
            2
           /|
          / |
         0--1
          \\ |
           \\|
            3
    """
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ])
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
    return build_mesh(V, F)


def test_bend_zero_at_rest():
    """Mesh starts at its rest dihedral ⇒ zero correction."""
    m = _flat_strip()
    sys = System.from_mesh(m, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Bend.from_mesh(m, k=1.0))
    X_before = sys.X.copy()
    sys.step(dt=1e-3, iters=10)
    np.testing.assert_allclose(sys.X, X_before, atol=1e-12)


def test_bend_pulls_perturbation_back():
    """Pin three corners, lift the fourth out of plane → bend pulls it back."""
    m = _flat_strip()
    sys = System.from_mesh(m, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Bend.from_mesh(m, k=1.0))
    sys.pin([0, 1, 3])
    sys.X[2, 2] = 0.1
    z0 = sys.X[2, 2]
    sys.step(dt=1e-3, iters=30)
    assert abs(sys.X[2, 2]) < z0 * 0.3, f"z stayed at {sys.X[2, 2]:.4f} (started {z0})"


def test_bend_invariant_under_rigid_translation():
    """A rigid translation of the whole mesh must not produce a correction."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.5],  # already bent — rest is preserved
    ])
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
    m = build_mesh(V, F)

    sys = System.from_mesh(m, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Bend.from_mesh(m, k=1.0))
    sys.X[:] += np.array([5.0, 7.0, -3.0])
    X_before = sys.X.copy()
    sys.step(dt=1e-3, iters=10)
    np.testing.assert_allclose(sys.X, X_before, atol=1e-9)


def test_bend_independent_of_edge_length():
    """The paper's headline claim (Figure 3): stretching the mesh does NOT
    change the dihedral angle. We stretch the strip along the shared edge
    (with no bend constraint active) and verify the dihedral angle is
    unchanged.
    """
    m = _flat_strip()

    # Hand-compute the dihedral under the paper's convention before stretching.
    rest_before = Bend.from_mesh(m, k=1.0).rest[0]

    # Now stretch the mesh along x by a large factor.
    V_stretched = m.V.copy()
    V_stretched[:, 0] *= 3.0
    m2 = build_mesh(V_stretched, m.F)
    rest_after = Bend.from_mesh(m2, k=1.0).rest[0]

    assert abs(rest_before - rest_after) < 1e-12, (
        f"dihedral should be edge-length invariant: {rest_before} → {rest_after}"
    )


def test_bend_conserves_linear_momentum_when_active():
    """All four vertices unpinned, displace one out of plane → bend correction
    must not shift the centre of mass (paper §3.3 momentum-preservation claim).
    """
    m = _flat_strip()
    sys = System.from_mesh(m, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Bend.from_mesh(m, k=1.0))
    sys.X[2, 2] = 0.2

    masses = sys.masses
    cm_before = (sys.X * masses[:, None]).sum(axis=0) / masses.sum()
    sys.step(dt=1e-3, iters=20)
    cm_after = (sys.X * masses[:, None]).sum(axis=0) / masses.sum()
    # Initial velocity from a single step is small; the constraint projection
    # itself should not shift CM. Allow modest tolerance for the integrated
    # gravity-free trajectory (no gravity here ⇒ V stays zero).
    np.testing.assert_allclose(cm_after, cm_before, atol=1e-10)
