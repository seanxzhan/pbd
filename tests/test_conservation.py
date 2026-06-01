"""Linear & angular momentum conservation — verification #5 and #6.

Paper §3.3 claims the constraint projection step preserves both linear
momentum (when constraints are translation-invariant) and angular momentum
(when constraints are rotation-invariant). Distance, bend, and volume
constraints are all rotation- and translation-invariant. Stepping a free-
floating mesh under purely-internal constraints should leave the centre of
mass moving at constant velocity and the total angular momentum unchanged.
"""
from __future__ import annotations

import numpy as np

from pbd import Bend, Stretch, System, build_mesh


def _floating_mesh():
    """A 4-vertex non-degenerate tetrahedron — closed, has interior edges
    for bend, and arbitrary geometry to avoid lucky symmetric cancellation.
    """
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.3, 0.0, 0.0],
        [0.4, 1.1, 0.0],
        [0.5, 0.4, 0.9],
    ])
    F = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ], dtype=np.int64)
    return build_mesh(V, F)


def _cm(X, masses):
    return (X * masses[:, None]).sum(axis=0) / masses.sum()


def _linear_momentum(V, masses):
    return (V * masses[:, None]).sum(axis=0)


def _angular_momentum(X, V, masses, origin):
    r = X - origin
    p = V * masses[:, None]
    return np.cross(r, p).sum(axis=0)


def test_linear_momentum_conserved_under_internal_constraints():
    """Bulk velocity, no gravity, no pinning ⇒ CoM advances at constant v."""
    m = _floating_mesh()
    sys = System.from_mesh(m, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Stretch.from_mesh(m, k=1.0))
    sys.add_constraint(Bend.from_mesh(m, k=1.0))

    # Inject a bulk translation + a small internal perturbation so the
    # constraints actually have to do work (testing the projection, not
    # an idle no-op).
    v_bulk = np.array([0.7, -0.4, 0.2])
    sys.V[:] = v_bulk
    sys.X[3] += np.array([0.0, 0.0, 0.05])  # break rest, force projection to act

    masses = sys.masses
    p_before = _linear_momentum(sys.V, masses)
    cm0 = _cm(sys.X, masses)

    dt = 1e-3
    n_steps = 50
    for _ in range(n_steps):
        sys.step(dt=dt, iters=10)

    p_after = _linear_momentum(sys.V, masses)
    cm1 = _cm(sys.X, masses)

    # CoM should advance linearly at (initial momentum) / total mass.
    expected_cm = cm0 + (p_before / masses.sum()) * (n_steps * dt)
    np.testing.assert_allclose(cm1, expected_cm, atol=1e-9)
    # Total linear momentum unchanged (no external forces).
    np.testing.assert_allclose(p_after, p_before, atol=1e-9)


def test_angular_momentum_conserved_under_internal_constraints():
    """Rigid rotation about CoM, no gravity ⇒ angular momentum preserved.

    Constraint projections of rotation-invariant constraints (distance,
    bend) cannot torque the system: Σ p × m·Δp = 0 analytically. The only
    drift PBD introduces is at the velocity-recovery step, where V_new =
    (P-X)/dt mixes pre- and post-projection positions; that drift is O(dt²)
    per step when the configuration starts at rest. We use rigid rotation
    initial conditions with no extra perturbation so the constraints are
    only mildly violated by the linear predict-step approximation to the
    curved rigid-rotation trajectory.
    """
    m = _floating_mesh()
    sys = System.from_mesh(m, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Stretch.from_mesh(m, k=1.0))
    sys.add_constraint(Bend.from_mesh(m, k=1.0))

    masses = sys.masses
    cm0 = _cm(sys.X, masses)

    # Rigid rotation field v_i = ω × r_i about the CoM.
    omega = np.array([0.3, 0.5, -0.2])
    r0 = sys.X - cm0
    sys.V[:] = np.cross(np.broadcast_to(omega, r0.shape), r0)

    # Measure L about a *fixed* origin so we don't fold CoM motion into the
    # comparison. Linear momentum is zero by construction (rotation about
    # CoM), so CoM should not drift either.
    origin = np.zeros(3)
    L_before = _angular_momentum(sys.X, sys.V, masses, origin)

    dt = 1e-3
    n_steps = 50
    for _ in range(n_steps):
        sys.step(dt=dt, iters=10)

    L_after = _angular_momentum(sys.X, sys.V, masses, origin)

    np.testing.assert_allclose(L_after, L_before, atol=5e-5)
