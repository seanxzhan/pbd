"""§3.5 rigid-mode-preserving damping.

Paper §3.5: damp the per-particle deviation from the global rigid-body
velocity field. With k_damp = 1, all internal motion dies in one step;
with 0 ≤ k_damp ≤ 1, internal motion is attenuated by (1 - k_damp) per
step. Bulk translation and rotation must survive *any* damping value.
"""
from __future__ import annotations

import numpy as np

from pbd import System, Stretch
from pbd.solver import damp_velocities


def test_damping_preserves_uniform_translation():
    """Pure rigid translation: V identical for all particles. Damping must
    not change V regardless of k_damp."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.7],
    ])
    V = np.tile(np.array([0.3, -0.2, 0.7]), (4, 1))
    W = np.ones(4)

    V_before = V.copy()
    damp_velocities(X, V, W, k_damp=1.0)
    np.testing.assert_allclose(V, V_before, atol=1e-12)


def test_damping_preserves_rigid_rotation():
    """Pure rigid rotation about origin: V_i = ω × X_i. Damping must
    leave this unchanged."""
    X = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7, 0.4, -0.3],
    ])
    omega = np.array([0.1, 0.5, -0.2])
    V = np.cross(np.broadcast_to(omega, X.shape), X)
    W = np.ones(4)

    V_before = V.copy()
    damp_velocities(X, V, W, k_damp=1.0)
    np.testing.assert_allclose(V, V_before, atol=1e-12)


def test_damping_kills_internal_motion_at_k1():
    """Pure non-rigid motion (zero linear & angular momentum, nonzero V).
    With k_damp=1, V must be zeroed in one call."""
    X = np.array([
        [-1.0, 0.0, 0.0],
        [+1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, +1.0, 0.0],
    ])
    # Symmetric expansion: each particle moves outward — no net p, no net L.
    V = X.copy()
    W = np.ones(4)

    damp_velocities(X, V, W, k_damp=1.0)
    np.testing.assert_allclose(V, 0.0, atol=1e-12)


def test_damping_partial_attenuation():
    """k_damp=0.5 ⇒ deviation from rigid field halves per call."""
    X = np.array([
        [-1.0, 0.0, 0.0],
        [+1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, +1.0, 0.0],
    ])
    V0 = X.copy()  # purely internal
    W = np.ones(4)

    V = V0.copy()
    damp_velocities(X, V, W, k_damp=0.5)
    np.testing.assert_allclose(V, 0.5 * V0, atol=1e-12)


def test_damping_ignores_pinned_particles():
    """A pinned vertex (W=0) must not influence the rigid-body fit and
    must not have its V touched (it's already zero, but check no NaN)."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.4, 0.9],
    ])
    V = np.array([
        [0.0, 0.0, 0.0],   # pinned
        [0.5, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
    ])
    W = np.array([0.0, 1.0, 1.0, 1.0])

    V_before = V.copy()
    damp_velocities(X, V, W, k_damp=1.0)
    # Pinned vertex untouched.
    np.testing.assert_array_equal(V[0], V_before[0])
    # Free particles damped to a rigid-body fit (residual deviation = 0).
    Vf = V[1:]
    Xf = X[1:]
    cm = Xf.mean(axis=0)
    vcm = Vf.mean(axis=0)
    r = Xf - cm
    L = np.sum(np.cross(r, Vf), axis=0)
    rsq = np.sum(r * r, axis=1).sum()
    I = np.eye(3) * rsq - np.einsum("ij,ik->jk", r, r)
    omega = np.linalg.solve(I, L)
    rigid = vcm + np.cross(np.broadcast_to(omega, r.shape), r)
    np.testing.assert_allclose(Vf, rigid, atol=1e-12)


def test_damping_in_step_does_not_grow_velocity():
    """Integration test: a stretched two-particle system damped during step
    cannot have its CM velocity changed by damping (should be 0 → 0)."""
    X = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    sys = System(X, np.ones(2), gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), k=1.0))

    masses = sys.masses
    cm0 = (sys.X * masses[:, None]).sum(axis=0) / masses.sum()

    for _ in range(50):
        sys.step(dt=1e-3, iters=10, k_damp=0.5)

    cm1 = (sys.X * masses[:, None]).sum(axis=0) / masses.sum()
    # Initial linear momentum is zero (V=0 at start), so CM should not move.
    np.testing.assert_allclose(cm1, cm0, atol=1e-9)
