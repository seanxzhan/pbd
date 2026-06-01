"""Verification #1: free-fall trajectory.

Two complementary checks:
  * PBD-without-constraints == symplectic Euler (bit-exact).
  * Symplectic Euler approximates analytic ballistic motion within
    its known O(dt) global truncation error.
Plus a pinning sanity check.
"""
from __future__ import annotations

import numpy as np

from pbd import System


# ----------------------------------------------------------------- helpers

GRAVITY = np.array([0.0, -9.81, 0.0])


def _symplectic_euler(x0: np.ndarray, v0: np.ndarray, g: np.ndarray, dt: float, n: int):
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    for _ in range(n):
        v += dt * g
        x += dt * v
    return x, v


# ---------------------------------------------------------------- tests

def test_freefall_matches_symplectic_euler():
    """PBD with no constraints reproduces symplectic Euler exactly."""
    X0 = np.array([[0.0, 10.0, 0.0]])
    V0 = np.array([[1.0, 2.0, -0.5]])
    masses = np.array([1.0])
    sys = System(X0, masses, gravity=tuple(GRAVITY))
    sys.V[:] = V0

    dt, n_steps = 1e-3, 2000
    for _ in range(n_steps):
        sys.step(dt)

    x_ref, v_ref = _symplectic_euler(X0[0], V0[0], GRAVITY, dt, n_steps)
    np.testing.assert_allclose(sys.X[0], x_ref, atol=1e-12)
    np.testing.assert_allclose(sys.V[0], v_ref, atol=1e-12)


def test_freefall_approaches_analytic():
    """O(dt) global error vs ½gt² closed form."""
    X0 = np.array([[0.0, 0.0, 0.0]])
    masses = np.array([1.0])
    sys = System(X0, masses, gravity=tuple(GRAVITY))

    dt, n_steps = 1e-4, 5000
    for _ in range(n_steps):
        sys.step(dt)

    t = n_steps * dt
    analytic = X0[0] + 0.5 * GRAVITY * t**2
    # Symplectic Euler global error after time t is dt * t * |g| / 2.
    bound = dt * t * np.linalg.norm(GRAVITY)
    err = np.linalg.norm(sys.X[0] - analytic)
    assert err < bound, f"err={err:.3e} > bound={bound:.3e}"


def test_pinned_particle_never_moves():
    X0 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    masses = np.array([1.0, 1.0])
    sys = System(X0, masses, gravity=tuple(GRAVITY))
    sys.pin([0])

    for _ in range(500):
        sys.step(1e-2)

    np.testing.assert_array_equal(sys.X[0], X0[0])
    np.testing.assert_array_equal(sys.V[0], np.zeros(3))
    # The free particle should have moved.
    assert sys.X[1, 1] < X0[1, 1] - 1.0


if __name__ == "__main__":
    test_freefall_matches_symplectic_euler()
    test_freefall_approaches_analytic()
    test_pinned_particle_never_moves()
    print("Phase A: freefall tests passed.")
