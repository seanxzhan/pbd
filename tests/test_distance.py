"""Stretch (distance) constraint behaviour — verification tests #2, #7, #8."""
from __future__ import annotations

import numpy as np

from pbd import Stretch, System


def _two_particle_system(distance: float, masses=(1.0, 1.0), gravity=(0.0, 0.0, 0.0)):
    X = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    s = System(X, np.array(masses), gravity=gravity)
    return s


# ----------------------- (test #2) two-particle equilibrium ------------

def test_residual_zero_at_rest():
    """Particles already at rest length: |C| stays at 0 after a step."""
    sys = _two_particle_system(distance=1.0)
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), k=1.0))
    sys.step(dt=1e-2, iters=1)
    L = np.linalg.norm(sys.X[0] - sys.X[1])
    assert abs(L - 1.0) < 1e-12


def test_one_iter_full_stiffness_kills_initial_gap():
    """k=1, one iter ⇒ stretched pair pulled to rest length."""
    sys = _two_particle_system(distance=1.5)
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), k=1.0))
    sys.step(dt=1e-3, iters=1)
    L = np.linalg.norm(sys.X[0] - sys.X[1])
    assert abs(L - 1.0) < 1e-9, f"length={L}, expected 1.0"


def test_pinned_endpoint_determines_position():
    """Particle 0 pinned at origin; rest length 1 ⇒ free particle settles to |x| = 1."""
    sys = _two_particle_system(distance=1.5)
    sys.pin([0])
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), k=1.0))
    sys.step(dt=1e-3, iters=1)
    np.testing.assert_array_equal(sys.X[0], [0.0, 0.0, 0.0])
    L = np.linalg.norm(sys.X[1])
    assert abs(L - 1.0) < 1e-9


# --------------- (test #7) residual decreases with iterations ----------

def test_residual_monotone_in_iters():
    """At a fixed sub-1 stiffness, residual after n_s iters drops as n_s grows
    when the k' linearization is *not* applied — i.e. raw stiffness k.
    Here we use k=1 with rest=1.0, displacement=2.0, and verify that more
    iters never increases the residual.
    """
    edges = np.array([[0, 1]])
    rest = np.array([1.0])
    residuals = []
    for n_iter in (1, 4, 16, 64):
        sys = _two_particle_system(distance=2.0)
        sys.add_constraint(Stretch(edges, rest, k=0.5))
        sys.step(dt=1e-3, iters=n_iter)
        residuals.append(abs(np.linalg.norm(sys.X[0] - sys.X[1]) - 1.0))
    # Monotone non-increasing
    for a, b in zip(residuals, residuals[1:]):
        assert b <= a + 1e-12, f"residual rose: {residuals}"


# --------------- (test #8) k' linearization is iter-independent --------

def test_k_prime_makes_residual_iter_independent():
    """With the §3.3 linearization, residual after n_s iters → (1-k)·gap0
    independent of n_s. Tolerance reflects float roundoff in (1-k)^(1/n_s).
    """
    edges = np.array([[0, 1]])
    rest = np.array([1.0])
    k = 0.4
    gap0 = 2.0  # initial |C| = |2.0 - 1.0| = 1.0 displacement
    expected = (1.0 - k) * (gap0 - 1.0)

    residuals = []
    for n_iter in (1, 5, 25, 100):
        sys = _two_particle_system(distance=gap0)
        sys.add_constraint(Stretch(edges, rest, k=k))
        sys.step(dt=1e-3, iters=n_iter)
        residuals.append(abs(np.linalg.norm(sys.X[0] - sys.X[1]) - 1.0))

    for r in residuals:
        assert abs(r - expected) < 1e-9, (
            f"got residuals {residuals}, expected ~{expected}"
        )


# --------------- pinning correctness when both endpoints pinned --------

def test_both_endpoints_pinned_no_change():
    """Both pinned ⇒ projection should be a no-op; positions unchanged."""
    sys = _two_particle_system(distance=2.0)
    sys.pin([0, 1])
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), k=1.0))
    X_before = sys.X.copy()
    sys.step(dt=1e-3, iters=10)
    np.testing.assert_array_equal(sys.X, X_before)
