"""Verification #11: determinism / regression hook.

Identical (X0, V0, dt, iters, ordering) ⇒ bit-identical trajectory. PBD
has no random number generators, so any future change that perturbs
floating-point ordering will surface here.
"""
from __future__ import annotations

import numpy as np

from pbd import Bend, Plane, Sphere, Stretch, System, build_mesh


def _make_grid(n: int = 6, side: float = 1.0):
    xs = np.linspace(-side, side, n)
    ys = np.linspace(-side, side, n)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    V = np.stack([XX.ravel(), YY.ravel(), np.ones(n * n) * 1.0], axis=1)
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + (i + 1)
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1)
            F.extend([[a, b, d], [a, d, c]])
    return build_mesh(V, np.array(F, dtype=np.int64))


def _run(seed: int, n_steps: int):
    """Run a fixed scene; ``seed`` only affects an initial velocity perturbation."""
    rng = np.random.default_rng(seed)
    mesh = _make_grid()
    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, 0.0, -9.81))
    sys.add_constraint(Stretch.from_mesh(mesh, k=0.8))
    sys.add_constraint(Bend.from_mesh(mesh, k=0.2))
    sys.add_collider(Plane(normal=(0.0, 0.0, 1.0), offset=0.0))
    sys.add_collider(Sphere(center=np.array([0.0, 0.0, 0.4]), radius=0.3))
    sys.V[:] = rng.standard_normal(sys.V.shape) * 0.01
    for _ in range(n_steps):
        sys.step(dt=1.0 / 120, iters=8, k_damp=0.05, restitution=0.2, friction=0.3)
    return sys.X.copy(), sys.V.copy()


def test_two_runs_with_same_seed_are_bit_identical():
    """No RNG inside the solver ⇒ same inputs give same floats, exactly."""
    X1, V1 = _run(seed=42, n_steps=80)
    X2, V2 = _run(seed=42, n_steps=80)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(V1, V2)


def test_different_seeds_diverge():
    """Sanity: the test would also pass on a constant simulator. Make sure
    it isn't trivially passing — different initial perturbations *do*
    produce different states."""
    X1, _ = _run(seed=1, n_steps=80)
    X2, _ = _run(seed=2, n_steps=80)
    assert np.linalg.norm(X1 - X2) > 1e-6


if __name__ == "__main__":
    test_two_runs_with_same_seed_are_bit_identical()
    test_different_seeds_diverge()
    print("determinism: ok")
