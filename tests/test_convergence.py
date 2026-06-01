"""Verification #7: more iters ⇒ smaller residual on a multi-edge chain.

Notes on what this tests vs. what the existing tests cover:

* test_distance.py::test_residual_monotone_in_iters covers a single edge.
* test_distance.py::test_k_prime_makes_residual_iter_independent covers
  the §3.3 linearization at k<1 (which deliberately makes residual iter-
  independent — the whole point of §3.3).
* This file covers the multi-edge case at k=1, where §3.3 is a no-op
  (k' = 1) and Jacobi convergence is exposed: the residual must be
  monotone non-increasing with iters and reach roundoff for "enough"
  iters.

For a 3-vertex chain with k=1, Jacobi convergence ratio is cos(π/3)=0.5,
so residual halves per iter and 32 iters lands well below 1e-9.
"""
from __future__ import annotations

import numpy as np

from pbd import Stretch, System, build_mesh


def _strip(n: int = 3, side: float = 1.0):
    V = np.zeros((n, 3), dtype=np.float64)
    V[:, 0] = np.linspace(0.0, side * (n - 1), n)
    F = np.array([[0, 1, n - 1]], dtype=np.int64)  # placeholder face
    return build_mesh(V, F), V


def _run_strip(iters: int, k: float, n: int = 3, dt: float = 1e-3):
    mesh, V0 = _strip(n)
    sys = System(V0.copy(), masses=np.ones(n), gravity=(0.0, 0.0, 0.0))
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    rest = np.linalg.norm(V0[edges[:, 1]] - V0[edges[:, 0]], axis=1)
    sys.add_constraint(Stretch(edges, rest, k=k))
    sys.X *= 1.5  # stretch the whole chain uniformly by 50%
    sys.step(dt=dt, iters=iters)
    L = np.linalg.norm(sys.X[edges[:, 1]] - sys.X[edges[:, 0]], axis=1)
    return float(np.max(np.abs(L - rest)))


def test_residual_monotone_in_iters_chain():
    """Iters sweep at k=1: residual non-increasing in iters."""
    residuals = [_run_strip(iters=n, k=1.0) for n in (1, 2, 4, 8, 16, 32)]
    for a, b in zip(residuals, residuals[1:]):
        assert b <= a + 1e-12, f"residual rose: {residuals}"
    # And the asymptote at 32 iters should be well below the 1-iter result.
    assert residuals[-1] < 0.01 * residuals[0], (
        f"first-iter residual {residuals[0]:.3e} vs final {residuals[-1]:.3e}"
    )


def test_residual_converges_to_zero_at_full_stiffness():
    """k=1 with enough iters ⇒ residual collapses to ~roundoff."""
    r = _run_strip(iters=32, k=1.0)
    assert r < 1e-9, f"k=1, 32 iters left residual {r:.3e}"


if __name__ == "__main__":
    test_residual_monotone_in_iters_chain()
    test_residual_converges_to_zero_at_full_stiffness()
    print("convergence: ok")
