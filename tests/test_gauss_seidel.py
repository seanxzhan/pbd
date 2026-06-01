"""Phase E: graph-colored Gauss-Seidel solver."""
from __future__ import annotations

import numpy as np

from pbd import Bend, Plane, Stretch, System, build_mesh
from pbd.coloring import greedy_color


# ------------------------------------------------ greedy coloring properties

def test_greedy_color_chain_uses_two_colors():
    """A 1-D chain of edges has chromatic number 2 (alternating)."""
    n = 10
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    colors = greedy_color(edges)
    assert colors.max() + 1 == 2, f"chain should use 2 colors, got {colors.max() + 1}"


def test_greedy_color_classes_are_vertex_disjoint():
    """For every color class, no two constraints share a vertex."""
    rng = np.random.default_rng(0)
    n_verts = 50
    n_edges = 80
    edges = rng.integers(0, n_verts, size=(n_edges, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]  # drop self-edges
    colors = greedy_color(edges)
    for c in range(colors.max() + 1):
        verts = edges[colors == c].ravel()
        assert len(verts) == len(set(verts.tolist())), (
            f"color {c} has duplicate vertices"
        )


def test_greedy_color_empty():
    assert greedy_color(np.zeros((0, 2), dtype=np.int64)).shape == (0,)


# --------------------------- GS reduces to Jacobi for non-conflicting batches

def test_gs_matches_jacobi_for_single_edge():
    """One edge: there's no GS gain to be had — both should produce the same."""
    X = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    sys_j = System(X.copy(), masses=np.ones(2), gravity=(0, 0, 0))
    sys_g = System(X.copy(), masses=np.ones(2), gravity=(0, 0, 0))
    sys_j.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), k=1.0))
    sys_g.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), k=1.0))
    sys_j.step(dt=1e-3, iters=5, solver="jacobi")
    sys_g.step(dt=1e-3, iters=5, solver="gauss-seidel")
    np.testing.assert_allclose(sys_j.X, sys_g.X, atol=1e-12)


# ---------------------- GS converges faster than Jacobi on a multi-edge chain

def _chain_residual(n: int, iters: int, solver: str, k: float = 1.0) -> float:
    V = np.zeros((n, 3))
    V[:, 0] = np.linspace(0, n - 1, n)
    F = np.array([[0, 1, n - 1]], dtype=np.int64)
    mesh = build_mesh(V, F)
    sys = System(V.copy(), masses=np.ones(n), gravity=(0, 0, 0))
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    rest = np.linalg.norm(V[edges[:, 1]] - V[edges[:, 0]], axis=1)
    sys.add_constraint(Stretch(edges, rest, k=k))
    sys.X *= 1.5
    sys.step(dt=1e-3, iters=iters, solver=solver)
    L = np.linalg.norm(sys.X[edges[:, 1]] - sys.X[edges[:, 0]], axis=1)
    return float(np.max(np.abs(L - rest)))


def test_gs_converges_strictly_faster_than_jacobi_on_chain():
    """At k=1, GS gives a strictly smaller residual than Jacobi at same iters.

    Linear Poisson theory says Jacobi has convergence ratio cos(π/N) and
    GS has cos²(π/N), but PBD's distance constraint is nonlinear so the
    practical factor is smaller — typically 2-4× at moderate iters.
    """
    n = 8
    for iters in (10, 30):
        r_j = _chain_residual(n, iters, "jacobi")
        r_g = _chain_residual(n, iters, "gauss-seidel")
        assert r_g < r_j, (
            f"iters={iters}: GS residual {r_g:.3e} should beat Jacobi {r_j:.3e}"
        )
    # At iters=30, the gap should be at least 2× — easily reachable.
    r_j = _chain_residual(n, 30, "jacobi")
    r_g = _chain_residual(n, 30, "gauss-seidel")
    assert r_g < 0.5 * r_j, (
        f"expected ≥2× reduction at 30 iters; got Jacobi={r_j:.3e}, GS={r_g:.3e}"
    )


# ------------------------------------------- GS still respects pinning, gravity

def test_gs_respects_pinning_and_gravity():
    """A pinned vertex must remain at rest under GS too."""
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    sys = System(X.copy(), masses=np.ones(3), gravity=(0, 0, -9.81))
    sys.pin([0])
    edges = np.array([[0, 1], [1, 2]])
    rest = np.array([1.0, 1.0])
    sys.add_constraint(Stretch(edges, rest, k=1.0))
    for _ in range(50):
        sys.step(dt=1e-2, iters=10, solver="gauss-seidel")
    np.testing.assert_array_equal(sys.X[0], [0.0, 0.0, 0.0])
    # Free vertices should have moved.
    assert sys.X[1, 2] < -0.01
    assert sys.X[2, 2] < -0.01


# ------------------------------- GS works with cloth + bend + collision

def test_gs_runs_full_cloth_pipeline():
    """End-to-end: stretch + bend + plane collision under GS, no NaNs, settles."""
    n = 6
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    V = np.stack([XX.ravel(), np.ones(n * n) * 1.5, YY.ravel()], axis=1)
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i; b = j * n + (i + 1)
            c = (j + 1) * n + i; d = (j + 1) * n + (i + 1)
            F.extend([[a, b, d], [a, d, c]])
    mesh = build_mesh(V, np.array(F, dtype=np.int64))
    sys = System.from_mesh(mesh, density=1.0, gravity=(0, -9.81, 0))
    sys.add_constraint(Stretch.from_mesh(mesh, k=0.9))
    sys.add_constraint(Bend.from_mesh(mesh, k=0.2))
    sys.add_collider(Plane(normal=(0, 1, 0), offset=0.0))
    for _ in range(120):
        sys.step(dt=1 / 60, iters=10, k_damp=0.05,
                 restitution=0.0, friction=0.4, solver="gauss-seidel")
    assert np.all(np.isfinite(sys.X)), "GS pipeline produced NaNs/infs"
    assert sys.X[:, 1].min() >= -1e-6, (
        f"vertex below floor: y_min={sys.X[:, 1].min():.3e}"
    )


if __name__ == "__main__":
    test_greedy_color_chain_uses_two_colors()
    test_greedy_color_classes_are_vertex_disjoint()
    test_greedy_color_empty()
    test_gs_matches_jacobi_for_single_edge()
    test_gs_converges_strictly_faster_than_jacobi_on_chain()
    test_gs_respects_pinning_and_gravity()
    test_gs_runs_full_cloth_pipeline()
    print("Phase E: gauss-seidel tests passed.")
