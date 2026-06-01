"""Hanging cloth golden scene — verification #13.

A 32×32 vertex grid pinned at its top two corners, gravity along -y.
Coordinate convention: y-up.
"""
from __future__ import annotations

import argparse

import numpy as np

from pbd import Bend, Stretch, System, build_mesh
from pbd.viz import Viewer


def make_grid_mesh(n: int = 32, side: float = 1.0):
    """Cloth in the xy-plane (z=0), centred on origin. y is "up"."""
    xs = np.linspace(-side, side, n)
    ys = np.linspace(-side, side, n)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    V = np.stack([XX.ravel(), YY.ravel(), np.zeros(n * n)], axis=1)
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + (i + 1)
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1)
            if (i + j) % 2 == 0:
                F.extend([[a, b, d], [a, d, c]])
            else:
                F.extend([[a, b, c], [b, d, c]])
    return build_mesh(V, np.array(F, dtype=np.int64)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="grid resolution")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dt", type=float, default=1.0 / 60)
    ap.add_argument("--k-stretch", type=float, default=0.9)
    ap.add_argument("--k-bend", type=float, default=0.2)
    ap.add_argument("--k-damp", type=float, default=0.02)
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"],
                    default="jacobi",
                    help="Constraint solver: Jacobi (default) or graph-colored Gauss-Seidel")
    ap.add_argument("--smoke", action="store_true",
                    help="step N frames headless and exit (CI / sanity)")
    ap.add_argument("--smoke-frames", type=int, default=120)
    args = ap.parse_args()

    mesh, n = make_grid_mesh(args.n)
    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, -9.81, 0.0))
    sys.add_constraint(Stretch.from_mesh(mesh, k=args.k_stretch))
    sys.add_constraint(Bend.from_mesh(mesh, k=args.k_bend))

    # Pin the two top corners (max-y vertices in the y-up cloth).
    j_top = n - 1
    sys.pin([j_top * n + 0, j_top * n + (n - 1)])

    if args.smoke:
        import time
        t0 = time.time()
        for _ in range(args.smoke_frames):
            sys.step(dt=args.dt, iters=args.iters, k_damp=args.k_damp,
                     solver=args.solver)
        elapsed = time.time() - t0
        fps = args.smoke_frames / elapsed
        print(f"{args.smoke_frames} frames in {elapsed:.3f}s ({fps:.1f} fps) "
              f"[solver={args.solver}]")
        print(f"final mean vertex y: {sys.X[:, 1].mean():.4f}")
        return

    viewer = Viewer(sys, mesh.F, name="cloth")
    viewer.run(dt=args.dt, iters=args.iters, k_damp=args.k_damp,
               solver=args.solver)


if __name__ == "__main__":
    main()
