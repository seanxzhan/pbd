"""Closed sphere mesh with volume constraint k_pressure=1.5 ⇒ inflates.

Verification #14, qualitative.
"""
from __future__ import annotations

import argparse

import numpy as np

from pbd import Bend, Stretch, System, Volume, build_mesh
from pbd.viz import Viewer


def icosphere(subdivisions: int = 2):
    """Build an icosphere by subdividing an icosahedron and projecting to
    the unit sphere. Returns build_mesh(V, F)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    V = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=np.float64)
    F = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int64)

    for _ in range(subdivisions):
        edge_mid = {}
        new_V = list(V)
        new_F = []

        def get_mid(a, b):
            key = (min(a, b), max(a, b))
            if key in edge_mid:
                return edge_mid[key]
            m = (V[a] + V[b]) * 0.5
            idx = len(new_V)
            new_V.append(m)
            edge_mid[key] = idx
            return idx

        for tri in F:
            a, b, c = tri
            ab = get_mid(a, b)
            bc = get_mid(b, c)
            ca = get_mid(c, a)
            new_F.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        V = np.array(new_V)
        F = np.array(new_F, dtype=np.int64)

    # Project to unit sphere.
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return build_mesh(V, F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdivisions", type=int, default=2)
    ap.add_argument("--k-pressure", type=float, default=1.5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dt", type=float, default=1.0 / 60)
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"],
                    default="jacobi",
                    help="Constraint solver: Jacobi (default) or graph-colored Gauss-Seidel")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-frames", type=int, default=120)
    args = ap.parse_args()

    mesh = icosphere(args.subdivisions)
    assert mesh.is_closed
    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Stretch.from_mesh(mesh, k=0.5))
    sys.add_constraint(Bend.from_mesh(mesh, k=0.1))
    sys.add_constraint(Volume.from_mesh(mesh, k_pressure=args.k_pressure, k=1.0))

    if args.smoke:
        from pbd.constraints.volume import _volume_sum
        V0 = _volume_sum(sys.X, mesh.F)

        import time
        t0 = time.time()
        for _ in range(args.smoke_frames):
            sys.step(dt=args.dt, iters=args.iters, k_damp=0.05,
                     solver=args.solver)
        elapsed = time.time() - t0
        fps = args.smoke_frames / elapsed

        V_now = _volume_sum(sys.X, mesh.F)
        print(f"verts: {mesh.n_verts}, faces: {mesh.n_faces}")
        print(f"{args.smoke_frames} frames in {elapsed:.3f}s ({fps:.1f} fps) "
              f"[solver={args.solver}]")
        print(f"volume ratio V/V0: {V_now/V0:.3f} (target {args.k_pressure})")
        return

    viewer = Viewer(sys, mesh.F, name="balloon")
    viewer.run(dt=args.dt, iters=args.iters, k_damp=0.05, solver=args.solver)


if __name__ == "__main__":
    main()
