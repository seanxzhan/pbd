"""Proxy mesh in random wind, top 5% of vertices pinned.

Take an arbitrary triangle mesh, pin the topmost 5% of vertices (sorted
by y), and apply a per-vertex wind force with random direction and
magnitude every step. Defaults to a 20×20 cloth grid; pass ``--obj`` to
load any OBJ.

Wind generation: each step we draw a fresh random force per free vertex
and exponentially blend it into the previous wind via ``wind_coherence``.
Coherence ≈ 1 freezes the wind in place; ≈ 0 is pure white noise. The
default 0.92 gives gusty, naturally-correlated motion.

Coordinate convention: y-up; "highest" = largest y.
"""
from __future__ import annotations

import argparse

import numpy as np

from pbd import Bend, Stretch, System, build_mesh, load_obj
from pbd.viz import Viewer


def make_default_mesh(n: int = 20, side: float = 1.0):
    """Cloth-style grid in the xy-plane (z=0). y is up."""
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
            F.extend([[a, b, d], [a, d, c]])
    return build_mesh(V, np.array(F, dtype=np.int64))


def pick_top_fraction(V: np.ndarray, frac: float) -> np.ndarray:
    """Indices of the top ``frac`` of vertices, ranked by y. At least one."""
    n = V.shape[0]
    n_pin = max(1, int(np.ceil(frac * n)))
    return np.argsort(V[:, 1])[-n_pin:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default=None,
                    help="OBJ to load; default = 20x20 cloth grid")
    ap.add_argument("--n", type=int, default=20,
                    help="default grid resolution when --obj is not set")
    ap.add_argument("--pin-fraction", type=float, default=0.05,
                    help="fraction of topmost (highest-y) verts to pin")
    ap.add_argument("--wind-mean", type=float, default=1.0,
                    help="mean per-vertex wind force magnitude")
    ap.add_argument("--wind-std", type=float, default=1.0,
                    help="std of per-vertex wind force magnitude")
    ap.add_argument("--wind-coherence", type=float, default=0.95,
                    help="OU smoothing on wind (0=white noise, 1=frozen)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--dt", type=float, default=1.0 / 60)
    ap.add_argument("--k-stretch", type=float, default=0.99)
    ap.add_argument("--k-bend", type=float, default=0.3)
    ap.add_argument("--k-damp", type=float, default=0.05)
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"],
                    default="jacobi")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-frames", type=int, default=120)
    args = ap.parse_args()

    if args.obj:
        V, F = load_obj(args.obj)
        mesh = build_mesh(V, F)
    else:
        mesh = make_default_mesh(args.n)

    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, -9.81, 0.0))
    sys.add_constraint(Stretch.from_mesh(mesh, k=args.k_stretch))
    sys.add_constraint(Bend.from_mesh(mesh, k=args.k_bend))

    pinned = pick_top_fraction(mesh.V, args.pin_fraction)
    sys.pin(pinned)
    print(f"verts={mesh.n_verts}  faces={mesh.n_faces}  "
          f"pinned={len(pinned)} (top {100*args.pin_fraction:.1f}%)")

    rng = np.random.default_rng(args.seed)
    wind_state = np.zeros_like(sys.X)
    free = sys.W > 0.0

    def apply_wind(_frame=None):
        # Coherent random wind: blend a fresh per-vertex random force into
        # the previous wind. Translates to v += dt * w * F_wind.
        new_dir = rng.standard_normal(sys.X.shape)
        new_mag = np.abs(args.wind_mean
                         + args.wind_std * rng.standard_normal(sys.X.shape[0]))
        new_kick = new_dir * new_mag[:, None]
        wind_state[:] = (args.wind_coherence * wind_state
                         + (1.0 - args.wind_coherence) * new_kick)
        sys.V[free] += args.dt * sys.W[free, None] * wind_state[free]

    if args.smoke:
        import time
        t0 = time.time()
        for _ in range(args.smoke_frames):
            apply_wind()
            sys.step(dt=args.dt, iters=args.iters,
                     k_damp=args.k_damp, solver=args.solver)
        elapsed = time.time() - t0
        fps = args.smoke_frames / elapsed
        print(f"{args.smoke_frames} frames in {elapsed:.3f}s "
              f"({fps:.1f} fps) [solver={args.solver}]")
        print(f"final y range: [{sys.X[:, 1].min():.3f}, "
              f"{sys.X[:, 1].max():.3f}]")
        return

    viewer = Viewer(sys, mesh.F, name="proxy")
    # Highlight the pinned verts in red so the constraint is visible.
    import polyscope as ps
    pin_pc = ps.register_point_cloud("pinned", sys.X[pinned])
    pin_pc.set_radius(0.012, relative=False)
    pin_pc.set_color((0.9, 0.2, 0.2))

    # Wind is applied as a velocity kick before each sys.step. Hooking it
    # into Viewer's on_step (post-step) means a 1-frame lag, imperceptible
    # given the 60 Hz tick — but the cleaner integration would be a
    # pre-step callback if Viewer ever exposes one.
    viewer.run(dt=args.dt, iters=args.iters, k_damp=args.k_damp,
               solver=args.solver, on_step=apply_wind)


if __name__ == "__main__":
    main()
