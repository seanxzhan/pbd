"""Reflection symmetry — verification #4.

A symmetric setup (mesh + pin pattern + gravity) under symmetric initial
conditions should evolve symmetrically. We use a small grid pinned at two
opposite corners so the symmetry plane is well-defined and the drape is
non-trivial.
"""
from __future__ import annotations

import numpy as np

from pbd import Bend, Stretch, System, build_mesh


def _grid(n: int, side: float = 1.0):
    """An (n × n)-vertex regular grid mesh in the xy-plane, n odd.

    Each cell's diagonal alternates by parity of (i + j) so that the
    *topology* is invariant under y → -y reflection (when n is odd, a
    cell's parity is preserved by j → n-1-j). Without this the reflected
    grid is not a valid mirror of the original and tests can't measure
    symmetry breakage from the solver alone.
    """
    assert n % 2 == 1, "use odd n so parity is preserved under y-reflection"
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
                F.append([a, b, d])
                F.append([a, d, c])
            else:
                F.append([a, b, c])
                F.append([b, d, c])
    return build_mesh(V, np.array(F, dtype=np.int64))


def test_drape_symmetric_under_x_reflection():
    """Grid mesh, gravity = -ẑ, pinned at (-x, ±y) corners only on the −x edge.

    The setup is symmetric under (x, y, z) → (x, -y, z). After many steps,
    the drape must remain symmetric: X[i] reflected through y=0 must equal
    X[mirror(i)] up to numerical noise.
    """
    n = 5
    m = _grid(n, side=1.0)
    sys = System.from_mesh(m, density=1.0, gravity=(0.0, 0.0, -1.0))
    sys.add_constraint(Stretch.from_mesh(m, k=0.8))
    sys.add_constraint(Bend.from_mesh(m, k=0.2))

    # Pin the two corners on the −x edge (top and bottom).
    pin_corners = [0, (n - 1) * n]  # (j=0, i=0) and (j=n-1, i=0)
    sys.pin(pin_corners)

    for _ in range(100):
        sys.step(dt=5e-3, iters=10)

    # Mirror map: vertex (i, j) ↔ (i, n-1-j) under y→-y reflection.
    def mirror_idx(j: int, i: int) -> int:
        return (n - 1 - j) * n + i

    for j in range(n):
        for i in range(n):
            a = j * n + i
            b = mirror_idx(j, i)
            xa = sys.X[a].copy()
            xb = sys.X[b].copy()
            xb_reflected = xb * np.array([1.0, -1.0, 1.0])
            np.testing.assert_allclose(xa, xb_reflected, atol=1e-6,
                err_msg=f"asymmetry at vertex ({j},{i}): {xa} vs reflected {xb_reflected}")
