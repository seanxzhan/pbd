"""Energy non-increase — verification #9.

PBD's constraint projection is dissipative in general (it shortens the
position residual without restoring kinetic energy lost to projection).
With damping=0 and no gravity, total energy must not *increase*: the
constraint solve only projects positions onto the manifold and the
implicit velocity recovery converts that to a velocity that is at least
as small in the constraint-violating mode as before. We can't claim
exact conservation, but we can claim non-amplification.
"""
from __future__ import annotations

import numpy as np

from pbd import Bend, Stretch, System, build_mesh


def _tetra():
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.4, 0.9],
    ])
    F = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ], dtype=np.int64)
    return build_mesh(V, F)


def _kinetic_energy(V, masses):
    return 0.5 * float(np.sum(masses * np.sum(V * V, axis=1)))


def test_kinetic_energy_does_not_grow_from_internal_motion():
    """Rest configuration with internal (non-rigid) velocity field.

    Constraints are not violated at t=0, so projection only fires against
    the dt² nonlinearity of the predict step. With zero gravity and no
    damping, total KE must not *exceed* its initial value: PBD's projection
    is dissipative on the constraint-violating component of velocity.
    """
    m = _tetra()
    sys = System.from_mesh(m, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Stretch.from_mesh(m, k=1.0))
    sys.add_constraint(Bend.from_mesh(m, k=1.0))

    # Pure internal velocity field: zero net momentum, zero net angular
    # momentum, but non-rigid (nonzero strain rate). This is the mode
    # PBD is supposed to dissipate.
    # An expansion-mode velocity field: each particle moves radially out
    # from the CoM. This is purely strain-rate, no rigid component.
    cm = (sys.X * sys.masses[:, None]).sum(axis=0) / sys.masses.sum()
    r = sys.X - cm
    sys.V[:] = 2.0 * r
    # Strip any residual linear momentum so we isolate internal modes.
    sys.V -= (sys.V * sys.masses[:, None]).sum(axis=0) / sys.masses.sum()

    ke0 = _kinetic_energy(sys.V, sys.masses)
    assert ke0 > 1e-3, "test setup must inject real KE"

    dt = 1e-3
    ke_max = ke0
    for _ in range(500):
        sys.step(dt=dt, iters=10)
        ke = _kinetic_energy(sys.V, sys.masses)
        # Allow tiny floating-point slop. Real growth would be a bug.
        assert ke <= ke0 + 1e-9, (
            f"KE grew from {ke0:.6e} to {ke:.6e}; PBD must be dissipative"
        )
        ke_max = max(ke_max, ke)
    # With full stiffness over 500 steps, should have dissipated noticeably.
    ke_final = _kinetic_energy(sys.V, sys.masses)
    assert ke_final < ke0, f"expected some dissipation, got {ke_final} >= {ke0}"
