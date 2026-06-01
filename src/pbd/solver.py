"""Step helpers: §3.5 rigid-mode-preserving damping (Phase B), and
collision-constraint generation (Phase C, TBD).
"""
from __future__ import annotations

import numpy as np


def damp_velocities(
    X: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    k_damp: float,
) -> None:
    """Damp the per-particle deviation from the global rigid-body velocity field.

    Implements paper §3.5 in-place on V. The fixed point (k_damp = 1) is
    rigid-body motion of the free particles; bulk translation and rotation
    survive any value of ``k_damp``.

    Pinned particles (W = 0) are excluded both from defining the rigid
    motion and from being damped.
    """
    if k_damp <= 0.0:
        return
    free = W > 0.0
    if free.sum() < 2:
        return

    masses = np.zeros_like(W)
    masses[free] = 1.0 / W[free]
    M = float(masses.sum())
    if M == 0.0:
        return

    Xf = X[free]
    Vf = V[free]
    mf = masses[free]

    xcm = (Xf * mf[:, None]).sum(axis=0) / M
    vcm = (Vf * mf[:, None]).sum(axis=0) / M

    r = Xf - xcm                                    # (n_free, 3)
    L = np.sum(np.cross(r, Vf * mf[:, None]), axis=0)  # (3,)

    # I = Σ m (||r||² I_3 - r r^T)
    rsq = np.sum(r * r, axis=1)                     # (n_free,)
    I = np.eye(3) * float(np.sum(mf * rsq))
    I -= np.einsum("i,ij,ik->jk", mf, r, r)

    # Solve I ω = L, falling back to least-squares if singular (e.g. colinear points).
    try:
        omega = np.linalg.solve(I, L)
    except np.linalg.LinAlgError:
        omega, *_ = np.linalg.lstsq(I, L, rcond=None)

    rigid_v = vcm + np.cross(np.broadcast_to(omega, r.shape), r)
    delta = rigid_v - Vf
    V[free] = Vf + k_damp * delta
