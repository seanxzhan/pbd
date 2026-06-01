"""Particle system + simulation step (paper §3.1)."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from pbd.constraints.base import ConstraintGroup
from pbd.constraints.collision import generate_collision_constraints
from pbd.mesh import Mesh
from pbd.solver import damp_velocities


class System:
    """Particles + the PBD step loop.

    Attributes
    ----------
    X : (N, 3) float64
        Current positions.
    V : (N, 3) float64
        Current velocities.
    W : (N,) float64
        Inverse masses. ``W[i] == 0`` means vertex i is pinned.
    """

    def __init__(
        self,
        X: np.ndarray,
        masses: np.ndarray,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
    ):
        X = np.ascontiguousarray(X, dtype=np.float64)
        masses = np.ascontiguousarray(masses, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"X must be (N, 3); got {X.shape}")
        if masses.shape != (X.shape[0],):
            raise ValueError(f"masses shape {masses.shape} != ({X.shape[0]},)")
        if (masses <= 0).any():
            raise ValueError("masses must be > 0; use pin() for fixed verts")

        self.X = X
        self.V = np.zeros_like(X)
        self.masses = masses
        self.W = 1.0 / masses
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.constraints: list[ConstraintGroup] = []
        self.colliders: list = []

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
        density: float,
        **kw,
    ) -> "System":
        masses = mesh.vertex_masses(density)
        return cls(mesh.V.copy(), masses, **kw)

    def pin(self, indices: Iterable[int]) -> None:
        """Make the listed vertices kinematic by setting their inverse mass to 0."""
        self.W[list(indices)] = 0.0
        # Also kill any residual velocity on pinned verts.
        self.V[self.W == 0.0] = 0.0

    def add_constraint(self, group: ConstraintGroup) -> None:
        self.constraints.append(group)

    def add_collider(self, collider) -> None:
        """Register a static collider (Plane, Sphere). Per-step inequality
        constraints will be generated against it."""
        self.colliders.append(collider)

    # ------------------------------------------------------------------ step

    def step(
        self,
        dt: float,
        iters: int = 1,
        k_damp: float = 0.0,
        restitution: float = 0.0,
        friction: float = 0.0,
        solver: str = "jacobi",
    ) -> None:
        """One PBD step. See paper §3.1 lines (5)–(16).

        Parameters
        ----------
        solver : {"jacobi", "gauss-seidel"}
            "jacobi" (default) — within an iteration each constraint
            group's batch is computed from the *same* P, then summed via
            ``np.add.at``. Fast and simple; convergence rate is
            cos(π/n) on a 1-D chain.
            "gauss-seidel" — within an iteration each group's color
            classes are processed sequentially: constraints in a class
            are vertex-disjoint (so still vectorized), but the next
            class sees the corrections from previous classes. Roughly
            doubles the convergence rate on cloth, at the cost of one
            small Python loop per group per iter.
        """
        if solver not in ("jacobi", "gauss-seidel"):
            raise ValueError(
                f"solver must be 'jacobi' or 'gauss-seidel', got {solver!r}"
            )
        free = self.W > 0.0

        # (5) gravity (f_ext = m*g  ⇒  w*f_ext = g)
        self.V[free] += dt * self.gravity

        # (6) §3.5 damping
        if k_damp > 0.0:
            damp_velocities(self.X, self.V, self.W, k_damp)

        # Snapshot pre-step velocity for restitution/friction (eq. 16).
        V_pre = self.V.copy()

        # (7) predict
        P = self.X + dt * self.V

        # (8) generate per-step collision constraints (paper §6).
        coll = generate_collision_constraints(P, self.colliders, free, k=1.0)
        n_static = len(self.constraints)

        # §3.3 stiffness linearization: pre-compute k' per group so the
        # residual after `iters` iterations is independent of `iters`.
        if iters > 0:
            k_eff_static = np.array(
                [1.0 - (1.0 - g.k) ** (1.0 / iters) for g in self.constraints],
                dtype=np.float64,
            )
            # Collision constraints stay at k=1 each iter — no compounding
            # since they're regenerated fresh each step.
            k_eff_coll = 1.0
        else:
            k_eff_static = np.zeros(n_static)
            k_eff_coll = 0.0

        # (9)–(11) constraint projection — Jacobi or graph-colored G-S
        if solver == "jacobi":
            for _ in range(iters):
                for ke, group in zip(k_eff_static, self.constraints):
                    P += ke * group.project_batch(P, self.W)
                if coll.idx.shape[0] > 0:
                    P += k_eff_coll * coll.project_batch(P, self.W)
        else:  # gauss-seidel: sweep color classes sequentially
            for _ in range(iters):
                for ke, group in zip(k_eff_static, self.constraints):
                    for c in range(group.n_colors):
                        P += ke * group.project_color(P, self.W, c)
                if coll.idx.shape[0] > 0:
                    for c in range(coll.n_colors):
                        P += k_eff_coll * coll.project_color(P, self.W, c)

        # (13) velocity from positional change
        self.V = (P - self.X) / dt

        # (14) commit
        self.X = P

        # (16) velocity update for collided particles (restitution + friction)
        if coll.idx.shape[0] > 0:
            self._collision_velocity_update(coll, V_pre, restitution, friction)

    def _collision_velocity_update(
        self,
        coll,
        V_pre: np.ndarray,
        restitution: float,
        friction: float,
    ) -> None:
        """Apply restitution and Coulomb friction (paper §6 / eq. 16).

        Decisions about which particles collided and how strong the impulse
        is are made from the *pre-step* velocity ``V_pre``. The current
        ``self.V`` is the post-projection velocity, which contains a
        spurious outward component from the position correction; we
        overwrite the normal component cleanly using v_pre and ε.
        """
        i = coll.idx
        n = coll.normals                                # (M, 3) outward
        v_pre = V_pre[i]
        v_post = self.V[i]

        vn_pre = np.einsum("ij,ij->i", v_pre, n)        # < 0 when entering wall
        active = vn_pre < 0.0
        if not active.any():
            return

        # Replace the normal component:
        #   v_n_new = -ε · v_n_pre   (positive outward when ε > 0)
        # i.e. v_post -= (v_post · n)·n + ε·(v_pre · n)·n
        vn_post = np.einsum("ij,ij->i", v_post[active], n[active])
        v_post[active] -= (vn_post[:, None] * n[active]
                           + restitution * vn_pre[active][:, None] * n[active])

        # Coulomb friction: damp the (post-projection) tangential velocity
        # by min(μ · |v_n_pre|, |v_t|).
        if friction > 0.0:
            vn_now = np.einsum("ij,ij->i", v_post[active], n[active])
            v_t = v_post[active] - vn_now[:, None] * n[active]
            vt_mag = np.linalg.norm(v_t, axis=1)
            cap = np.minimum(friction * np.abs(vn_pre[active]), vt_mag)
            ok = vt_mag > 1e-12
            t_hat = np.zeros_like(v_t)
            t_hat[ok] = v_t[ok] / vt_mag[ok, None]
            v_post[active] -= cap[:, None] * t_hat

        self.V[i] = v_post
