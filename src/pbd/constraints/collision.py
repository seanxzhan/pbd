"""Static colliders (plane, sphere) + per-step collision constraint group.

Paper §6: collisions are encoded as inequality constraints C(p) ≥ 0 that
are *generated each step* (not static topology-derived like Stretch). At
the start of each step we check predicted positions against colliders;
penetrating particles get a one-shot constraint that pushes them onto the
surface (C = 0). After the step, the velocities of those particles are
post-processed with restitution and Coulomb friction (eq. 16).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbd.coloring import greedy_color
from pbd.constraints.base import ConstraintGroup


# ---------------------------------------------------------------- colliders


@dataclass
class Plane:
    """Half-space {x : n · x ≥ d}, with outward normal n (unit) and offset d.

    A particle at position p is inside the allowed region iff
    ``n · p - d ≥ 0``.
    """
    normal: np.ndarray
    offset: float

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=np.float64)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            raise ValueError("plane normal must be nonzero")
        self.normal = n / nn
        self.offset = float(self.offset)

    def signed_distance(self, P: np.ndarray) -> np.ndarray:
        """(N,) signed distance; ≥ 0 outside, < 0 inside the wall."""
        return P @ self.normal - self.offset

    def surface_normals(self, P: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.normal, P.shape).copy()

    def contact_anchors(
        self,
        X: np.ndarray,
        P: np.ndarray,
        free_mask: np.ndarray,
        skin: float,
    ):
        """Static SDF detection — closed-form for a half-space."""
        sdf = self.signed_distance(P)
        hit = (sdf < skin) & free_mask
        if not hit.any():
            return None
        Phit = P[hit]
        n = self.surface_normals(Phit)
        # offset = n · q where q is the surface point directly below P.
        offsets = np.einsum("ij,ij->i", Phit, n) - sdf[hit]
        return np.where(hit)[0], n, offsets


@dataclass
class Sphere:
    """Solid sphere obstacle: outside region {x : |x - c| ≥ r}."""
    center: np.ndarray
    radius: float

    def __post_init__(self):
        self.center = np.ascontiguousarray(self.center, dtype=np.float64)
        if self.center.shape != (3,):
            raise ValueError(f"sphere center must be (3,); got {self.center.shape}")
        if self.radius <= 0:
            raise ValueError("sphere radius must be > 0")
        self.radius = float(self.radius)

    def signed_distance(self, P: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(P - self.center, axis=1)
        return d - self.radius

    def surface_normals(self, P: np.ndarray) -> np.ndarray:
        d = P - self.center
        L = np.linalg.norm(d, axis=1, keepdims=True)
        # If a particle is exactly at the centre, fall back to +x; this is
        # degenerate but shouldn't NaN.
        ok = L > 1e-12
        n = np.zeros_like(d)
        n[ok[:, 0]] = d[ok[:, 0]] / L[ok[:, 0]]
        n[~ok[:, 0]] = np.array([1.0, 0.0, 0.0])
        return n

    def contact_anchors(
        self,
        X: np.ndarray,
        P: np.ndarray,
        free_mask: np.ndarray,
        skin: float,
    ):
        """Static SDF detection — closed-form for a sphere."""
        sdf = self.signed_distance(P)
        hit = (sdf < skin) & free_mask
        if not hit.any():
            return None
        Phit = P[hit]
        n = self.surface_normals(Phit)
        offsets = np.einsum("ij,ij->i", Phit, n) - sdf[hit]
        return np.where(hit)[0], n, offsets


class TriangleMesh:
    """Static triangle-mesh obstacle implementing the paper §3.4 flow.

    Detection is the two-step recipe from the paper:

    1. **Continuous collision** — ray-cast the segment ``X_i → P_i``
       against every triangle. The smallest valid ``t`` gives an
       entry point ``q_c`` and outward face normal ``n_c``; the
       constraint ``n_c · (p − q_c) ≥ 0`` (eq. 17 in the paper) pushes
       the particle back to the outside half-space. This is the case
       the paper describes as "the entire motion … crosses an object's
       boundary".
    2. **Closest-point fallback** — when CCD finds no hit but ``P_i``
       still ends up inside the obstacle (or within the skin band),
       use the closest point on the mesh and that face's outward
       normal in place of ``q_c, n_c``. The paper: "If the predicted
       position p_i is inside an object, … we may use the closest
       point on the surface to p_i and … the surface normal at this
       point."

    The two paths together cover (a) clean first crossings, (b)
    resting/grazing contact via the skin band, and (c) recovery for
    particles that snuck inside during constraint iteration. Pure CCD
    misses (b)/(c); pure closest-point misses fast crossings of thin
    obstacles. Together they match §3.4.

    Winding requirement: face normals are computed from ``cross(V1-V0,
    V2-V0)`` and **must point outward consistently** across the mesh.
    Most OBJ exporters do this; ``io.fix_winding`` repairs the rest.
    With inconsistent winding the closest-point fallback will push
    particles in the wrong direction on the flipped faces.

    Convexity assumption: the closest-point fallback uses the host
    face's plane normal as the half-space normal. On a convex mesh
    that is the SDF gradient at the surface; on a non-convex mesh it
    can disagree (a particle far outside the mesh but "behind" some
    face's plane reads as "inside" and gets a phantom constraint).
    For non-convex obstacles, replace the mesh with its convex hull
    via ``io.convex_hull`` before constructing the collider.

    No BVH; both ray-vs-tri and closest-point-on-tri are vectorized
    over ``(n_active, n_faces)``. Suitable for obstacles up to a few
    thousand triangles; beyond that, add a BVH prune.
    """

    def __init__(self, V: np.ndarray, F: np.ndarray):
        V = np.ascontiguousarray(V, dtype=np.float64)
        F = np.ascontiguousarray(F, dtype=np.int64)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(f"V must be (Vn, 3); got {V.shape}")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"F must be (Fn, 3); got {F.shape}")
        self.V = V
        self.F = F
        # Edge / normal precompute.
        self.V0 = V[F[:, 0]]                                # (Fn, 3)
        self.V1 = V[F[:, 1]]
        self.V2 = V[F[:, 2]]
        self._e1 = self.V1 - self.V0
        self._e2 = self.V2 - self.V0
        n = np.cross(self._e1, self._e2)
        L = np.linalg.norm(n, axis=1, keepdims=True)
        if (L < 1e-20).any():
            raise ValueError("TriangleMesh has degenerate (zero-area) faces")
        self.face_normals = n / L
        # Cache dot-products needed for closest-point bary solve.
        self._e1e1 = np.einsum("fi,fi->f", self._e1, self._e1)
        self._e1e2 = np.einsum("fi,fi->f", self._e1, self._e2)
        self._e2e2 = np.einsum("fi,fi->f", self._e2, self._e2)
        self._det = self._e1e1 * self._e2e2 - self._e1e2 * self._e1e2

    def contact_anchors(
        self,
        X: np.ndarray,
        P: np.ndarray,
        free_mask: np.ndarray,
        skin: float,
    ):
        """Paper §3.4 detection: CCD first, closest-point fallback."""
        active = np.where(free_mask)[0]
        if active.size == 0 or self.F.shape[0] == 0:
            return None

        o = X[active]                                       # (Na, 3)
        p = P[active]                                       # (Na, 3)
        d = p - o                                           # ray (Na, 3)

        # ---------- (1) CCD: first triangle hit on the X→P segment.
        t_min, face_min = _ray_segment_min_t(
            o, d, self.V0, self._e1, self._e2, skin=skin
        )
        ccd_hit = np.isfinite(t_min)                        # (Na,)

        # ---------- (2) Closest-point fallback for the non-CCD-hit set.
        # We compute closest-point for ALL active particles (it's cheap
        # vs. the alternative of branching), but only consume it for
        # those CCD missed.
        cp = _closest_point_on_tris(
            p, self.V0, self._e1, self._e2, self.face_normals,
            self._e1e1, self._e1e2, self._e2e2, self._det,
        )                                                   # (Na, F, 3)
        diff = p[:, None, :] - cp
        dist_sq = np.einsum("nfi,nfi->nf", diff, diff)
        cp_face_idx = np.argmin(dist_sq, axis=1)            # (Na,)
        rows = np.arange(active.size)
        cp_best = cp[rows, cp_face_idx]                     # (Na, 3)
        n_cp_face = self.face_normals[cp_face_idx]          # (Na, 3)
        # Signed distance at the closest point: < 0 ⇒ inside.
        sd = np.einsum("ni,ni->n", p - cp_best, n_cp_face)  # (Na,)
        cp_hit = (sd < skin) & (~ccd_hit)

        # ---------- assemble the unified hit set.
        any_hit = ccd_hit | cp_hit
        if not any_hit.any():
            return None

        local = np.where(any_hit)[0]
        normals_out = np.where(
            ccd_hit[local, None],
            self.face_normals[face_min[local]],
            self.face_normals[cp_face_idx[local]],
        )
        # Anchor: CCD entry point q_c = o + t·d for hits; closest-point
        # cp for the fallback. (Clip t to 0 for non-CCD rows to avoid
        # multiplying +inf·d, which would warn; the np.where below then
        # discards those dummy anchors.)
        t_safe = np.where(ccd_hit[local], t_min[local], 0.0)
        q_ccd = o[local] + t_safe[:, None] * d[local]
        anchors = np.where(ccd_hit[local, None], q_ccd, cp_best[local])
        offsets = np.einsum("ni,ni->n", anchors, normals_out)
        return active[local], normals_out, offsets


def _ray_segment_min_t(
    o: np.ndarray,
    d: np.ndarray,
    V0: np.ndarray,
    E1: np.ndarray,
    E2: np.ndarray,
    skin: float = 0.0,
    eps: float = 1e-12,
):
    """Vectorized Möller–Trumbore: smallest valid t per ray, or +inf.

    A "valid" t is one where (i) ``t ∈ [0, 1 + skin/|d|]`` so the hit
    is on the segment (extended by ``skin`` for hysteresis), AND (ii)
    the ray is heading **into** the face's outward half-space, i.e.
    ``d · n_face < 0``. Outgoing crossings are rejected so a particle
    leaving the obstacle along ``X→P`` never gets pinned in place.
    """
    N = o.shape[0]
    # Broadcast: shape (N, F, 3).
    h = np.cross(d[:, None, :], E2[None, :, :])             # (N, F, 3)
    a = np.einsum("fi,nfi->nf", E1, h)                      # (N, F)
    parallel = np.abs(a) < eps
    inv_a = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, a))

    s = o[:, None, :] - V0[None, :, :]                      # (N, F, 3)
    u = inv_a * np.einsum("nfi,nfi->nf", s, h)
    miss_u = (u < 0.0) | (u > 1.0)

    qv = np.cross(s, E1[None, :, :])                        # (N, F, 3)
    v = inv_a * np.einsum("nfi,nfi->nf", d[:, None, :], qv)
    miss_v = (v < 0.0) | (u + v > 1.0)

    t = inv_a * np.einsum("fi,nfi->nf", E2, qv)
    d_len = np.linalg.norm(d, axis=1)                       # (N,)
    upper = np.where(d_len > eps, 1.0 + skin / np.where(d_len > eps, d_len, 1.0), 0.0)
    miss_t = (t < 0.0) | (t > upper[:, None])

    # Reject "outgoing" crossings: a particle moving from inside → out
    # should NOT generate a constraint anchored at the exit face — that
    # would pin it inside. Check d · n_face < 0 to keep only ingoing
    # crossings (face normals are precomputed and consistent).
    n_face = np.cross(E1, E2)                               # (F, 3) unnorm OK
    d_dot_n = np.einsum("ni,fi->nf", d, n_face)             # (N, F)
    miss_dir = d_dot_n >= 0.0

    miss = parallel | miss_u | miss_v | miss_t | miss_dir
    t = np.where(miss, np.inf, t)
    face_min = np.argmin(t, axis=1)
    t_min = t[np.arange(N), face_min]
    return t_min, face_min


def _closest_point_on_tris(
    p: np.ndarray,
    V0: np.ndarray,
    E1: np.ndarray,
    E2: np.ndarray,
    n: np.ndarray,
    e1e1: np.ndarray,
    e1e2: np.ndarray,
    e2e2: np.ndarray,
    det: np.ndarray,
):
    """Vectorized closest point on each triangle for each particle.

    Parameters
    ----------
    p   : (N, 3) particle positions
    V0  : (F, 3) anchor vertex of each face
    E1  : (F, 3) edge V1-V0
    E2  : (F, 3) edge V2-V0
    n   : (F, 3) unit face normals
    e1e1, e1e2, e2e2, det : (F,) precomputed dot products and
        ``det = e1e1·e2e2 - e1e2²`` for the bary solve.

    Returns
    -------
    cp : (N, F, 3) closest point on triangle f to particle i.

    Strategy: project each particle onto each face's plane, solve for
    barycentric (s, t). If (s, t) lies inside the triangle, the
    projection is the closest point. Otherwise compute the closest
    point on each of the three edges (clamped to [0, 1]) and pick the
    edge with the smallest distance. The two cases are stitched with
    ``np.where`` on the in-triangle mask.
    """
    # In-plane projection.
    diff0 = p[:, None, :] - V0[None, :, :]                  # (N, F, 3)
    dn = np.einsum("nfi,fi->nf", diff0, n)                  # (N, F)
    proj = p[:, None, :] - dn[..., None] * n[None, :, :]    # (N, F, 3)

    # Barycentric of proj w.r.t. (V0, V0+E1, V0+E2).
    diff_proj = proj - V0[None, :, :]                       # (N, F, 3)
    re1 = np.einsum("nfi,fi->nf", diff_proj, E1)
    re2 = np.einsum("nfi,fi->nf", diff_proj, E2)
    s = (e2e2[None, :] * re1 - e1e2[None, :] * re2) / det[None, :]
    t = (e1e1[None, :] * re2 - e1e2[None, :] * re1) / det[None, :]
    inside = (s >= 0.0) & (t >= 0.0) & (s + t <= 1.0)       # (N, F)

    # Closest point on each of the 3 edges (clamped).
    # Edge V0→V1.
    u1 = np.clip(np.einsum("nfi,fi->nf", diff0, E1) / e1e1[None, :], 0.0, 1.0)
    cp1 = V0[None, :, :] + u1[..., None] * E1[None, :, :]
    # Edge V0→V2.
    u2 = np.clip(np.einsum("nfi,fi->nf", diff0, E2) / e2e2[None, :], 0.0, 1.0)
    cp2 = V0[None, :, :] + u2[..., None] * E2[None, :, :]
    # Edge V1→V2 = (V0+E1) → (V0+E2): direction E2-E1.
    e3 = E2 - E1
    e33 = np.einsum("fi,fi->f", e3, e3)
    diff1 = p[:, None, :] - (V0 + E1)[None, :, :]
    u3 = np.clip(np.einsum("nfi,fi->nf", diff1, e3) / e33[None, :], 0.0, 1.0)
    cp3 = (V0 + E1)[None, :, :] + u3[..., None] * e3[None, :, :]

    pb = p[:, None, :]
    d1 = np.einsum("nfi,nfi->nf", cp1 - pb, cp1 - pb)
    d2 = np.einsum("nfi,nfi->nf", cp2 - pb, cp2 - pb)
    d3 = np.einsum("nfi,nfi->nf", cp3 - pb, cp3 - pb)
    edge_d = np.stack([d1, d2, d3], axis=-1)                # (N, F, 3)
    edge_cp = np.stack([cp1, cp2, cp3], axis=-2)            # (N, F, 3, 3)
    e_idx = np.argmin(edge_d, axis=-1)                      # (N, F)
    cp_edge = np.take_along_axis(
        edge_cp, e_idx[..., None, None].repeat(3, axis=-1), axis=-2
    ).squeeze(-2)                                           # (N, F, 3)

    return np.where(inside[..., None], proj, cp_edge)


# ----------------------------------------------------- collision constraint


class CollisionGroup(ConstraintGroup):
    """One-shot inequality constraints generated at predict time.

    Each constraint pushes a single particle ``i`` onto a target half-space
    defined by an outward unit normal ``n_i`` and a target signed distance
    ``d_i``: enforce ``n_i · (p_i - p0_i) ≥ d_i`` where ``p0_i`` is the
    surface anchor (precomputed at generation time so the constraint
    becomes a simple plane projection during iteration).

    Stored fields:
        idx     : (M,) int64 particle indices
        normals : (M, 3) float64 unit outward normals
        offsets : (M,) float64 — the value ``offset_i = n_i · p0_i`` so
                  the constraint reads C_i(p) = n_i · p - offset_i.
    """

    def __init__(
        self,
        idx: np.ndarray,
        normals: np.ndarray,
        offsets: np.ndarray,
        k: float = 1.0,
    ):
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        normals = np.ascontiguousarray(normals, dtype=np.float64)
        offsets = np.ascontiguousarray(offsets, dtype=np.float64)
        if idx.ndim != 1:
            raise ValueError(f"idx must be (M,); got {idx.shape}")
        if normals.shape != (idx.shape[0], 3):
            raise ValueError(f"normals shape {normals.shape} != ({idx.shape[0]}, 3)")
        if offsets.shape != idx.shape:
            raise ValueError(f"offsets shape {offsets.shape} != {idx.shape}")
        self.idx = idx
        self.normals = normals
        self.offsets = offsets
        self.k = float(k)
        self._colors: np.ndarray | None = None  # lazy

    def _project(
        self,
        P: np.ndarray,
        W: np.ndarray,
        i: np.ndarray,
        n: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros_like(P)
        if i.shape[0] == 0:
            return out
        # C(p) = n · p - offset; only project when C < 0 (still penetrating).
        C = np.einsum("ij,ij->i", P[i], n) - offsets
        active = (C < 0.0) & (W[i] > 0.0)
        if not active.any():
            return out
        # Single-vertex constraint with |∇C| = 1: λ = C / (w·1) gives
        # Δp = -λ · w · n = -C · n. The mass cancels, so the projection
        # always lands exactly on the surface in one shot regardless of W.
        np.add.at(out, i[active], (-C[active])[:, None] * n[active])
        return out

    def project_batch(self, P: np.ndarray, W: np.ndarray) -> np.ndarray:
        return self._project(P, W, self.idx, self.normals, self.offsets)

    # ------------------------------------------------------- Gauss–Seidel

    def _ensure_colors(self) -> np.ndarray:
        if self._colors is None:
            # idx is (M,); reshape to (M, 1) so greedy_color can run.
            self._colors = greedy_color(self.idx.reshape(-1, 1))
        return self._colors

    @property
    def n_colors(self) -> int:
        c = self._ensure_colors()
        return int(c.max() + 1) if c.size else 1

    def project_color(self, P: np.ndarray, W: np.ndarray, c: int) -> np.ndarray:
        colors = self._ensure_colors()
        mask = colors == c
        return self._project(P, W, self.idx[mask], self.normals[mask], self.offsets[mask])


# ------------------------------------------- per-step constraint generation


def generate_collision_constraints(
    X: np.ndarray,
    P: np.ndarray,
    colliders,
    free_mask: np.ndarray,
    k: float = 1.0,
    skin: float = 0.0,
) -> CollisionGroup:
    """Build a CollisionGroup from the X→P predicted move.

    Each collider's ``contact_anchors`` method decides for itself how to
    detect penetration:

    * ``Plane`` / ``Sphere`` use a closed-form SDF on ``P`` (X ignored).
    * ``TriangleMesh`` uses continuous-collision ray-casting along the
      segment ``X_i → P_i`` (paper §3.4): a hit pinpoints the *entry*
      surface point and face normal, and the resulting half-space
      constraint pushes the particle back to the entry side.

    All colliders return ``(idx, normals, offsets)`` so this function
    just stitches the per-collider results together.

    Parameters
    ----------
    X : (N, 3) float64
        Pre-step positions (ray origins for CCD; ignored by SDF colliders).
    P : (N, 3) float64
        Predicted positions after the gravity/predict step.
    colliders : iterable of Plane / Sphere / TriangleMesh.
    free_mask : (N,) bool
        Only generate constraints for particles with W > 0; pinned ones
        cannot move and would just create degenerate constraints.
    skin : float
        Contact-detection tolerance. For SDF colliders, a vertex with
        ``sdf < skin`` (i.e. either inside the surface or within ``skin``
        of it) gets a constraint generated. For the CCD mesh collider,
        the ray length is extended by ``skin`` so a near-miss within the
        skin band also acquires a constraint. Constraints with the
        particle still outside are "armed but inactive" — the projection
        only fires when C < 0 — giving contact hysteresis. Recommended
        ~1e-3 for cloth on smooth obstacles.
    """
    idx_list, n_list, off_list = [], [], []
    for c in colliders:
        out = c.contact_anchors(X, P, free_mask, skin)
        if out is None:
            continue
        idx, n, offsets = out
        idx_list.append(idx)
        n_list.append(n)
        off_list.append(offsets)

    if not idx_list:
        return CollisionGroup(
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            k=k,
        )

    return CollisionGroup(
        np.concatenate(idx_list),
        np.concatenate(n_list, axis=0),
        np.concatenate(off_list),
        k=k,
    )
