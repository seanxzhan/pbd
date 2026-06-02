"""Static colliders — verification #10 + plane/sphere coverage."""
from __future__ import annotations

import numpy as np

from pbd import Plane, Sphere, Stretch, System, TriangleMesh


def _single_particle(pos, v=(0.0, 0.0, 0.0), gravity=(0.0, 0.0, -9.81)):
    sys = System(np.array([pos], dtype=np.float64), np.array([1.0]), gravity=gravity)
    sys.V[0] = v
    return sys


# ----------------------------------------------------------- plane geometry


def test_plane_signed_distance_basic():
    p = Plane(normal=(0.0, 0.0, 1.0), offset=0.0)
    P = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -0.5], [0.0, 0.0, 0.0]])
    sd = p.signed_distance(P)
    np.testing.assert_allclose(sd, [1.0, -0.5, 0.0])


def test_plane_normalizes_input_normal():
    p = Plane(normal=(0.0, 0.0, 5.0), offset=0.0)
    np.testing.assert_allclose(np.linalg.norm(p.normal), 1.0)


# ----------------------------------------------------------- sphere geometry


def test_sphere_signed_distance_basic():
    s = Sphere(center=np.zeros(3), radius=1.0)
    P = np.array([[2.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sd = s.signed_distance(P)
    np.testing.assert_allclose(sd, [1.0, -0.5, 0.0])


def test_sphere_outward_normals():
    s = Sphere(center=np.zeros(3), radius=1.0)
    P = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    n = s.surface_normals(P)
    np.testing.assert_allclose(n, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


# ----------------------------------- (test #10) particle settles on plane


def test_particle_drops_to_plane_and_stops():
    """Drop a single particle onto a flat floor at z=0; after enough damped
    steps it must rest exactly on the plane (signed dist ≥ 0, ~0)."""
    sys = _single_particle(pos=(0.0, 0.0, 0.5))
    sys.add_collider(Plane(normal=(0.0, 0.0, 1.0), offset=0.0))

    # restitution=0 ⇒ velocity reflects with no rebound; it'll just stop.
    for _ in range(500):
        sys.step(dt=1e-2, iters=1, restitution=0.0, friction=0.0)

    z = sys.X[0, 2]
    assert z >= -1e-12, f"particle below floor: z={z}"
    assert z < 1e-9, f"particle should rest on z=0; got z={z}"


def test_particle_with_restitution_bounces():
    """With restitution=1 the particle bounces back at the same speed it
    arrived with (elastic collision)."""
    sys = _single_particle(pos=(0.0, 0.0, 1.0), v=(0.0, 0.0, -2.0),
                            gravity=(0.0, 0.0, 0.0))
    sys.add_collider(Plane(normal=(0.0, 0.0, 1.0), offset=0.0))

    sys.step(dt=1.0, iters=1, restitution=1.0, friction=0.0)
    # Position lands on the plane; velocity = -ε · v_pre_n = +2 (ε=1).
    np.testing.assert_allclose(sys.X[0, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(sys.V[0, 2], 2.0, atol=1e-12)


def test_particle_with_zero_restitution_stops_normal_velocity():
    """ε=0 ⇒ no normal rebound; the projection's spurious outward velocity
    must also be zeroed."""
    sys = _single_particle(pos=(0.0, 0.0, 1.0), v=(0.0, 0.0, -2.0),
                            gravity=(0.0, 0.0, 0.0))
    sys.add_collider(Plane(normal=(0.0, 0.0, 1.0), offset=0.0))

    sys.step(dt=1.0, iters=1, restitution=0.0, friction=0.0)
    np.testing.assert_allclose(sys.X[0, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(sys.V[0, 2], 0.0, atol=1e-12)


def test_friction_brings_tangential_velocity_to_zero():
    """A particle sliding along a floor with friction=large halts in one step."""
    sys = _single_particle(pos=(0.0, 0.0, -0.05), v=(2.0, 0.0, -1.0),
                            gravity=(0.0, 0.0, 0.0))
    sys.add_collider(Plane(normal=(0.0, 0.0, 1.0), offset=0.0))

    sys.step(dt=1e-2, iters=1, restitution=0.0, friction=100.0)
    # Friction strong enough to fully arrest tangential v.
    assert abs(sys.V[0, 0]) < 1e-9, f"V_x should be ~0 with high friction; got {sys.V[0, 0]}"


# -------------------------------------------------- sphere collision tests


def test_particle_pushed_off_sphere_surface():
    """Place a particle inside a sphere; one step must push it onto the
    surface."""
    sys = _single_particle(pos=(0.3, 0.0, 0.0), gravity=(0.0, 0.0, 0.0))
    sys.add_collider(Sphere(center=np.zeros(3), radius=1.0))

    sys.step(dt=1e-2, iters=1)
    r = float(np.linalg.norm(sys.X[0]))
    assert r >= 1.0 - 1e-9, f"particle still inside sphere: r={r}"
    assert r < 1.0 + 1e-9, f"particle pushed outside sphere: r={r}"


def test_pinned_particle_unaffected_by_collider():
    """A pinned vertex should not be moved by collision projection even if
    'inside' a collider."""
    sys = _single_particle(pos=(0.0, 0.0, -0.5))
    sys.pin([0])
    sys.add_collider(Plane(normal=(0.0, 0.0, 1.0), offset=0.0))

    x0 = sys.X[0].copy()
    sys.step(dt=1e-3, iters=10)
    np.testing.assert_array_equal(sys.X[0], x0)


# ------------------------------------------------------- mesh + collider


def test_cloth_drape_settles_on_plane():
    """A small triangle pinned at one vertex, falling under gravity onto
    a plane: after enough steps it must rest above the plane."""
    V = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.5, 1.0, 1.0],
    ])
    masses = np.array([1.0, 1.0, 1.0])
    sys = System(V, masses, gravity=(0.0, 0.0, -9.81))
    sys.add_constraint(Stretch(np.array([[0, 1], [1, 2], [2, 0]]),
                                np.array([1.0, np.sqrt(1.25), np.sqrt(1.25)]),
                                k=1.0))
    sys.add_collider(Plane(normal=(0.0, 0.0, 1.0), offset=0.0))

    for _ in range(500):
        sys.step(dt=1e-2, iters=20, k_damp=0.5)

    # All three vertices must be on or above the plane.
    assert (sys.X[:, 2] >= -1e-9).all(), f"penetration: zs={sys.X[:, 2]}"


# ----------------------------------------------------- triangle mesh CCD


def _unit_quad_xy_plane():
    """One axis-aligned quad in the z=0 plane, two CCW triangles, normal +z."""
    V = np.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ])
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return V, F


def test_trianglemesh_validates_shape():
    import pytest

    with pytest.raises(ValueError):
        TriangleMesh(np.zeros((3, 2)), np.zeros((1, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        TriangleMesh(np.zeros((3, 3)), np.zeros((1, 4), dtype=np.int64))


def test_trianglemesh_rejects_degenerate_face():
    import pytest

    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    F = np.array([[0, 1, 2]], dtype=np.int64)        # collinear → zero area
    with pytest.raises(ValueError):
        TriangleMesh(V, F)


def test_trianglemesh_particle_falls_onto_quad():
    """Particle starts above the quad and falls; CCD must catch the segment
    crossing and push it back to z >= 0."""
    V, F = _unit_quad_xy_plane()
    sys = _single_particle(pos=(0.0, 0.0, 0.5))   # gravity = (0,0,-9.81)
    sys.add_collider(TriangleMesh(V, F))

    for _ in range(300):
        sys.step(dt=1e-2, iters=1, restitution=0.0, friction=0.0)

    z = sys.X[0, 2]
    assert z >= -1e-9, f"penetration through mesh: z={z}"
    assert z < 1e-6, f"particle should rest on z=0; got z={z}"


def test_trianglemesh_no_hit_when_particle_misses():
    """Particle moves but stays clear of the (small) quad — no collision
    constraint should be generated, so motion is pure ballistics."""
    V, F = _unit_quad_xy_plane()
    # Move particle far off to one side so its segment doesn't cross the quad.
    sys = _single_particle(pos=(5.0, 5.0, 0.5), gravity=(0.0, 0.0, 0.0))
    sys.V[0] = (0.0, 0.0, -1.0)
    sys.add_collider(TriangleMesh(V, F))

    sys.step(dt=1e-2, iters=1)
    # Predicted z = 0.5 + dt * -1 = 0.49; no collision, position unchanged
    # by projection.
    np.testing.assert_allclose(sys.X[0], [5.0, 5.0, 0.49], atol=1e-12)


def test_trianglemesh_auto_orients_normal():
    """Approach the quad from below (-z side): the face normal would be
    +z by winding, but auto-orient must flip it to -z to push us back."""
    V, F = _unit_quad_xy_plane()
    sys = _single_particle(pos=(0.0, 0.0, -0.5), gravity=(0.0, 0.0, 0.0))
    sys.V[0] = (0.0, 0.0, 5.0)                    # heading +z, will pierce quad
    sys.add_collider(TriangleMesh(V, F))

    sys.step(dt=1e-1, iters=1, restitution=0.0, friction=0.0)
    # Should be pushed back to z ≈ 0 from below (z ≤ 0).
    assert sys.X[0, 2] <= 1e-9, (
        f"auto-orient failed; particle pushed past mesh to z={sys.X[0, 2]}"
    )


def test_trianglemesh_matches_plane_for_planar_obstacle():
    """A flat triangulated quad in the z=0 plane should behave just like
    a Plane(normal=+z, offset=0) for a falling particle."""
    V, F = _unit_quad_xy_plane()

    sys_mesh = _single_particle(pos=(0.0, 0.0, 0.5))
    sys_mesh.add_collider(TriangleMesh(V, F))

    sys_plane = _single_particle(pos=(0.0, 0.0, 0.5))
    sys_plane.add_collider(Plane(normal=(0.0, 0.0, 1.0), offset=0.0))

    for _ in range(200):
        sys_mesh.step(dt=1e-2, iters=1, restitution=0.0, friction=0.0)
        sys_plane.step(dt=1e-2, iters=1, restitution=0.0, friction=0.0)

    # Both should rest at the same z (within 1e-9).
    np.testing.assert_allclose(sys_mesh.X[0, 2], sys_plane.X[0, 2], atol=1e-9)


def test_trianglemesh_pinned_particle_unaffected():
    """A pinned vertex must not be moved by CCD even if its segment crosses
    the mesh — we never generate constraints for W=0 verts."""
    V, F = _unit_quad_xy_plane()
    sys = _single_particle(pos=(0.0, 0.0, 0.5))
    sys.V[0] = (0.0, 0.0, -10.0)                  # would pierce quad
    sys.pin([0])
    sys.add_collider(TriangleMesh(V, F))

    x0 = sys.X[0].copy()
    sys.step(dt=1e-1, iters=1)
    np.testing.assert_array_equal(sys.X[0], x0)
