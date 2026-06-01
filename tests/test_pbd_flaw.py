"""PBD's central flaw — verification #15.

PBD's stiffness is *not* a material parameter: at fixed k and iters, the
effective stiffness depends on (Δt, iters). Same scene visibly stiffens
at finer steps, even though "physically" the cloth is unchanged. This is
the failure that XPBD addresses (decouples stiffness from step size by
introducing a compliance parameter α).

The test here pins this dependence: a single mass on a Stretch constraint
falls under gravity, dropped through one full step. The fall distance
depends on Δt (finer Δt ⇒ less fall over the same wall-clock interval
because the constraint is solved more often relative to the integration).
This isn't a *bug* — it's a property the implementation should reproduce.
"""
from __future__ import annotations

import numpy as np

from pbd import Stretch, System


def _spring_drop(dt: float, iters: int, total_t: float, k: float):
    """Drop a mass at (1, 0, 0) anchored to a pinned (0,0,0) spring with
    rest length 1. Run for total_t seconds and return the steady-state
    extension (length - rest)."""
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sys = System(X, np.ones(2), gravity=(0.0, 0.0, -9.81))
    sys.pin([0])
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), k=k))

    n_steps = int(round(total_t / dt))
    for _ in range(n_steps):
        sys.step(dt=dt, iters=iters)
    return float(np.linalg.norm(sys.X[1])) - 1.0


def test_effective_stiffness_depends_on_dt_at_fixed_k_and_iters():
    """Same physical scene at four (dt, iters) settings — extension differs.

    With *constant* k and iters, halving dt halves the effective compliance
    (because the constraint is satisfied more times per second of physical
    time). Equivalently the cloth stiffens. This is exactly the flaw §3.3
    introduces the k' linearization to mitigate (and XPBD to fix entirely).
    """
    total_t = 1.0  # 1 second of physical time, settled
    iters = 10
    k = 0.05  # mid-soft so the dt-dependence is large enough to measure

    extensions = []
    for dt in (1.0 / 30, 1.0 / 60, 1.0 / 120, 1.0 / 240):
        ext = _spring_drop(dt=dt, iters=iters, total_t=total_t, k=k)
        extensions.append(ext)

    # Stiffening: finer dt ⇒ smaller extension (closer to a "stiffer" spring).
    # We check monotone-decreasing across the sweep.
    for a, b in zip(extensions, extensions[1:]):
        assert b < a, (
            f"expected finer dt to stiffen (smaller extension), got {extensions}"
        )

    # And the spread is large — a factor of >2 across the sweep, which is the
    # exact "non-physical" behaviour that motivates XPBD.
    assert extensions[0] / extensions[-1] > 2.0, (
        f"flaw should produce >2× spread; got {extensions}"
    )


def test_k_prime_linearization_makes_extension_iter_independent():
    """At fixed dt, varying iters must NOT change the equilibrium when
    System uses k' = 1 - (1-k)^(1/iters). This is the §3.3 fix.
    """
    dt = 1.0 / 60
    total_t = 1.0
    k = 0.3

    extensions = []
    for iters in (5, 10, 20, 40):
        ext = _spring_drop(dt=dt, iters=iters, total_t=total_t, k=k)
        extensions.append(ext)

    spread = max(extensions) - min(extensions)
    assert spread < 1e-6, (
        f"k' linearization should make iters-sweep flat; got {extensions}"
    )
