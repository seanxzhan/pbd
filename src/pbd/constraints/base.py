"""Batch-oriented constraint group interface.

The performance-critical contract: a constraint *group* owns an
``(M, n)`` index array and projects all M constraints in one numpy
call. Per-constraint Python objects are 100× too slow at 8k verts.

``project_batch`` returns the *raw* ΔP for a Jacobi sweep. The system
applies the ``k' = 1 - (1 - k)^{1/iters}`` linearization (paper §3.3)
outside.

For graph-colored Gauss–Seidel, groups optionally expose an additional
pair of methods:

* ``n_colors`` — number of color classes (default 1: the whole group is
  one class, equivalent to Jacobi).
* ``project_color(P, W, c)`` — project just the constraints of color c.
  Default delegates to ``project_batch`` when c == 0 and returns zeros
  otherwise; groups that benefit from coloring (Stretch, Bend, Collision)
  override both.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ConstraintGroup(ABC):
    """A vectorized batch of constraints of one type."""

    k: float  # stiffness in [0, 1]

    # ------------------------------------------------------------------ Jacobi

    @abstractmethod
    def project_batch(self, P: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute the position correction for this constraint batch.

        Parameters
        ----------
        P : (N, 3) float64
            Predicted positions (read-only).
        W : (N,) float64
            Inverse masses; ``W[i] == 0`` ⇒ pinned.

        Returns
        -------
        dP : (N, 3) float64
            Raw position correction. The caller multiplies by k_eff.
        """

    # ------------------------------------------------------------- Gauss–Seidel

    @property
    def n_colors(self) -> int:
        """Number of color classes. Default 1: the whole group is one class."""
        return 1

    def project_color(self, P: np.ndarray, W: np.ndarray, c: int) -> np.ndarray:
        """Project just the constraints of color ``c``.

        Default implementation: groups without coloring fold into a single
        class (c=0 is the whole batch). Override in subclasses to expose
        true vertex-disjoint color classes for Gauss–Seidel.
        """
        if c == 0:
            return self.project_batch(P, W)
        return np.zeros_like(P)
