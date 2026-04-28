"""Multi-start wrapper for MPCC solves (§3.3).

Local solvers for MPCCs are sensitive to the starting iterate: different
basins can yield different stationary points (and even different active
sets).  :func:`multistart` runs :func:`pympcc.solve` from ``n_starts``
randomly perturbed initial points and returns the best successful run
along with the full list, exposing single-start bias to the caller.

The first start uses ``problem.x0`` verbatim; subsequent starts perturb
each coordinate by Gaussian noise with standard deviation
``perturb_scale * max(|x0|, 1)`` and clip to ``[xl, xu]`` when bounds
are present.

Example
-------
>>> ms = pympcc.multistart(problem, n_starts=8, perturb_scale=0.2)
>>> ms.best.x          # best objective among successful runs
>>> ms.unique_optima() # distinct local optima discovered
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np

from .models import StructuredMPCC

if TYPE_CHECKING:
    from .problem import MPCCProblem
    from .result import MPCCResult

__all__ = ["multistart", "MultiStartResult"]

_ProblemLike = Union["MPCCProblem", StructuredMPCC]


@dataclass
class MultiStartResult:
    """Aggregate result of a multi-start solve.

    Attributes
    ----------
    best : MPCCResult
        Result with the lowest objective among successful runs.  Falls
        back to the lowest-objective run overall if every start failed
        (in which case ``best.success`` is ``False``).
    runs : list of MPCCResult
        All per-start results in start order.  May be shorter than
        ``n_starts`` if any starts raised an exception (those are skipped).

    The shortcut attributes ``x``, ``obj``, ``success``, ``comp_residual``
    proxy ``best`` so callers that don't care about diversity can use the
    object as a drop-in replacement for an :class:`MPCCResult`.
    """

    best: "MPCCResult"
    runs: list["MPCCResult"] = field(default_factory=list)

    @property
    def x(self) -> np.ndarray:
        return self.best.x

    @property
    def obj(self) -> float:
        return self.best.obj

    @property
    def success(self) -> bool:
        return self.best.success

    @property
    def comp_residual(self) -> float:
        return self.best.comp_residual

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.runs if r.success)

    def unique_optima(
        self,
        *,
        atol_obj: float = 1e-6,
        atol_x: float = 1e-4,
    ) -> list["MPCCResult"]:
        """Cluster successful runs by closeness in (obj, x).

        Two runs share a cluster when their objectives differ by less
        than ``atol_obj`` *and* their iterates agree within ``atol_x``
        in the infinity norm.  The first run encountered in each
        cluster is returned, in start order.
        """
        succ = [r for r in self.runs if r.success]
        clusters: list["MPCCResult"] = []
        for r in succ:
            for c in clusters:
                if (abs(r.obj - c.obj) < atol_obj
                        and np.linalg.norm(r.x - c.x, np.inf) < atol_x):
                    break
            else:
                clusters.append(r)
        return clusters


def multistart(
    problem: _ProblemLike,
    *,
    n_starts: int = 16,
    perturb_scale: float = 0.1,
    seed: int = 0,
    **solve_kwargs,
) -> MultiStartResult:
    """Solve *problem* from *n_starts* perturbed initial points.

    Parameters
    ----------
    problem : MPCCProblem or StructuredMPCC
        Source problem.  ``problem.x0`` is the centre of the perturbation
        distribution and is restored on return (the function does not
        leave the problem mutated).
    n_starts : int, default 16
        Number of starting points.  Must be ``>= 1``.  Start 0 always
        uses ``problem.x0`` verbatim, so ``n_starts=1`` is equivalent to
        a plain :func:`pympcc.solve` call.
    perturb_scale : float, default 0.1
        Standard deviation of the Gaussian perturbation applied to each
        coordinate, expressed as a fraction of ``max(|x0|, 1)``.
    seed : int, default 0
        RNG seed for reproducible perturbations.
    **solve_kwargs
        Forwarded verbatim to :func:`pympcc.solve` (``strategy``,
        ``backend``, ``ipopt_options``, ``presolve``, ``autoscale``,
        strategy options, etc.).

    Returns
    -------
    MultiStartResult
    """
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")
    if perturb_scale <= 0.0:
        raise ValueError("perturb_scale must be > 0")

    from .solver import solve as _solve  # local import: avoids circular dep

    if isinstance(problem, StructuredMPCC):
        problem = problem.to_mpcc_problem()

    x0 = np.asarray(problem.x0, dtype=float)
    span = np.maximum(np.abs(x0), 1.0) * float(perturb_scale)
    xl = (np.asarray(problem.xl, dtype=float)
          if problem.xl is not None else np.full(problem.n, -np.inf))
    xu = (np.asarray(problem.xu, dtype=float)
          if problem.xu is not None else np.full(problem.n, np.inf))

    rng = np.random.default_rng(seed)
    runs: list["MPCCResult"] = []
    original_x0 = x0.copy()
    try:
        for k in range(n_starts):
            if k == 0:
                xk = x0.copy()
            else:
                xk = x0 + rng.normal(scale=span)
                np.maximum(xk, xl, out=xk)
                np.minimum(xk, xu, out=xk)
            problem.x0 = xk
            try:
                runs.append(_solve(problem, **solve_kwargs))
            except Exception:
                # Robustness: skip pathological starts and continue.
                continue
    finally:
        problem.x0 = original_x0

    if not runs:
        raise RuntimeError(
            "multistart: every start raised an exception during solve."
        )

    successful = [r for r in runs if r.success]
    pool = successful if successful else runs
    best = min(
        pool,
        key=lambda r: r.obj if r.obj is not None else float("inf"),
    )
    return MultiStartResult(best=best, runs=runs)
