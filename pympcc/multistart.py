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

Parallel execution
------------------
``n_jobs`` controls process-level parallelism: ``1`` (default) runs
sequentially, ``-1`` resolves to ``os.cpu_count()``, any other positive
integer fans out across that many workers.  The pool uses Python's
``spawn`` start method so cyipopt's IPOPT global state is freshly
initialised in each worker.

Picklability requirement: all ``MPCCProblem`` callables (``objective``,
``gradient``, the complementarity callbacks, etc.) must be picklable
when ``n_jobs != 1``.  Plain ``def``-defined module-level functions are
fine; bare ``lambda`` and locally-defined closures are not.  Build
problems out of named functions if you intend to parallelise.

Example
-------
>>> ms = pympcc.multistart(problem, n_starts=8, perturb_scale=0.2)
>>> ms.best.x          # best objective among successful runs
>>> ms.unique_optima() # distinct local optima discovered
>>> ms = pympcc.multistart(problem, n_starts=16, n_jobs=-1)  # parallel
"""
from __future__ import annotations

import copy
import logging
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np

from .models import StructuredMPCC

if TYPE_CHECKING:
    from .problem import MPCCProblem
    from .result import MPCCResult

__all__ = ["multistart", "MultiStartResult"]

_log = logging.getLogger(__name__)

_ProblemLike = Union["MPCCProblem", StructuredMPCC]


def _resolve_n_jobs(n_jobs: int) -> int:
    """Translate ``n_jobs`` into a concrete worker count (>= 1)."""
    if n_jobs == 0:
        raise ValueError("n_jobs must be >= 1 or -1 (got 0)")
    if n_jobs < -1:
        raise ValueError(f"n_jobs must be >= 1 or -1 (got {n_jobs})")
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    return int(n_jobs)


def _multistart_worker(problem: "MPCCProblem",
                       xk: np.ndarray,
                       solve_kwargs: dict,
                       worker_seed: int):
    """Top-level worker for ProcessPoolExecutor (must be picklable).

    The worker seeds NumPy's default global RNG from *worker_seed* so any
    randomised pass inside :func:`pympcc.solve` (autoscale probes, FD
    perturbations) produces the same result for the same
    ``(multistart_seed, start_index)`` pair regardless of completion
    order or worker count.  Each worker is its own process under the
    spawn-based pool, so the global seed mutation does not leak.

    The problem is shallow-copied before its ``x0`` is overwritten, so
    the user's :class:`MPCCProblem` remains untouched even if the
    underlying ``solve`` retains a reference.
    """
    from .solver import solve as _solve

    np.random.seed(worker_seed)
    local_problem = copy.copy(problem)
    local_problem.x0 = xk
    return _solve(local_problem, **solve_kwargs)


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
        ``n_starts`` if any starts raised an exception (those are
        recorded in ``failures`` instead).
    failures : list of (int, str, str)
        Tuples ``(start_index, exception_type_name, message)`` for every
        start whose ``solve()`` raised.  Empty list when all starts
        completed (success or otherwise).

    The shortcut attributes ``x``, ``obj``, ``success``, ``comp_residual``
    proxy ``best`` so callers that don't care about diversity can use the
    object as a drop-in replacement for an :class:`MPCCResult`.
    """

    best: "MPCCResult"
    runs: list["MPCCResult"] = field(default_factory=list)
    failures: list[tuple[int, str, str]] = field(default_factory=list)

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
    n_jobs: int = 1,
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
    n_jobs : int, default 1
        Worker count for parallel execution.  ``1`` (default) runs
        sequentially in-process; ``-1`` resolves to ``os.cpu_count()``;
        any other positive integer fans out across that many processes
        via ``ProcessPoolExecutor`` with the ``spawn`` start method.
        When ``n_jobs != 1`` all problem callables must be picklable
        (no bare lambdas / local closures).
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
    workers = _resolve_n_jobs(n_jobs)

    if isinstance(problem, StructuredMPCC):
        problem = problem.to_mpcc_problem()

    x0 = np.asarray(problem.x0, dtype=float)
    span = np.maximum(np.abs(x0), 1.0) * float(perturb_scale)
    xl = (np.asarray(problem.xl, dtype=float)
          if problem.xl is not None else np.full(problem.n, -np.inf))
    xu = (np.asarray(problem.xu, dtype=float)
          if problem.xu is not None else np.full(problem.n, np.inf))

    # Pre-compute every start vector deterministically in the parent so
    # that completion order in the parallel path doesn't perturb the seed
    # stream — same (seed, n_starts) yields the same starts regardless of
    # n_jobs.
    seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seq)
    starts: list[np.ndarray] = []
    for k in range(n_starts):
        if k == 0:
            xk = x0.copy()
        else:
            xk = x0 + rng.normal(scale=span)
            np.maximum(xk, xl, out=xk)
            np.minimum(xk, xu, out=xk)
        starts.append(xk)

    # Per-start worker seeds derived from the same SeedSequence so that
    # randomised passes inside ``solve`` (autoscale, FD probes) reproduce
    # exactly across re-runs, regardless of completion order.  Folded
    # into a 32-bit int because :func:`numpy.random.seed` requires that.
    worker_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seq.spawn(n_starts)
    ]

    if workers == 1:
        runs, failures = _run_sequential(problem, starts, solve_kwargs)
    else:
        runs, failures = _run_parallel(
            problem, starts, workers, solve_kwargs, worker_seeds,
        )

    if failures:
        warnings.warn(
            f"multistart: {len(failures)}/{n_starts} starts raised "
            f"exceptions during solve; see MultiStartResult.failures "
            f"for the (start_index, exception_type, message) tuples.",
            RuntimeWarning,
            stacklevel=2,
        )

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
    return MultiStartResult(best=best, runs=runs, failures=failures)


def _run_sequential(problem, starts, solve_kwargs):
    from .solver import solve as _solve  # local: avoids circular import

    runs: list["MPCCResult"] = []
    failures: list[tuple[int, str, str]] = []
    for k, xk in enumerate(starts):
        # Shallow-copy so we don't mutate the user's problem.x0 in place.
        # All callable / array fields are shared by reference; only the
        # ``x0`` attribute on the copy diverges.
        local_problem = copy.copy(problem)
        local_problem.x0 = xk
        try:
            runs.append(_solve(local_problem, **solve_kwargs))
        except Exception as exc:
            _log.debug("multistart: start %d failed, %s: %s",
                       k, type(exc).__name__, exc)
            failures.append((k, type(exc).__name__, str(exc)))
    return runs, failures


def _run_parallel(problem, starts, workers, solve_kwargs, worker_seeds):
    """Fan starts out to a spawn-based ProcessPoolExecutor.

    Each worker receives its own pickled copy of *problem* and a
    deterministic *worker_seed* so randomised internal passes
    (autoscale, FD probes) reproduce across runs.  The worker
    shallow-copies the problem before setting its ``x0``, so neither
    the parent's nor the pickled worker copy's external view of
    ``problem`` is mutated.  Results are collected in start order
    (the index travels with each future) so callers see the same
    ordering as the sequential path.
    """
    ctx = multiprocessing.get_context("spawn")
    runs: list[tuple[int, "MPCCResult"]] = []
    failures: list[tuple[int, str, str]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(
                _multistart_worker, problem, xk, solve_kwargs, worker_seeds[k]
            ): k
            for k, xk in enumerate(starts)
        }
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                runs.append((k, fut.result()))
            except Exception as exc:
                _log.debug("multistart: start %d failed in worker, %s: %s",
                           k, type(exc).__name__, exc)
                failures.append((k, type(exc).__name__, str(exc)))
    runs.sort(key=lambda kv: kv[0])
    return [r for _, r in runs], failures
