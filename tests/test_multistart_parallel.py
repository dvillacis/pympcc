"""Tests for the parallel (n_jobs) path of :func:`pympcc.multistart` (§6.1).

The ``spawn`` start method pickles the problem into each worker, so all
callables live at module scope (named ``def``s, never lambdas) — that's
what real users will need to do too.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import pympcc
from pympcc.multistart import (
    MultiStartResult,
    _resolve_n_jobs,
    multistart,
)


# ======================================================================= #
# Picklable problem builder (top-level functions, not lambdas)             #
# ======================================================================= #

def _obj(x):
    return float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2)


def _grad(x):
    return np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)])


def _comp_G(x):
    return np.array([x[0]])


def _comp_G_jac(x):
    return np.array([[1.0, 0.0]])


def _comp_H(x):
    return np.array([x[1]])


def _comp_H_jac(x):
    return np.array([[0.0, 1.0]])


def _picklable_problem(x0=None):
    """Convex MPCC built from top-level functions — survives spawn pickling."""
    if x0 is None:
        x0 = np.array([0.5, 0.5])
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.asarray(x0, dtype=float),
        objective=_obj,
        gradient=_grad,
        comp_G=_comp_G,
        comp_G_jacobian=_comp_G_jac,
        comp_H=_comp_H,
        comp_H_jacobian=_comp_H_jac,
    )


def _lambda_problem():
    """Convex MPCC built from lambdas — cannot survive spawn pickling."""
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2),
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


# ======================================================================= #
# n_jobs resolution                                                         #
# ======================================================================= #

class TestResolveNJobs:
    def test_one_passthrough(self):
        assert _resolve_n_jobs(1) == 1

    def test_positive(self):
        assert _resolve_n_jobs(4) == 4

    def test_minus_one_uses_cpu_count(self):
        assert _resolve_n_jobs(-1) == max(1, os.cpu_count() or 1)

    def test_zero_rejected(self):
        with pytest.raises(ValueError, match=">= 1 or -1"):
            _resolve_n_jobs(0)

    def test_negative_other_rejected(self):
        with pytest.raises(ValueError, match=">= 1 or -1"):
            _resolve_n_jobs(-2)


# ======================================================================= #
# Parallel == sequential                                                    #
# ======================================================================= #

class TestParallelEquivalence:
    def test_n_jobs_one_unchanged(self):
        # The default sequential path must remain bit-for-bit identical.
        ms_default = multistart(_picklable_problem(), n_starts=4, seed=0)
        ms_explicit = multistart(_picklable_problem(),
                                 n_starts=4, seed=0, n_jobs=1)
        assert len(ms_default.runs) == len(ms_explicit.runs) == 4
        for r1, r2 in zip(ms_default.runs, ms_explicit.runs):
            np.testing.assert_allclose(r1.x, r2.x, rtol=1e-10, atol=1e-12)

    def test_parallel_matches_sequential_on_convex(self):
        # Same seed → same starts → same converged points (convex problem).
        seq = multistart(_picklable_problem(), n_starts=4,
                         perturb_scale=0.3, seed=42, n_jobs=1)
        par = multistart(_picklable_problem(), n_starts=4,
                         perturb_scale=0.3, seed=42, n_jobs=2)
        assert len(seq.runs) == len(par.runs) == 4
        # Convex: every start should converge to the same global optimum.
        for r in par.runs:
            assert r.success
            np.testing.assert_allclose(r.x, [2.0, 0.0], atol=1e-3)
        # Best objective should agree.
        assert par.best.obj == pytest.approx(seq.best.obj, rel=1e-6)


# ======================================================================= #
# Result ordering                                                           #
# ======================================================================= #

class TestDeterminismParallel:
    def test_x0_unchanged_after_parallel_call(self):
        # The parent must not mutate problem.x0 when fanning out to workers.
        # Each worker shallow-copies the problem before overwriting x0.
        p = _picklable_problem(x0=np.array([0.42, 0.31]))
        original = p.x0.copy()
        multistart(p, n_starts=3, perturb_scale=0.4, seed=0, n_jobs=2)
        np.testing.assert_array_equal(p.x0, original)

    def test_seed_determinism_parallel(self):
        # Same (seed, n_starts, n_jobs) must yield bit-identical iterates
        # across two independent multistart invocations, regardless of
        # worker completion order.  The per-start worker_seeds derive from
        # SeedSequence(seed).spawn(n_starts) so they're stable.
        p1 = _picklable_problem()
        p2 = _picklable_problem()
        ms1 = multistart(p1, n_starts=4, perturb_scale=0.5, seed=42, n_jobs=2)
        ms2 = multistart(p2, n_starts=4, perturb_scale=0.5, seed=42, n_jobs=2)
        for r1, r2 in zip(ms1.runs, ms2.runs):
            np.testing.assert_allclose(r1.x, r2.x, rtol=1e-8, atol=1e-10)

    def test_seed_determinism_matches_sequential(self):
        # The choice of n_jobs (1 vs >1) must not change the per-start
        # iterate.  Cross-checks that worker_seed plumbing is identical
        # on the sequential and parallel paths.
        p_seq = _picklable_problem()
        p_par = _picklable_problem()
        ms_seq = multistart(p_seq, n_starts=4, perturb_scale=0.5,
                            seed=7, n_jobs=1)
        ms_par = multistart(p_par, n_starts=4, perturb_scale=0.5,
                            seed=7, n_jobs=2)
        for r_s, r_p in zip(ms_seq.runs, ms_par.runs):
            np.testing.assert_allclose(r_s.x, r_p.x, rtol=1e-8, atol=1e-10)


class TestOrdering:
    def test_runs_in_start_order(self):
        # The k=0 start uses x0 verbatim; perturbed starts come after.
        # Even though futures complete out-of-order, runs[0] must be the
        # k=0 start.
        p = _picklable_problem(x0=np.array([0.7, 0.3]))
        ms = multistart(p, n_starts=4, perturb_scale=0.5,
                        seed=0, n_jobs=2)
        # Compare runs[0] against a single solve from the same x0.
        r0 = pympcc.solve(_picklable_problem(x0=np.array([0.7, 0.3])))
        np.testing.assert_allclose(ms.runs[0].x, r0.x, atol=1e-6)


# ======================================================================= #
# Top-level dispatch                                                        #
# ======================================================================= #

class TestSolveDispatchParallel:
    def test_solve_forwards_n_jobs(self):
        result = pympcc.solve(_picklable_problem(),
                              n_starts=3, n_jobs=2, multistart_seed=0)
        assert isinstance(result, MultiStartResult)
        assert len(result.runs) == 3
        assert result.success

    def test_solve_n_jobs_minus_one(self):
        # Don't actually fan out to all cores in CI — just verify dispatch.
        # A 2-start run with n_jobs=-1 will use min(2, cpu_count) workers.
        result = pympcc.solve(_picklable_problem(),
                              n_starts=2, n_jobs=-1, multistart_seed=0)
        assert isinstance(result, MultiStartResult)
        assert result.success


# ======================================================================= #
# Picklability                                                              #
# ======================================================================= #

class TestPicklability:
    def test_lambda_problem_fails_under_parallel(self):
        # Bare lambdas don't survive spawn pickling.  The pool raises and
        # multistart converts every-start-failed into a RuntimeError.
        # (Either path — pickle error in submit, or every future raising —
        # the user-visible outcome is a clear failure, not a silent hang.)
        p = _lambda_problem()
        with pytest.raises((RuntimeError, AttributeError, TypeError)):
            multistart(p, n_starts=2, n_jobs=2, seed=0)
