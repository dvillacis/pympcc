"""Unit tests for the per-NLP-iter ``inner_callback`` hook (§4.8)."""

from __future__ import annotations

import numpy as np
import pytest

import pympcc

from .macmpec_problems import PROBLEM_NAMES

SIMPLE = PROBLEM_NAMES["simple"]

# Sparse version of SIMPLE: comp_G/H Jacobians return 1-D nnz values; this
# routes the build through _SparseNLP, which is the second adapter that needs
# the inner_callback hook.
SIMPLE_SPARSE = pympcc.MPCCProblem(
    n=2, n_comp=1,
    x0=np.array([0.5, 0.5]),
    objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
    gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
    comp_G=lambda x: np.array([x[0]]),
    comp_G_jacobian=lambda x: np.array([1.0]),
    comp_G_jacobian_sparsity=(np.array([0]), np.array([0])),
    comp_H=lambda x: np.array([x[1]]),
    comp_H_jacobian=lambda x: np.array([1.0]),
    comp_H_jacobian_sparsity=(np.array([0]), np.array([1])),
)

_EXPECTED_KEYS = {
    "alg_mod", "obj_value", "inf_pr", "inf_du", "mu", "d_norm",
    "regularization_size", "alpha_du", "alpha_pr", "ls_trials",
}


class TestInnerCallbackDense:
    def test_fires_at_least_once(self):
        calls: list[tuple[int, dict]] = []

        def cb(it, info):
            calls.append((it, dict(info)))
            return True

        result = pympcc.solve(SIMPLE.problem, strategy="scholtes",
                              inner_callback=cb, max_iter=3)
        assert result.success
        assert len(calls) >= 1

    def test_info_dict_keys_and_types(self):
        snapshots: list[dict] = []

        def cb(it, info):
            snapshots.append(info)
            return True

        pympcc.solve(SIMPLE.problem, strategy="scholtes",
                     inner_callback=cb, max_iter=2)
        assert snapshots, "callback never fired"
        for info in snapshots:
            assert set(info.keys()) == _EXPECTED_KEYS
            for k in _EXPECTED_KEYS - {"alg_mod", "ls_trials"}:
                assert isinstance(info[k], float)
            assert isinstance(info["alg_mod"], int)
            assert isinstance(info["ls_trials"], int)

    def test_iter_count_monotonic_within_solve(self):
        # iter_count resets at the start of each outer NLP solve.  Within a
        # single solve it must be monotone non-decreasing.
        calls: list[int] = []

        def cb(it, info):
            calls.append(it)
            return True

        pympcc.solve(SIMPLE.problem, strategy="scholtes",
                     inner_callback=cb, max_iter=1)
        assert calls == sorted(calls)

    def test_default_no_op(self):
        # Smoke-test: with no inner_callback the solver still works.
        result = pympcc.solve(SIMPLE.problem, strategy="scholtes", max_iter=3)
        assert result.success

    def test_direct_strategy_dense(self):
        # DirectStrategy uses BaseStrategy.__init__ unchanged; the hook still
        # has to flow through.
        calls = []
        pympcc.solve(SIMPLE.problem, strategy="direct",
                     inner_callback=lambda it, info: calls.append(it) or True)
        assert calls, "DirectStrategy never invoked the inner_callback"


class TestInnerCallbackSparse:
    def test_fires_with_sparse_problem(self):
        # Routes through _SparseNLP — the adapter that the docstring blurb
        # forgot to mention.
        calls = []

        def cb(it, info):
            calls.append((it, info["obj_value"]))
            return True

        result = pympcc.solve(SIMPLE_SPARSE, strategy="scholtes",
                              inner_callback=cb, max_iter=3)
        assert result.success
        assert calls, "inner_callback never fired on sparse path"

    def test_slack_strategy_sparse(self):
        # The slack strategy always builds a _SparseNLP, regardless of how
        # the user supplied the comp Jacobians.
        calls = []
        result = pympcc.solve(SIMPLE.problem, strategy="slack",
                              inner_callback=lambda it, info:
                                  calls.append(it) or True,
                              max_iter=3)
        assert result.success
        assert calls, "slack strategy never invoked the inner_callback"


class TestInnerCallbackEarlyStop:
    def test_returning_false_stops_inner_solve(self):
        # When the callback returns False, IPOPT halts the inner solve;
        # n_ipopt_iter should be very small (typically 1 or 2).
        n_calls = {"n": 0}

        def cb(it, info):
            n_calls["n"] += 1
            return False  # halt immediately

        result = pympcc.solve(SIMPLE.problem, strategy="scholtes",
                              inner_callback=cb, max_iter=1)
        # Callback fires at least once before IPOPT processes the False.
        assert n_calls["n"] >= 1
        # The first inner solve aborted early — history should reflect a
        # short inner run rather than full convergence.
        assert len(result.history) >= 1
        assert result.history[0].n_ipopt_iter <= 5

    def test_returning_none_treated_as_continue(self):
        # Common user mistake: forgetting the return.  Should not stop.
        calls = []

        def cb(it, info):
            calls.append(it)
            # implicit return None

        result = pympcc.solve(SIMPLE.problem, strategy="scholtes",
                              inner_callback=cb, max_iter=2)
        assert result.success
        # Several iterations should have run despite the None returns.
        assert len(calls) >= 2


class TestInnerCallbackInteractsWithOuter:
    def test_outer_and_inner_both_fire(self):
        outer = []
        inner = []

        def outer_cb(k, info):
            outer.append(k)

        def inner_cb(it, info):
            inner.append(it)
            return True

        pympcc.solve(SIMPLE.problem, strategy="scholtes",
                     callback=outer_cb, inner_callback=inner_cb, max_iter=2)
        assert outer, "outer callback didn't fire"
        assert inner, "inner callback didn't fire"
        # Inner should fire many more times than outer (per-NLP-iter vs
        # per-NLP-solve).
        assert len(inner) >= len(outer)
