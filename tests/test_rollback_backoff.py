"""Tests for rollback ε backoff behaviour.

Verifies that on safeguard rejection the next inner solve sees a *larger*
ε (capped at the last accepted ε), preventing the previously-observed
infinite-loop pattern where the rollback retried an identical NLP.
"""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc.strategies._base import SAFEGUARD_DEFAULTS
from pympcc.strategies.scholtes import ScholtesStrategy


def _make_problem():
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


# ======================================================================= #
# Defaults                                                                  #
# ======================================================================= #

class TestDefaults:
    def test_eps_hold_factor_default(self):
        # Bug fix: default flipped from 1.0 to 3.0 so rollback genuinely
        # backs off rather than retrying the same ε.
        assert SAFEGUARD_DEFAULTS["eps_hold_factor"] == pytest.approx(3.0)

    def test_strategy_picks_up_default(self):
        p = _make_problem()
        strat = ScholtesStrategy(p, ipopt_options={}, safeguards="all")
        assert strat.eps_hold_factor == pytest.approx(3.0)


# ======================================================================= #
# Backoff trajectory                                                        #
# ======================================================================= #

class TestBackoffTrajectory:
    """Drive ``_run_epsilon_continuation`` with a stubbed ``_timed_solve``
    so we can dictate which inner solves get rejected and observe the
    resulting ε sequence."""

    def _build(self, *, reject_on_iter, eps_hold_factor=3.0):
        """Build a Scholtes strategy whose inner solve fails on the
        outer indices in ``reject_on_iter`` and succeeds elsewhere."""
        p = _make_problem()
        strat = ScholtesStrategy(
            p,
            ipopt_options={},
            safeguard_rollback=True,
            eps_hold_factor=eps_hold_factor,
            epsilon_0=1.0,
            reduction=0.1,
            epsilon_min=1e-12,
            max_iter=10,
            rollback_max_count=10,  # make sure cap doesn't end the trace early
            comp_eps_ratio_theta=1e18,  # accept on residual; we control via status
        )

        observed_eps: list[float] = []
        call_idx = {"k": 0}

        # Scholtes mult_g layout: [G(n_comp) | H(n_comp) | GH(n_comp)] = 3.
        n_mult = 3 * p.n_comp + p.n_ineq + p.n_eq

        def fake_solve(nlp, x, warm_dual):  # noqa: ARG001
            k = call_idx["k"]
            call_idx["k"] += 1
            observed_eps.append(float(nlp._last_epsilon))
            status = -2 if k in reject_on_iter else 0
            return x, {
                "status": status,
                "status_msg": b"ok" if status == 0 else b"restoration_failed",
                "mult_g": np.zeros(n_mult),
                "mult_x_L": np.zeros(p.n),
                "mult_x_U": np.zeros(p.n),
                "obj_val": 0.0,
            }, 0.001

        strat._timed_solve = fake_solve  # type: ignore[assignment]
        return strat, observed_eps

    def _patch_eps_capture(self, strat):
        """Wrap _build_nlp/eps_setter so each outer iter records ε on the NLP."""
        orig_run = strat._run_epsilon_continuation

        def wrapped(nlp, x0, eps_ref, make_iteration):
            class _Probe:
                def __init__(self, inner): self._inner = inner; self._last_epsilon = float("nan")
                def __getattr__(self, k): return getattr(self._inner, k)
            probe = _Probe(nlp)

            # Hook: every read of eps_ref[0] reflects the current ε; copy
            # into probe so fake_solve can observe it. We do this by
            # subclassing the eps_ref list.
            class _ERef(list):
                def __setitem__(self, idx, val):
                    super().__setitem__(idx, val)
                    probe._last_epsilon = float(val)

            new_eps_ref = _ERef(eps_ref)
            return orig_run(probe, x0, new_eps_ref, make_iteration)

        strat._run_epsilon_continuation = wrapped  # type: ignore[assignment]

    def test_no_reject_clean_geometric_decay(self):
        strat, eps_seen = self._build(reject_on_iter=set())
        self._patch_eps_capture(strat)
        result = pympcc.solve(strat.problem, strategy="scholtes",
                              _strategy_override=strat) if False else None
        # Direct path: invoke solve via the strategy's own pipeline.
        from pympcc.solver import MPCCSolver
        solver = MPCCSolver(strat.problem)
        solver._strategy = strat  # type: ignore[attr-defined]
        # Run the strategy directly; we just need the eps trace.
        strat.solve()
        # ε should decay geometrically: 1.0, 0.1, 0.01, ... until <epsilon_min
        for prev, cur in zip(eps_seen, eps_seen[1:]):
            assert cur == pytest.approx(prev * 0.1, rel=1e-9)

    def test_rejection_backs_off_capped(self):
        # Reject iter 2 (ε = 0.01). After backoff, ε should grow toward
        # the last accepted ε = 0.1, capped at 0.1 (since 0.01 * 3 = 0.03).
        strat, eps_seen = self._build(reject_on_iter={2})
        self._patch_eps_capture(strat)
        strat.solve()
        # Expected trajectory:
        #   k=0: ε=1.0  (accept)
        #   k=1: ε=0.1  (accept, last_good_eps=0.1)
        #   k=2: ε=0.01 (REJECT → eps = min(0.01 * 3, 0.1) = 0.03)
        #   k=3: ε=0.03 (accept)
        assert eps_seen[0] == pytest.approx(1.0)
        assert eps_seen[1] == pytest.approx(0.1)
        assert eps_seen[2] == pytest.approx(0.01)
        assert eps_seen[3] == pytest.approx(0.03, rel=1e-9)

    def test_rejection_capped_at_last_good(self):
        # Big eps_hold (10×) → backoff would land at 0.1, exactly the last good.
        strat, eps_seen = self._build(reject_on_iter={2}, eps_hold_factor=10.0)
        self._patch_eps_capture(strat)
        strat.solve()
        # 0.01 * 10 = 0.1, last_good_eps = 0.1 → cap fires, eps = 0.1.
        assert eps_seen[3] == pytest.approx(0.1, rel=1e-9)

    def test_eps_hold_one_breaks_to_avoid_loop(self):
        # eps_hold_factor=1.0 (legacy default) would reproduce the bug:
        # the new guard ``if eps <= snap_eps: break`` should terminate
        # rather than retry the same NLP.
        strat, eps_seen = self._build(reject_on_iter={2}, eps_hold_factor=1.0)
        self._patch_eps_capture(strat)
        strat.solve()
        # After reject at k=2, eps stays at 0.01 (== snap_eps), the new
        # guard breaks the outer loop. So no k=3 iter should occur.
        assert len(eps_seen) == 3
