"""Tests for the ``inner_tol_mode='matched'`` schedule.

The continuation loop ships three inner-tolerance schedules:

* ``linear`` — ``tol = max(user_tol, 0.01·ε)``       (default)
* ``quadratic`` — ``tol = max(user_tol, ε^1.5)``     (Leyffer; ``--safeguards``)
* ``matched`` — ``tol = max(inner_tol_floor, factor·ε)`` (no user_tol floor)

The ``matched`` mode loosens the inner tolerance as ε shrinks, so IPOPT
terminates at a precision the relaxation can actually deliver — fixing
the MAX_ITER spin observed on large TV-MPCC problems where ε drops
below ``user_tol`` and IPOPT keeps chasing 1e-8 it can't reach.
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
# Defaults & validation                                                     #
# ======================================================================= #

class TestDefaults:
    def test_factor_in_defaults(self):
        assert SAFEGUARD_DEFAULTS["inner_tol_factor"] == pytest.approx(0.1)
        assert SAFEGUARD_DEFAULTS["inner_tol_floor"] == pytest.approx(1e-10)

    def test_strategy_picks_up_defaults(self):
        strat = ScholtesStrategy(_make_problem(), ipopt_options={})
        assert strat.inner_tol_factor == pytest.approx(0.1)
        assert strat.inner_tol_floor == pytest.approx(1e-10)
        # Default mode is unchanged.
        assert strat.inner_tol_mode == "linear"

    def test_matched_mode_accepted(self):
        strat = ScholtesStrategy(
            _make_problem(), ipopt_options={}, inner_tol_mode="matched"
        )
        assert strat.inner_tol_mode == "matched"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="inner_tol_mode must be"):
            ScholtesStrategy(
                _make_problem(), ipopt_options={}, inner_tol_mode="bogus"
            )

    def test_invalid_factor(self):
        with pytest.raises(ValueError, match="inner_tol_factor must be > 0"):
            ScholtesStrategy(
                _make_problem(), ipopt_options={}, inner_tol_factor=0.0
            )

    def test_invalid_floor(self):
        with pytest.raises(ValueError, match="inner_tol_floor must be > 0"):
            ScholtesStrategy(
                _make_problem(), ipopt_options={}, inner_tol_floor=0.0
            )


# ======================================================================= #
# Mode arithmetic — observe the tol IPOPT receives                          #
# ======================================================================= #

class TestModeArithmetic:
    """Capture the ``tol`` values pushed to ``nlp.add_option`` across
    the outer loop and verify they follow the chosen schedule."""

    def _run_capture(self, **strategy_opts):
        p = _make_problem()
        strat = ScholtesStrategy(
            p,
            ipopt_options={"tol": 1e-8},
            epsilon_0=1.0,
            reduction=0.1,
            epsilon_min=1e-12,
            max_iter=6,
            comp_eps_ratio_theta=1e18,  # accept everything
            **strategy_opts,
        )
        captured: list[tuple[str, float]] = []

        def fake_solve(nlp, x, warm_dual):  # noqa: ARG001
            return x, {
                "status": 0,
                "status_msg": b"ok",
                "obj_val": 1.0,
                "mult_g":   np.zeros(3),
                "mult_x_L": np.zeros(2),
                "mult_x_U": np.zeros(2),
            }, 0.001

        strat._timed_solve = fake_solve  # type: ignore[assignment]

        # Wrap nlp.add_option to record every (key, val) pair.
        orig_run = strat._run_epsilon_continuation

        def wrapped(nlp, x0, eps_ref, make_iteration):
            orig_add = nlp.add_option
            def add_option(k, v):
                captured.append((k, float(v) if k == "tol" else v))
                return orig_add(k, v)
            nlp.add_option = add_option  # type: ignore[assignment]
            return orig_run(nlp, x0, eps_ref, make_iteration)

        strat._run_epsilon_continuation = wrapped  # type: ignore[assignment]
        strat.solve()
        return [v for k, v in captured if k == "tol"]

    def test_linear_floors_at_user_tol(self):
        tols = self._run_capture(inner_tol_mode="linear")
        # ε starts at 1.0, reduction 0.1 → 1.0, 0.1, 0.01, 0.001, ...
        # linear: max(1e-8, 0.01·ε) → 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-8
        assert tols[0] == pytest.approx(0.01)
        assert tols[1] == pytest.approx(0.001)
        # By the time ε ≤ 1e-6, tol pins to user_tol=1e-8.
        assert tols[-1] >= 1e-8 - 1e-15

    def test_matched_loosens_below_user_tol(self):
        tols = self._run_capture(inner_tol_mode="matched", inner_tol_factor=0.1)
        # matched: max(1e-10, 0.1·ε) → 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6
        assert tols[0] == pytest.approx(0.1)
        assert tols[1] == pytest.approx(0.01)
        # When ε = 1e-5 (iter 5), tol = 1e-6 — looser than the linear-mode
        # 1e-8 floor.  This is the whole point of "matched".
        assert tols[5] == pytest.approx(1e-6)

    def test_matched_floor(self):
        # Use a large floor so matched gets clipped.
        tols = self._run_capture(
            inner_tol_mode="matched",
            inner_tol_factor=0.1,
            inner_tol_floor=1e-3,
        )
        # All tols >= floor.
        assert min(tols) >= 1e-3 - 1e-15

    def test_matched_factor_scales(self):
        tols = self._run_capture(inner_tol_mode="matched", inner_tol_factor=1.0)
        # factor=1 → tol == ε at each iter (above floor).
        # ε sequence: 1.0, 0.1, 0.01, ...
        assert tols[0] == pytest.approx(1.0)
        assert tols[2] == pytest.approx(0.01)


# ======================================================================= #
# End-to-end                                                                #
# ======================================================================= #

class TestEndToEnd:
    def test_matched_mode_solves_clean_problem(self):
        result = pympcc.solve(
            _make_problem(),
            strategy="scholtes",
            inner_tol_mode="matched",
        )
        assert result.success
        assert result.comp_residual < 1e-4

    def test_matched_compatible_with_safeguards_all(self):
        # safeguards="all" leaves inner_tol_mode at its default; explicit "matched" is respected.
        result = pympcc.solve(
            _make_problem(),
            strategy="scholtes",
            safeguards="all",
            inner_tol_mode="matched",
        )
        assert result.success
