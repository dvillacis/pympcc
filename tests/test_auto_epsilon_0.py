"""Tests for the central-path auto-init of epsilon_0.

Covers the resolver math (clipping at low/high ends, passthrough in the
band), the per-strategy sentinel wiring, and the bit-for-bit
preservation of numeric ``epsilon_0`` behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc.strategies._base import BaseStrategy
from pympcc.strategies.lin_fukushima import LinFukushimaStrategy
from pympcc.strategies.scholtes import ScholtesStrategy
from pympcc.strategies.slack import SlackStrategy
from pympcc.strategies.smoothing import SmoothingStrategy


def _make_problem(g_val: float = 0.5, h_val: float = 0.5):
    """Tiny dense MPCC where G(x0) = g_val, H(x0) = h_val."""
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([g_val, h_val]),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


# ======================================================================= #
# Resolver math                                                             #
# ======================================================================= #

class TestResolver:
    def test_clips_low(self):
        # G*H = 1e-4 → resolver pins to lo=1e-1.
        p = _make_problem(g_val=1e-2, h_val=1e-2)
        eps0, raw = BaseStrategy._resolve_auto_epsilon_0(p)
        assert raw == pytest.approx(1e-4)
        assert eps0 == pytest.approx(1e-1)

    def test_clips_high(self):
        # G*H = 1e3 → resolver pins to hi=1.0.
        p = _make_problem(g_val=1e2, h_val=10.0)
        eps0, raw = BaseStrategy._resolve_auto_epsilon_0(p)
        assert raw == pytest.approx(1e3)
        assert eps0 == pytest.approx(1.0)

    def test_passthrough(self):
        # G*H = 0.3 lies in [0.1, 1.0] → resolver returns it untouched.
        p = _make_problem(g_val=0.6, h_val=0.5)
        eps0, raw = BaseStrategy._resolve_auto_epsilon_0(p)
        assert raw == pytest.approx(0.30)
        assert eps0 == pytest.approx(0.30)

    def test_theta_buffer(self):
        # theta=2.0 doubles the residual before clipping.
        p = _make_problem(g_val=0.4, h_val=0.5)  # G*H = 0.2
        eps0, _ = BaseStrategy._resolve_auto_epsilon_0(p, theta=2.0)
        assert eps0 == pytest.approx(0.4)

    def test_zero_residual(self):
        # G(x0)*H(x0) == 0 must clip to lo, not crash.
        p = _make_problem(g_val=0.0, h_val=0.5)
        eps0, raw = BaseStrategy._resolve_auto_epsilon_0(p)
        assert raw == 0.0
        assert eps0 == pytest.approx(1e-1)


# ======================================================================= #
# Per-strategy sentinel wiring                                              #
# ======================================================================= #

_STRATEGY_CLASSES = {
    "scholtes": ScholtesStrategy,
    "slack":    SlackStrategy,
    "smoothing": SmoothingStrategy,
    "lin_fukushima": LinFukushimaStrategy,
}


class TestSentinelWiring:
    @pytest.mark.parametrize("name,cls", list(_STRATEGY_CLASSES.items()))
    def test_auto_resolves_at_init(self, name, cls):
        p = _make_problem(g_val=0.6, h_val=0.5)
        strat = cls(p, ipopt_options={}, epsilon_0="auto")
        # Resolver picks raw 0.30, in the [0.1, 1.0] band, so passthrough.
        assert strat.epsilon_0 == pytest.approx(0.30)
        assert strat._auto_eps0_origin is not None
        raw, eps0 = strat._auto_eps0_origin
        assert raw == pytest.approx(0.30)
        assert eps0 == strat.epsilon_0

    @pytest.mark.parametrize("name,cls", list(_STRATEGY_CLASSES.items()))
    def test_numeric_unchanged(self, name, cls):
        p = _make_problem(g_val=0.6, h_val=0.5)
        strat = cls(p, ipopt_options={}, epsilon_0=0.7)
        assert strat.epsilon_0 == pytest.approx(0.7)
        assert strat._auto_eps0_origin is None

    @pytest.mark.parametrize("strategy", list(_STRATEGY_CLASSES))
    def test_solve_with_auto(self, strategy):
        """End-to-end: 'auto' yields a converging solve, history starts at
        the resolved ε₀ (within reduction-loop precision)."""
        p = _make_problem(g_val=0.5, h_val=0.5)
        result = pympcc.solve(p, strategy=strategy, epsilon_0="auto")
        assert result.success
        assert len(result.history) >= 1
        # First iter's ε equals the resolved ε₀ (= clip(0.25, 0.1, 1.0) = 0.25).
        assert result.history[0].epsilon == pytest.approx(0.25)
