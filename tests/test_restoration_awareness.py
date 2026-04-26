"""Tests for restoration-phase awareness.

Covers the cyipopt ``intermediate`` callback bookkeeping (alg_mod=1
detection + reset), the IterationInfo population, and the rollback
threshold validation. Triggering an actual restoration phase requires
a problem IPOPT cannot solve cleanly, so we simulate the NLP-side
counters directly for unit-level coverage and add an integration
smoke test that the field plumbing reaches IterationInfo.
"""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc.result import IterationInfo


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
# IterationInfo defaults                                                    #
# ======================================================================= #

class TestIterationInfoFields:
    def test_default_values(self):
        info = IterationInfo(
            epsilon=1.0, x=np.zeros(1), obj=0.0,
            status=0, message="ok",
            comp_residual=0.0, comp_residual_mean=0.0,
            n_ipopt_iter=1, iter_time=0.0,
        )
        assert info.restoration_iter_count == 0
        assert info.entered_restoration is False

    def test_can_be_set(self):
        info = IterationInfo(
            epsilon=1.0, x=np.zeros(1), obj=0.0,
            status=-2, message="restoration_failed",
            comp_residual=0.0, comp_residual_mean=0.0,
            n_ipopt_iter=10, iter_time=0.0,
            restoration_iter_count=7, entered_restoration=True,
        )
        assert info.restoration_iter_count == 7
        assert info.entered_restoration is True


# ======================================================================= #
# NLP-side restoration counters                                             #
# ======================================================================= #

class TestNLPCallback:
    """Drive ``intermediate`` directly to simulate restoration, then
    verify the bookkeeping. Avoids needing a problem that actually
    enters restoration (which is hard to construct deterministically)."""

    def _build_nlp(self):
        from pympcc.strategies.scholtes import ScholtesStrategy
        p = _make_problem()
        strat = ScholtesStrategy(p, ipopt_options={})
        # The strategy build path constructs the NLP lazily inside solve().
        # Pull one out via the public API by triggering a solve build then
        # grabbing the cached object — instead we just use _DenseNLP directly.
        from pympcc._nlp import _DenseNLP
        nlp = _DenseNLP(
            n=2, m=0,
            xl=np.full(2, -1e19), xu=np.full(2, 1e19),
            cl=np.empty(0), cu=np.empty(0),
            obj_fn=lambda x: 0.0,
            grad_fn=lambda x: np.zeros(2),
            con_fn=lambda x: np.empty(0),
            jac_fn=lambda x: np.empty((0, 2)),
        )
        return nlp

    def test_initial_state(self):
        nlp = self._build_nlp()
        assert nlp.n_ipopt_iter == 0
        assert nlp.entered_restoration is False
        assert nlp.restoration_iter_count == 0
        assert nlp.last_alg_mod == 0

    def test_regular_phase(self):
        nlp = self._build_nlp()
        for k in range(3):
            nlp.intermediate(0, k, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        assert nlp.n_ipopt_iter == 3
        assert nlp.entered_restoration is False
        assert nlp.restoration_iter_count == 0
        assert nlp.last_alg_mod == 0

    def test_restoration_phase_detected(self):
        nlp = self._build_nlp()
        # 2 regular iters, then 4 restoration iters, then 1 regular.
        for k in range(2):
            nlp.intermediate(0, k, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        for k in range(2, 6):
            nlp.intermediate(1, k, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        nlp.intermediate(0, 6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        assert nlp.n_ipopt_iter == 7
        assert nlp.entered_restoration is True
        assert nlp.restoration_iter_count == 4
        assert nlp.last_alg_mod == 0  # back to regular at the end

    def test_reset_clears(self):
        nlp = self._build_nlp()
        nlp.intermediate(1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        assert nlp.entered_restoration is True
        nlp.reset_iter_counters()
        assert nlp.n_ipopt_iter == 0
        assert nlp.entered_restoration is False
        assert nlp.restoration_iter_count == 0
        assert nlp.last_alg_mod == 0


# ======================================================================= #
# Safeguard option validation                                               #
# ======================================================================= #

class TestSafeguardOption:
    def test_accepts_threshold(self):
        from pympcc.strategies.scholtes import ScholtesStrategy
        p = _make_problem()
        strat = ScholtesStrategy(p, ipopt_options={},
                                 safeguards="all", restoration_iter_threshold=10)
        assert strat.restoration_iter_threshold == 10

    def test_default_threshold(self):
        from pympcc.strategies.scholtes import ScholtesStrategy
        p = _make_problem()
        strat = ScholtesStrategy(p, ipopt_options={})
        assert strat.restoration_iter_threshold == 5

    def test_rejects_zero(self):
        from pympcc.strategies.scholtes import ScholtesStrategy
        p = _make_problem()
        with pytest.raises(ValueError, match="restoration_iter_threshold"):
            ScholtesStrategy(p, ipopt_options={}, restoration_iter_threshold=0)


# ======================================================================= #
# Integration: clean solve has empty restoration history                    #
# ======================================================================= #

class TestIntegration:
    @pytest.mark.parametrize("strategy", ["scholtes", "slack", "smoothing", "lin_fukushima"])
    def test_clean_solve_no_restoration(self, strategy):
        """Tiny well-conditioned MPCC must converge without restoration."""
        p = _make_problem()
        result = pympcc.solve(p, strategy=strategy)
        assert result.success
        for info in result.history:
            assert info.entered_restoration is False
            assert info.restoration_iter_count == 0
