"""Tests for §5.5 EPEC multi-leader-common-follower emitter (`from_epec`)."""
from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc.bilevel import Leader, LowerLevel, from_epec

try:
    import jax.numpy as jnp  # noqa: F401
    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False

requires_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")


# ---------------------------------------------------------------------------
# Toy A — closed-form 2-leader EPEC, complementarity inactive at Nash.
#
#   Leaders:  min_{x_i}  F_i(x, y),  i = 1, 2
#       F_1(x, y) = (x_1 - 1)² + (y - 1)²
#       F_2(x, y) = (x_2 - 2)² + (y - 1)²
#   Common lower:  y ∈ argmin_y { (y - x_1 - x_2)² : y >= 0 }
#
#   Nash:  x_1* = 1/3,  x_2* = 4/3,  y* = 5/3,  λ_lo* = 0  (inactive).
#   Derivation: best-response gives 2x_1 + x_2 = 2,  x_1 + 2x_2 = 3.
# ---------------------------------------------------------------------------

def _toy_A():
    return from_epec(
        leaders=[
            Leader(n_x=1, x0=np.array([0.5]),
                   F=lambda x, y: (x[0] - 1.0) ** 2 + (y[0] - 1.0) ** 2),
            Leader(n_x=1, x0=np.array([0.5]),
                   F=lambda x, y: (x[1] - 2.0) ** 2 + (y[0] - 1.0) ** 2),
        ],
        common_lower=LowerLevel(
            n_y=1, y0=np.array([0.5]),
            f=lambda x, y: (y[0] - x[0] - x[1]) ** 2,
            n_g=1, g=lambda x, y: jnp.array([-y[0]]),
        ),
    )


# ---------------------------------------------------------------------------
# Toy B — same structure, complementarity active at Nash.
#
#   F_1 = (x_1 + 0.5)² + y²,   F_2 = (x_2 + 0.5)² + y²
#   Lower:  y ∈ argmin_y { (y - x_1 - x_2)² : y >= 0 }
#
#   Nash:  x_1* = x_2* = -0.5,  y* = 0,  λ_lo* = 2  (active).
# ---------------------------------------------------------------------------

def _toy_B():
    return from_epec(
        leaders=[
            Leader(n_x=1, x0=np.array([-0.2]),
                   F=lambda x, y: (x[0] + 0.5) ** 2 + y[0] ** 2),
            Leader(n_x=1, x0=np.array([-0.2]),
                   F=lambda x, y: (x[1] + 0.5) ** 2 + y[0] ** 2),
        ],
        common_lower=LowerLevel(
            n_y=1, y0=np.array([0.1]),
            f=lambda x, y: (y[0] - x[0] - x[1]) ** 2,
            n_g=1, g=lambda x, y: jnp.array([-y[0]]),
        ),
    )


# ---------------------------------------------------------------------------
# Construction & layout
# ---------------------------------------------------------------------------

@requires_jax
class TestConstruction:
    def test_returns_mpcc_problem(self):
        p = _toy_A()
        assert isinstance(p, pympcc.MPCCProblem)

    def test_variable_layout(self):
        # n = sum(n_x) + n_y + n_g + n_h + N·(n_y + n_h + 2·n_g)
        #   = 2 + 1 + 1 + 0 + 2·(1 + 0 + 2·1) = 4 + 6 = 10
        p = _toy_A()
        assert p.n == 10

    def test_n_eq(self):
        # n_eq = sum(n_x) + N·(n_y + n_g + n_h) + n_y + n_h
        #      = 2 + 2·(1 + 1 + 0) + 1 + 0 = 7
        p = _toy_A()
        assert p.n_eq == 7

    def test_n_comp(self):
        # n_comp = (1 + 2N)·n_g = 5
        p = _toy_A()
        assert p.n_comp == 5

    def test_lambda_lower_bound_zero(self):
        # λ_lo lives at index n_x_total + n_y = 3
        p = _toy_A()
        assert p.xl[3] == 0.0
        assert np.isinf(p.xu[3])

    def test_theta_nu_lower_bounds_zero(self):
        # First leader's dual block starts at off_mu + n_h = 4 (since n_h = 0).
        # Layout: ξ_y(1) θ(1) ν(1).  θ at index 5, ν at index 6.
        p = _toy_A()
        assert p.xl[5] == 0.0   # θ_1
        assert p.xl[6] == 0.0   # ν_1
        assert p.xl[8] == 0.0   # θ_2
        assert p.xl[9] == 0.0   # ν_2

    def test_x_blocks_unbounded(self):
        # Leader x blocks live at z[0:2]; default bounds are ±inf.
        p = _toy_A()
        assert np.isneginf(p.xl[0]) and np.isneginf(p.xl[1])
        assert np.isposinf(p.xu[0]) and np.isposinf(p.xu[1])

    def test_initial_point_uses_supplied_x0(self):
        p = _toy_A()
        np.testing.assert_allclose(p.x0[:2], [0.5, 0.5])
        np.testing.assert_allclose(p.x0[2:3], [0.5])
        # Multipliers default to zero
        np.testing.assert_allclose(p.x0[3:], 0.0)


# ---------------------------------------------------------------------------
# Solve & convergence to closed-form Nash equilibria
# ---------------------------------------------------------------------------

# 'smoothing' (FB-equality) hits NO_DOF on the over-determined EPEC
# KKT-stack — the Fischer-Burmeister equation φ_ε(G,H)=0 adds n_comp
# equality rows on top of the already-tight stationarity system.  The
# inequality-based strategies converge cleanly.
_EPEC_STRATEGIES = ["direct", "scholtes", "lin_fukushima"]


@requires_jax
class TestToyA:
    @pytest.mark.parametrize("strategy", _EPEC_STRATEGIES)
    def test_converges(self, strategy):
        p = _toy_A()
        r = pympcc.solve(p, strategy=strategy, ipopt_options={"print_level": 0})
        assert r.success, f"{strategy} did not converge: status={r.status}"

    @pytest.mark.parametrize("strategy", _EPEC_STRATEGIES)
    def test_x1_at_nash(self, strategy):
        p = _toy_A()
        r = pympcc.solve(p, strategy=strategy, ipopt_options={"print_level": 0})
        assert abs(r.x[0] - 1.0 / 3.0) < 1e-3

    @pytest.mark.parametrize("strategy", _EPEC_STRATEGIES)
    def test_x2_at_nash(self, strategy):
        p = _toy_A()
        r = pympcc.solve(p, strategy=strategy, ipopt_options={"print_level": 0})
        assert abs(r.x[1] - 4.0 / 3.0) < 1e-3

    @pytest.mark.parametrize("strategy", _EPEC_STRATEGIES)
    def test_y_at_nash(self, strategy):
        p = _toy_A()
        r = pympcc.solve(p, strategy=strategy, ipopt_options={"print_level": 0})
        assert abs(r.x[2] - 5.0 / 3.0) < 1e-3

    @pytest.mark.parametrize("strategy", _EPEC_STRATEGIES)
    def test_lambda_inactive(self, strategy):
        p = _toy_A()
        r = pympcc.solve(p, strategy=strategy, ipopt_options={"print_level": 0})
        assert r.x[3] < 1e-4   # λ_lo* = 0

    @pytest.mark.parametrize("strategy", _EPEC_STRATEGIES)
    def test_comp_residual_small(self, strategy):
        p = _toy_A()
        r = pympcc.solve(p, strategy=strategy, ipopt_options={"print_level": 0})
        assert r.comp_residual < 1e-4


@requires_jax
class TestToyB:
    @pytest.mark.parametrize("strategy", _EPEC_STRATEGIES)
    def test_converges(self, strategy):
        p = _toy_B()
        r = pympcc.solve(p, strategy=strategy, ipopt_options={"print_level": 0})
        assert r.success, f"{strategy} did not converge: status={r.status}"

    @pytest.mark.parametrize("strategy", _EPEC_STRATEGIES)
    def test_x_at_nash(self, strategy):
        p = _toy_B()
        r = pympcc.solve(p, strategy=strategy, ipopt_options={"print_level": 0})
        assert abs(r.x[0] - (-0.5)) < 1e-3
        assert abs(r.x[1] - (-0.5)) < 1e-3

    @pytest.mark.parametrize("strategy", _EPEC_STRATEGIES)
    def test_y_at_zero_active(self, strategy):
        p = _toy_B()
        r = pympcc.solve(p, strategy=strategy, ipopt_options={"print_level": 0})
        assert abs(r.x[2]) < 1e-4

    @pytest.mark.parametrize("strategy", _EPEC_STRATEGIES)
    def test_lambda_active(self, strategy):
        # At Nash, lower-level constraint y >= 0 is active so λ_lo > 0.
        p = _toy_B()
        r = pympcc.solve(p, strategy=strategy, ipopt_options={"print_level": 0})
        assert abs(r.x[3] - 2.0) < 1e-3


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@requires_jax
class TestErrors:
    def test_fewer_than_two_leaders_raises(self):
        with pytest.raises(ValueError, match="at least 2 leaders"):
            from_epec(
                leaders=[
                    Leader(n_x=1, x0=np.array([0.0]),
                           F=lambda x, y: x[0] ** 2 + y[0] ** 2),
                ],
                common_lower=LowerLevel(
                    n_y=1, y0=np.array([0.0]),
                    f=lambda x, y: y[0] ** 2,
                    n_g=1, g=lambda x, y: jnp.array([-y[0]]),
                ),
            )

    def test_no_lower_level_inequality_raises(self):
        with pytest.raises(ValueError, match="n_g must be"):
            from_epec(
                leaders=[
                    Leader(n_x=1, x0=np.array([0.0]),
                           F=lambda x, y: x[0] ** 2 + y[0] ** 2),
                    Leader(n_x=1, x0=np.array([0.0]),
                           F=lambda x, y: x[1] ** 2 + y[0] ** 2),
                ],
                common_lower=LowerLevel(
                    n_y=1, y0=np.array([0.0]),
                    f=lambda x, y: y[0] ** 2,
                    n_g=0, g=None,
                ),
            )

    def test_h_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="must agree"):
            from_epec(
                leaders=[
                    Leader(n_x=1, x0=np.array([0.0]),
                           F=lambda x, y: x[0] ** 2 + y[0] ** 2),
                    Leader(n_x=1, x0=np.array([0.0]),
                           F=lambda x, y: x[1] ** 2 + y[0] ** 2),
                ],
                common_lower=LowerLevel(
                    n_y=1, y0=np.array([0.0]),
                    f=lambda x, y: y[0] ** 2,
                    n_g=1, g=lambda x, y: jnp.array([-y[0]]),
                    n_h=1, h=None,   # mismatched: n_h>0 but h is None
                ),
            )

    def test_fd_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="jax"):
            from_epec(
                leaders=[
                    Leader(n_x=1, x0=np.array([0.0]),
                           F=lambda x, y: x[0] ** 2 + y[0] ** 2),
                    Leader(n_x=1, x0=np.array([0.0]),
                           F=lambda x, y: x[1] ** 2 + y[0] ** 2),
                ],
                common_lower=LowerLevel(
                    n_y=1, y0=np.array([0.0]),
                    f=lambda x, y: y[0] ** 2,
                    n_g=1, g=lambda x, y: np.array([-y[0]]),
                ),
                derivatives="fd",
            )

    def test_unknown_derivatives_raises(self):
        with pytest.raises(ValueError, match="derivatives"):
            from_epec(
                leaders=[
                    Leader(n_x=1, x0=np.array([0.0]),
                           F=lambda x, y: x[0] ** 2 + y[0] ** 2),
                    Leader(n_x=1, x0=np.array([0.0]),
                           F=lambda x, y: x[1] ** 2 + y[0] ** 2),
                ],
                common_lower=LowerLevel(
                    n_y=1, y0=np.array([0.0]),
                    f=lambda x, y: y[0] ** 2,
                    n_g=1, g=lambda x, y: jnp.array([-y[0]]),
                ),
                derivatives="symbolic",
            )

    def test_zero_size_leader_raises(self):
        with pytest.raises(ValueError, match="n_x"):
            from_epec(
                leaders=[
                    Leader(n_x=0, x0=np.zeros(0),
                           F=lambda x, y: y[0] ** 2),
                    Leader(n_x=1, x0=np.array([0.0]),
                           F=lambda x, y: x[0] ** 2 + y[0] ** 2),
                ],
                common_lower=LowerLevel(
                    n_y=1, y0=np.array([0.0]),
                    f=lambda x, y: y[0] ** 2,
                    n_g=1, g=lambda x, y: jnp.array([-y[0]]),
                ),
            )

    def test_x0_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="x0"):
            from_epec(
                leaders=[
                    Leader(n_x=2, x0=np.array([0.0]),   # n_x=2 but x0 shape (1,)
                           F=lambda x, y: jnp.sum(x[:2] ** 2) + y[0] ** 2),
                    Leader(n_x=1, x0=np.array([0.0]),
                           F=lambda x, y: x[2] ** 2 + y[0] ** 2),
                ],
                common_lower=LowerLevel(
                    n_y=1, y0=np.array([0.0]),
                    f=lambda x, y: y[0] ** 2,
                    n_g=1, g=lambda x, y: jnp.array([-y[0]]),
                ),
            )


# ---------------------------------------------------------------------------
# Solution diagnostics
# ---------------------------------------------------------------------------

@requires_jax
class TestSolutionDiagnostics:
    def test_history_present(self):
        p = _toy_A()
        r = pympcc.solve(p, strategy="scholtes", ipopt_options={"print_level": 0})
        assert r.history is not None and len(r.history) > 0

    def test_strategy_recorded(self):
        p = _toy_A()
        r = pympcc.solve(p, strategy="direct", ipopt_options={"print_level": 0})
        assert r.strategy == "direct"

    def test_objective_is_zero(self):
        # EPEC objective is the constant zero — feasibility problem.
        p = _toy_A()
        r = pympcc.solve(p, strategy="scholtes", ipopt_options={"print_level": 0})
        assert abs(r.obj) < 1e-12
