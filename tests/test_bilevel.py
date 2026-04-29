"""Tests for §5.4 bilevel KKT-emitter frontend (`pympcc.bilevel`)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pympcc
from pympcc.bilevel import from_lower_level

try:
    import jax  # noqa: F401
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False


# ---------------------------------------------------------------------------
# Closed-form bilevel A (inactive comp at optimum)
#
#   min_{x,y}  (x - 1)^2 + (y - 1)^2
#   s.t.       y ∈ argmin_y { (y - x)^2 : y >= 0 }
#
# Optimum: (x*, y*, λ*) = (1, 1, 0); upper objective = 0.
# ---------------------------------------------------------------------------

def _bilevel_A_fd():
    return from_lower_level(
        n_x=1, n_y=1,
        x0=np.array([0.5]), y0=np.array([0.5]),
        f_upper=lambda x, y: (x[0] - 1.0) ** 2 + (y[0] - 1.0) ** 2,
        f_lower=lambda x, y: (y[0] - x[0]) ** 2,
        n_g_lower=1,
        g_lower=lambda x, y: np.array([-y[0]]),
        derivatives="fd",
    )


def _bilevel_A_jax():
    return from_lower_level(
        n_x=1, n_y=1,
        x0=np.array([0.5]), y0=np.array([0.5]),
        f_upper=lambda x, y: (x[0] - 1.0) ** 2 + (y[0] - 1.0) ** 2,
        f_lower=lambda x, y: (y[0] - x[0]) ** 2,
        n_g_lower=1,
        g_lower=lambda x, y: jnp.array([-y[0]]),
        derivatives="jax",
    )


# ---------------------------------------------------------------------------
# Closed-form bilevel B (active comp at optimum, λ > 0)
#
#   min_{x,y}  (x - 0.5)^2 + (y - 0.5)^2
#   s.t.       y ∈ argmin_y { y^2 + 2*x*y : y >= 0 }
#
# Optimum: (x*, y*, λ*) = (0.5, 0, 1); upper objective = 0.25.
# ---------------------------------------------------------------------------

def _bilevel_B_fd():
    return from_lower_level(
        n_x=1, n_y=1,
        x0=np.array([0.2]), y0=np.array([0.3]),
        f_upper=lambda x, y: (x[0] - 0.5) ** 2 + (y[0] - 0.5) ** 2,
        f_lower=lambda x, y: y[0] ** 2 + 2.0 * x[0] * y[0],
        n_g_lower=1,
        g_lower=lambda x, y: np.array([-y[0]]),
        derivatives="fd",
    )


def _bilevel_B_jax():
    return from_lower_level(
        n_x=1, n_y=1,
        x0=np.array([0.2]), y0=np.array([0.3]),
        f_upper=lambda x, y: (x[0] - 0.5) ** 2 + (y[0] - 0.5) ** 2,
        f_lower=lambda x, y: y[0] ** 2 + 2.0 * x[0] * y[0],
        n_g_lower=1,
        g_lower=lambda x, y: jnp.array([-y[0]]),
        derivatives="jax",
    )


# ---------------------------------------------------------------------------
# Construction & layout
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_returns_mpcc_problem(self):
        p = _bilevel_A_fd()
        assert isinstance(p, pympcc.MPCCProblem)

    def test_variable_layout(self):
        # n = n_x + n_y + n_g_lower + n_h_lower = 1 + 1 + 1 + 0 = 3
        p = _bilevel_A_fd()
        assert p.n == 3
        assert p.n_comp == 1
        assert p.n_eq == 1   # one stationarity row, no h_lower

    def test_initial_point_packs_correctly(self):
        p = _bilevel_A_fd()
        # z0 = [x0, y0, lambda0=0]
        np.testing.assert_allclose(p.x0, [0.5, 0.5, 0.0])

    def test_lambda_lower_bound_is_zero(self):
        p = _bilevel_A_fd()
        assert p.xl[2] == 0.0
        assert p.xu[2] == np.inf

    def test_initial_point_with_lambda0(self):
        p = from_lower_level(
            n_x=1, n_y=1,
            x0=np.array([0.0]), y0=np.array([0.0]),
            f_upper=lambda x, y: x[0] + y[0],
            f_lower=lambda x, y: y[0] ** 2,
            n_g_lower=1,
            g_lower=lambda x, y: np.array([-y[0]]),
            lambda0=np.array([0.5]),
            derivatives="fd",
        )
        np.testing.assert_allclose(p.x0, [0.0, 0.0, 0.5])

    def test_user_bounds_propagate(self):
        p = from_lower_level(
            n_x=2, n_y=1,
            x0=np.array([0.5, 0.5]), y0=np.array([0.5]),
            xl=np.array([0.0, -1.0]), xu=np.array([1.0, 1.0]),
            yl=np.array([-2.0]), yu=np.array([2.0]),
            f_upper=lambda x, y: x[0] + x[1] + y[0],
            f_lower=lambda x, y: y[0] ** 2,
            n_g_lower=1,
            g_lower=lambda x, y: np.array([-y[0]]),
            derivatives="fd",
        )
        # n = 2 + 1 + 1 = 4
        np.testing.assert_allclose(p.xl, [0.0, -1.0, -2.0, 0.0])
        np.testing.assert_allclose(p.xu, [1.0, 1.0, 2.0, np.inf])

    def test_with_equality_constraint(self):
        # Lower-level has an equality h_lower = y[0] - x[0] = 0
        p = from_lower_level(
            n_x=1, n_y=1,
            x0=np.array([0.5]), y0=np.array([0.5]),
            f_upper=lambda x, y: x[0] + y[0],
            f_lower=lambda x, y: y[0] ** 2,
            n_g_lower=1,
            g_lower=lambda x, y: np.array([-y[0]]),
            n_h_lower=1,
            h_lower=lambda x, y: np.array([y[0] - x[0]]),
            derivatives="fd",
        )
        # n = 1 + 1 + 1 + 1 = 4 (μ block adds one var)
        assert p.n == 4
        assert p.n_eq == 2   # 1 stationarity + 1 h_lower
        assert p.n_comp == 1
        # μ is unbounded
        assert p.xl[3] == -np.inf
        assert p.xu[3] == np.inf


# ---------------------------------------------------------------------------
# Stationarity and feasibility at the analytical optimum
# ---------------------------------------------------------------------------

class TestKKTAtOptimum:
    def test_bilevel_A_stationarity_zero_at_optimum(self):
        p = _bilevel_A_fd()
        # z* = (x=1, y=1, lambda=0)
        zstar = np.array([1.0, 1.0, 0.0])
        residual = p.eq_constraints(zstar)
        # ∇_y L = 2(y - x) - λ = 0 at z*
        assert np.allclose(residual, [0.0], atol=1e-5)

    def test_bilevel_A_complementarity_at_optimum(self):
        p = _bilevel_A_fd()
        zstar = np.array([1.0, 1.0, 0.0])
        G = p.comp_G(zstar)   # λ
        H = p.comp_H(zstar)   # -g = y
        np.testing.assert_allclose(G, [0.0])
        np.testing.assert_allclose(H, [1.0])
        assert float(G @ H) == 0.0

    def test_bilevel_B_stationarity_zero_at_optimum(self):
        p = _bilevel_B_fd()
        # z* = (x=0.5, y=0, lambda=1) -> 2y + 2x - lambda = 0 + 1 - 1 = 0
        zstar = np.array([0.5, 0.0, 1.0])
        residual = p.eq_constraints(zstar)
        assert np.allclose(residual, [0.0], atol=1e-5)

    def test_bilevel_B_complementarity_at_optimum(self):
        p = _bilevel_B_fd()
        zstar = np.array([0.5, 0.0, 1.0])
        G = p.comp_G(zstar)
        H = p.comp_H(zstar)
        np.testing.assert_allclose(G, [1.0])
        np.testing.assert_allclose(H, [0.0])
        assert float(G @ H) == 0.0


# ---------------------------------------------------------------------------
# End-to-end solves
# ---------------------------------------------------------------------------

class TestSolve:
    def test_bilevel_A_fd_converges(self):
        p = _bilevel_A_fd()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pympcc.solve(p, strategy="scholtes")
        assert res.success
        # x* = 1, y* = 1, lambda* = 0; upper obj = 0
        np.testing.assert_allclose(res.x[:2], [1.0, 1.0], atol=5e-3)
        assert res.obj < 1e-4

    def test_bilevel_B_fd_converges(self):
        p = _bilevel_B_fd()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pympcc.solve(p, strategy="scholtes")
        assert res.success
        # x* = 0.5, y* = 0, lambda* = 1; upper obj = 0.25
        np.testing.assert_allclose(res.x[:2], [0.5, 0.0], atol=5e-3)
        np.testing.assert_allclose(res.x[2], 1.0, atol=5e-3)
        np.testing.assert_allclose(res.obj, 0.25, atol=1e-3)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_bilevel_A_jax_converges(self):
        p = _bilevel_A_jax()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pympcc.solve(p, strategy="scholtes")
        assert res.success
        np.testing.assert_allclose(res.x[:2], [1.0, 1.0], atol=5e-3)
        assert res.obj < 1e-4

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_bilevel_B_jax_converges(self):
        p = _bilevel_B_jax()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pympcc.solve(p, strategy="scholtes")
        assert res.success
        np.testing.assert_allclose(res.x[:2], [0.5, 0.0], atol=5e-3)
        np.testing.assert_allclose(res.x[2], 1.0, atol=5e-3)
        np.testing.assert_allclose(res.obj, 0.25, atol=1e-3)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_invalid_derivatives(self):
        with pytest.raises(ValueError, match="derivatives"):
            from_lower_level(
                n_x=1, n_y=1,
                x0=np.array([0.0]), y0=np.array([0.0]),
                f_upper=lambda x, y: 0.0,
                f_lower=lambda x, y: 0.0,
                n_g_lower=1,
                g_lower=lambda x, y: np.array([0.0]),
                derivatives="autodiff",
            )

    def test_zero_g_lower_rejected(self):
        with pytest.raises(ValueError, match="n_g_lower"):
            from_lower_level(
                n_x=1, n_y=1,
                x0=np.array([0.0]), y0=np.array([0.0]),
                f_upper=lambda x, y: 0.0,
                f_lower=lambda x, y: 0.0,
                n_g_lower=0,
                derivatives="fd",
            )

    def test_g_lower_required(self):
        with pytest.raises(ValueError, match="g_lower"):
            from_lower_level(
                n_x=1, n_y=1,
                x0=np.array([0.0]), y0=np.array([0.0]),
                f_upper=lambda x, y: 0.0,
                f_lower=lambda x, y: 0.0,
                n_g_lower=1,
                g_lower=None,
                derivatives="fd",
            )

    def test_h_lower_must_match_n_h_lower(self):
        with pytest.raises(ValueError, match="h_lower"):
            from_lower_level(
                n_x=1, n_y=1,
                x0=np.array([0.0]), y0=np.array([0.0]),
                f_upper=lambda x, y: 0.0,
                f_lower=lambda x, y: 0.0,
                n_g_lower=1,
                g_lower=lambda x, y: np.array([0.0]),
                n_h_lower=2,
                h_lower=None,
                derivatives="fd",
            )

    def test_x0_shape_mismatch(self):
        with pytest.raises(ValueError, match="x0"):
            from_lower_level(
                n_x=2, n_y=1,
                x0=np.array([0.0]),  # wrong length
                y0=np.array([0.0]),
                f_upper=lambda x, y: 0.0,
                f_lower=lambda x, y: 0.0,
                n_g_lower=1,
                g_lower=lambda x, y: np.array([0.0]),
                derivatives="fd",
            )

    def test_y0_shape_mismatch(self):
        with pytest.raises(ValueError, match="y0"):
            from_lower_level(
                n_x=1, n_y=2,
                x0=np.array([0.0]),
                y0=np.array([0.0]),  # wrong length
                f_upper=lambda x, y: 0.0,
                f_lower=lambda x, y: 0.0,
                n_g_lower=1,
                g_lower=lambda x, y: np.array([0.0]),
                derivatives="fd",
            )

    def test_negative_lambda0_rejected(self):
        with pytest.raises(ValueError, match="lambda0"):
            from_lower_level(
                n_x=1, n_y=1,
                x0=np.array([0.0]), y0=np.array([0.0]),
                f_upper=lambda x, y: 0.0,
                f_lower=lambda x, y: 0.0,
                n_g_lower=1,
                g_lower=lambda x, y: np.array([0.0]),
                lambda0=np.array([-0.1]),
                derivatives="fd",
            )

    def test_module_exports_from_top_level(self):
        # `pympcc.bilevel.from_lower_level` is the documented entry point.
        assert pympcc.bilevel.from_lower_level is from_lower_level
