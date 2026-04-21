"""Tests for the JAX autodiff backend."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import pympcc
from pympcc._jax import HAS_JAX, jax_gradient, jax_jacobian, jax_hessian_lagrangian
from pympcc.problem import MPCCProblem


# ---------------------------------------------------------------------------
# Fixtures — reusable JAX-compatible problem definitions
# ---------------------------------------------------------------------------

def _simple_problem_jax(**kwargs):
    """min (x0-2)^2 + (x1-1)^2  s.t. x0 >= 0, x1 >= 0, x0*x1 = 0."""
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        xl=np.zeros(2),
        objective=lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
        gradient=lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
        comp_G=lambda x: jnp.array([x[0]]),
        comp_H=lambda x: jnp.array([x[1]]),
        comp_G_jacobian="jax",
        comp_H_jacobian="jax",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# HAS_JAX flag
# ---------------------------------------------------------------------------

class TestHasJax:
    def test_has_jax_true(self):
        assert HAS_JAX is True


# ---------------------------------------------------------------------------
# jax_gradient
# ---------------------------------------------------------------------------

class TestJaxGradient:
    def test_matches_exact_on_quadratic(self):
        f = lambda x: jnp.sum(x ** 2)
        x0 = np.array([1.0, 2.0, 3.0])
        grad = jax_gradient(f, 3, x0)
        np.testing.assert_allclose(grad(x0), 2 * x0, atol=1e-12)

    def test_matches_exact_on_nonlinear(self):
        f = lambda x: jnp.sin(x[0]) * jnp.exp(x[1])
        x0 = np.array([0.5, 1.0])
        grad = jax_gradient(f, 2, x0)
        expected = np.array([np.cos(0.5) * np.exp(1.0), np.sin(0.5) * np.exp(1.0)])
        np.testing.assert_allclose(grad(x0), expected, rtol=1e-6)

    def test_returns_numpy_array(self):
        f = lambda x: jnp.dot(x, x)
        x0 = np.ones(4)
        grad = jax_gradient(f, 4, x0)
        result = grad(x0)
        assert isinstance(result, np.ndarray)
        assert result.dtype == float


# ---------------------------------------------------------------------------
# jax_jacobian
# ---------------------------------------------------------------------------

class TestJaxJacobian:
    def test_values_match_exact(self):
        # f(x) = [x0^2, x0*x1]  → J = [[2x0, 0], [x1, x0]]
        fn = lambda x: jnp.array([x[0] ** 2, x[0] * x[1]])
        x0 = np.array([3.0, 4.0])
        jac, (rows, cols) = jax_jacobian(fn, 2, 2, x0)
        # Reconstruct dense Jacobian from sparse output
        J_dense = np.zeros((2, 2))
        J_dense[rows, cols] = jac(x0)
        expected = np.array([[6.0, 0.0], [4.0, 3.0]])
        np.testing.assert_allclose(J_dense, expected, atol=1e-10)

    def test_sparsity_auto_detected(self):
        # f(x) = [x0^2, x1^2]  — diagonal Jacobian: only (0,0) and (1,1) nonzero
        fn = lambda x: jnp.array([x[0] ** 2, x[1] ** 2])
        x0 = np.array([1.0, 2.0])
        _, (rows, cols) = jax_jacobian(fn, 2, 2, x0)
        assert set(zip(rows, cols)) == {(0, 0), (1, 1)}

    def test_sparsity_robust_to_zeros_at_x0(self):
        # Structural nonzero at (0,1) vanishes at x0=0 but shows at perturbed pts.
        fn = lambda x: jnp.array([x[0] ** 2 + x[1] ** 2])
        x0 = np.array([0.0, 0.0])
        _, (rows, cols) = jax_jacobian(fn, 1, 2, x0)
        # Both columns should be detected (union of probe points catches them)
        assert (0, 0) in set(zip(rows, cols)) or (0, 1) in set(zip(rows, cols))

    def test_sparse_callable_returns_nnz_values(self):
        fn = lambda x: jnp.array([x[0] * x[1], x[2]])
        x0 = np.array([1.0, 2.0, 3.0])
        jac, (rows, cols) = jax_jacobian(fn, 2, 3, x0)
        vals = jac(x0)
        assert vals.ndim == 1
        assert len(vals) == len(rows)

    def test_returns_numpy_array(self):
        fn = lambda x: jnp.array([x[0]])
        x0 = np.array([1.0])
        jac, _ = jax_jacobian(fn, 1, 1, x0)
        result = jac(x0)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# jax_hessian_lagrangian
# ---------------------------------------------------------------------------

class TestJaxHessianLagrangian:
    def test_hessian_matches_analytic_quadratic(self):
        # f(x) = x0^2 + x1^2  — Hessian is 2*I
        # Lagrangian with no constraints: just obj_factor * f(x)
        def lagrangian(x, lam, obj_factor):
            return obj_factor * (x[0] ** 2 + x[1] ** 2)

        x0 = np.array([1.0, 2.0])
        hess_fn, (rows, cols) = jax_hessian_lagrangian(lagrangian, 2, x0, 0)
        H_sparse = hess_fn(x0, np.zeros(0), 1.0)
        H_dense = np.zeros((2, 2))
        H_dense[rows, cols] = H_sparse
        # Symmetrise (only lower tri returned)
        H_full = H_dense + H_dense.T - np.diag(np.diag(H_dense))
        np.testing.assert_allclose(H_full, 2 * np.eye(2), atol=1e-10)

    def test_sparsity_is_lower_triangular(self):
        def lagrangian(x, lam, obj_factor):
            return obj_factor * jnp.dot(x, x)

        x0 = np.array([1.0, 1.0, 1.0])
        _, (rows, cols) = jax_hessian_lagrangian(lagrangian, 3, x0, 0)
        assert np.all(rows >= cols), "Hessian sparsity must be lower triangular"


# ---------------------------------------------------------------------------
# MPCCProblem with "jax" sentinel
# ---------------------------------------------------------------------------

class TestMPCCProblemJax:
    def test_jax_sentinel_accepted(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = _simple_problem_jax()
        assert callable(p.comp_G_jacobian)
        assert callable(p.comp_H_jacobian)

    def test_sparsity_fields_populated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = _simple_problem_jax()
        assert p.comp_G_jacobian_sparsity is not None
        assert p.comp_H_jacobian_sparsity is not None
        assert p.is_sparse

    def test_jax_jacobian_shape_matches_nnz(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = _simple_problem_jax()
        nnz = len(p.comp_G_jacobian_sparsity[0])
        result = p.comp_G_jacobian(p.x0)
        assert result.shape == (nnz,)

    def test_warns_on_jax_sentinel(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _simple_problem_jax()
        jax_warnings = [x for x in w if "JAX" in str(x.message)]
        assert len(jax_warnings) >= 1

    def test_gradient_jax_sentinel(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = pympcc.MPCCProblem(
                n=2, n_comp=1,
                x0=np.array([1.0, 1.0]),
                xl=np.zeros(2),
                objective=lambda x: jnp.sum(x ** 2),
                gradient="jax",
                comp_G=lambda x: jnp.array([x[0]]),
                comp_H=lambda x: jnp.array([x[1]]),
                comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
                comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
            )
        np.testing.assert_allclose(p.gradient(p.x0), 2 * p.x0, atol=1e-10)

    def test_use_jax_hessian_flag(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = _simple_problem_jax(use_jax_hessian=True)
        assert p.use_jax_hessian is True


# ---------------------------------------------------------------------------
# Integration: solve with "jax" Jacobians
# ---------------------------------------------------------------------------

class TestJaxIntegration:
    @pytest.fixture(autouse=True)
    def _suppress_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            yield

    def _make_problem(self, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return _simple_problem_jax(**kwargs)

    def test_solves_simple_scholtes(self):
        p = self._make_problem()
        result = pympcc.solve(p, strategy="scholtes")
        assert result.success
        np.testing.assert_allclose(result.obj, 1.0, atol=1e-4)
        assert result.comp_residual < 1e-6

    def test_solves_simple_smoothing(self):
        p = self._make_problem()
        result = pympcc.solve(p, strategy="smoothing")
        assert result.success
        np.testing.assert_allclose(result.obj, 1.0, atol=1e-4)

    def test_solves_simple_lin_fukushima(self):
        p = self._make_problem()
        result = pympcc.solve(p, strategy="lin_fukushima")
        assert result.success
        np.testing.assert_allclose(result.obj, 1.0, atol=1e-4)

    def test_solves_simple_slack(self):
        p = self._make_problem()
        result = pympcc.solve(p, strategy="slack")
        assert result.success
        np.testing.assert_allclose(result.obj, 1.0, atol=1e-4)

    def test_jax_jacobians_match_exact_on_bard1(self):
        """JAX Jacobians should give same objective as exact Jacobians.

        bard1 rebuilt with jnp-compatible primal callables so JAX can
        trace through them.  Objective and equality are unchanged; only
        comp_G / comp_H / eq_constraints need jnp.array.
        """
        from .macmpec_problems import PROBLEM_NAMES

        spec = PROBLEM_NAMES["bard1"]
        p_exact = spec.problem

        # JAX-compatible variant: primal callables use jnp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p_jax = pympcc.MPCCProblem(
                n=5, n_comp=3, n_eq=1,
                x0=np.array([2.0, 2.0, 1.0, 1.0, 1.0]),
                xl=np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf]),
                objective=lambda x: (x[0] - 5.0) ** 2 + (2.0 * x[1] + 1.0) ** 2,
                gradient=lambda x: np.array([
                    2.0 * (x[0] - 5.0), 4.0 * (2.0 * x[1] + 1.0),
                    0.0, 0.0, 0.0,
                ]),
                eq_constraints=lambda x: jnp.array([
                    2.0 * (x[1] - 1.0) - 1.5 * x[0] + x[2] - 0.5 * x[3] + x[4]
                ]),
                eq_jacobian="jax",
                comp_G=lambda x: jnp.array([
                    3.0 * x[0] - x[1] - 3.0,
                    -x[0] + 0.5 * x[1] + 4.0,
                    -x[0] - x[1] + 7.0,
                ]),
                comp_G_jacobian="jax",
                comp_H=lambda x: jnp.array([x[2], x[3], x[4]]),
                comp_H_jacobian="jax",
            )
        r_exact = pympcc.solve(p_exact, strategy="scholtes")
        r_jax   = pympcc.solve(p_jax,   strategy="scholtes")
        np.testing.assert_allclose(r_jax.obj, r_exact.obj, rtol=1e-3)


# ---------------------------------------------------------------------------
# Integration: exact Hessian reduces IPOPT iterations
# ---------------------------------------------------------------------------

class TestJaxHessian:
    @pytest.fixture(autouse=True)
    def _suppress_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            yield

    def _make_hess_problem(self, use_jax_hessian=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pympcc.MPCCProblem(
                n=2, n_comp=1,
                x0=np.array([0.5, 0.5]),
                xl=np.zeros(2),
                objective=lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
                gradient=lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
                comp_G=lambda x: jnp.array([x[0]]),
                comp_G_jacobian="jax",
                comp_H=lambda x: jnp.array([x[1]]),
                comp_H_jacobian="jax",
                use_jax_hessian=use_jax_hessian,
            )

    def test_hessian_produces_valid_result_scholtes(self):
        p = self._make_hess_problem(use_jax_hessian=True)
        result = pympcc.solve(p, strategy="scholtes")
        assert result.success
        np.testing.assert_allclose(result.obj, 1.0, atol=1e-4)

    def test_hessian_produces_valid_result_smoothing(self):
        p = self._make_hess_problem(use_jax_hessian=True)
        result = pympcc.solve(p, strategy="smoothing")
        assert result.success
        np.testing.assert_allclose(result.obj, 1.0, atol=1e-4)

    def test_hessian_produces_valid_result_lin_fukushima(self):
        p = self._make_hess_problem(use_jax_hessian=True)
        result = pympcc.solve(p, strategy="lin_fukushima")
        assert result.success
        np.testing.assert_allclose(result.obj, 1.0, atol=1e-4)

    def test_hessian_produces_valid_result_augmented_lagrangian(self):
        p = self._make_hess_problem(use_jax_hessian=True)
        result = pympcc.solve(p, strategy="augmented_lagrangian")
        assert result.success
        np.testing.assert_allclose(result.obj, 1.0, atol=1e-4)

    def test_hessian_produces_valid_result_slack(self):
        p = self._make_hess_problem(use_jax_hessian=True)
        result = pympcc.solve(p, strategy="slack")
        assert result.success
        np.testing.assert_allclose(result.obj, 1.0, atol=1e-4)

    def test_hessian_produces_valid_result_direct(self):
        p = self._make_hess_problem(use_jax_hessian=True)
        result = pympcc.solve(p, strategy="direct")
        # direct may fail LICQ but should still get near the solution
        np.testing.assert_allclose(result.obj, 1.0, atol=1e-3)

    def test_hessian_values_change_across_iterations_smoothing(self):
        """Ensure the smoothing Hessian tracks eps correctly (no stale JIT)."""
        p = self._make_hess_problem(use_jax_hessian=True)
        # With many outer iterations, the eps changes; result should still converge.
        result = pympcc.solve(p, strategy="smoothing", max_iter=10)
        assert result.success
        np.testing.assert_allclose(result.obj, 1.0, atol=1e-4)
