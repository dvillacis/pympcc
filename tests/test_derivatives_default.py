"""Tests for the ``derivatives`` keyword on MPCCProblem / StructuredMPCC."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pympcc
from pympcc.problem import MPCCProblem


# ======================================================================= #
# Validation                                                                #
# ======================================================================= #

class TestValidation:
    def test_invalid_derivatives_value(self):
        with pytest.raises(ValueError, match="derivatives must be"):
            MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: x[0],
                comp_G=lambda x: np.array([x[0]]),
                comp_H=lambda x: np.array([x[1]]),
                derivatives="autodiff",  # invalid
            )

    def test_missing_derivative_without_keyword(self):
        with pytest.raises(ValueError, match="Missing derivative callable"):
            MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: x[0],
                comp_G=lambda x: np.array([x[0]]),
                comp_H=lambda x: np.array([x[1]]),
                # gradient/comp_G_jacobian/comp_H_jacobian all None,
                # no derivatives keyword to fill them
            )

    def test_partial_user_supplied_no_keyword(self):
        # gradient supplied but jacobians missing — should still error.
        with pytest.raises(ValueError, match="comp_G_jacobian"):
            MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: x[0],
                gradient=lambda x: np.array([1.0, 0.0]),
                comp_G=lambda x: np.array([x[0]]),
                comp_H=lambda x: np.array([x[1]]),
            )


# ======================================================================= #
# derivatives="fd" — finite differences fill everything                     #
# ======================================================================= #

class TestDerivativesFd:
    def test_fills_all_derivative_fields(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2),
                comp_G=lambda x: np.array([x[0]]),
                comp_H=lambda x: np.array([x[1]]),
                derivatives="fd",
            )
        assert callable(p.gradient)
        assert callable(p.comp_G_jacobian)
        assert callable(p.comp_H_jacobian)

    def test_solves_with_fd(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2),
                comp_G=lambda x: np.array([x[0]]),
                comp_H=lambda x: np.array([x[1]]),
                derivatives="fd",
            )
        result = pympcc.solve(p, strategy="scholtes")
        assert result.success
        # Optimum is x=(2,0) with obj=1.0
        assert abs(result.obj - 1.0) < 1e-2

    def test_user_supplied_overrides_fd(self):
        # User-supplied gradient must not be replaced.
        sentinel_grad = np.array([42.0, 7.0])
        captured = {"called": False}

        def grad(x):
            captured["called"] = True
            return sentinel_grad.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: float(x[0] + x[1]),
                gradient=grad,
                comp_G=lambda x: np.array([x[0]]),
                comp_H=lambda x: np.array([x[1]]),
                derivatives="fd",
            )
        np.testing.assert_array_equal(p.gradient(np.zeros(2)), sentinel_grad)
        assert captured["called"]

    def test_includes_inequality_jacobian(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: float(x[0] + x[1]),
                comp_G=lambda x: np.array([x[0]]),
                comp_H=lambda x: np.array([x[1]]),
                n_ineq=1,
                ineq_constraints=lambda x: np.array([x[0] + x[1] - 10.0]),
                derivatives="fd",
            )
        assert callable(p.ineq_jacobian)


# ======================================================================= #
# derivatives="jax" — JAX autodiff fills everything                         #
# ======================================================================= #

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestDerivativesJax:
    def test_fills_all_derivative_fields(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: jnp.sum((x - jnp.array([2.0, 1.0])) ** 2),
                comp_G=lambda x: jnp.array([x[0]]),
                comp_H=lambda x: jnp.array([x[1]]),
                derivatives="jax",
            )
        assert callable(p.gradient)
        assert callable(p.comp_G_jacobian)
        assert callable(p.comp_H_jacobian)
        # Sparsity should be auto-detected by jax_jacobian
        assert p.comp_G_jacobian_sparsity is not None
        assert p.comp_H_jacobian_sparsity is not None

    def test_gradient_matches_analytical(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: jnp.sum((x - jnp.array([2.0, 1.0])) ** 2),
                comp_G=lambda x: jnp.array([x[0]]),
                comp_H=lambda x: jnp.array([x[1]]),
                derivatives="jax",
            )
        x = np.array([0.5, 0.5])
        np.testing.assert_allclose(p.gradient(x), 2 * (x - np.array([2.0, 1.0])),
                                   rtol=1e-6)

    def test_solves_with_jax(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: jnp.sum((x - jnp.array([2.0, 1.0])) ** 2),
                comp_G=lambda x: jnp.array([x[0]]),
                comp_H=lambda x: jnp.array([x[1]]),
                derivatives="jax",
            )
        result = pympcc.solve(p, strategy="scholtes")
        assert result.success
        assert abs(result.obj - 1.0) < 1e-2

    def test_user_supplied_overrides_jax(self):
        # User passes explicit gradient; only the missing fields get JAX.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: jnp.sum(x ** 2),
                gradient=lambda x: 2 * np.asarray(x),  # plain numpy
                comp_G=lambda x: jnp.array([x[0]]),
                comp_H=lambda x: jnp.array([x[1]]),
                derivatives="jax",
            )
        x = np.array([1.0, 2.0])
        np.testing.assert_allclose(p.gradient(x), np.array([2.0, 4.0]))

    def test_existing_jax_sentinel_still_works(self):
        # Backward compat: sentinel-based opt-in must keep working without
        # the derivatives keyword.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: jnp.sum(x ** 2),
                gradient=lambda x: 2 * np.asarray(x),
                comp_G=lambda x: jnp.array([x[0]]),
                comp_G_jacobian="jax",
                comp_H=lambda x: jnp.array([x[1]]),
                comp_H_jacobian="jax",
            )
        assert callable(p.comp_G_jacobian)
        assert callable(p.comp_H_jacobian)


# ======================================================================= #
# StructuredMPCC parity                                                     #
# ======================================================================= #

class TestStructuredMpcc:
    def test_fd_fills_structured(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = pympcc.StructuredMPCC(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2),
                comp_G=lambda x: np.array([x[0]]),
                comp_H=lambda x: np.array([x[1]]),
                derivatives="fd",
            )
        assert callable(m.gradient)
        assert callable(m.comp_G_jacobian)
        result = pympcc.solve(m, strategy="scholtes")
        assert result.success

    def test_invalid_value_structured(self):
        with pytest.raises(ValueError, match="derivatives must be"):
            pympcc.StructuredMPCC(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: x[0],
                comp_G=lambda x: np.array([x[0]]),
                comp_H=lambda x: np.array([x[1]]),
                derivatives="oops",
            )
