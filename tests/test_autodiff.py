"""Tests for §5.6 JAX-differentiable solve (``pympcc.solve_jax``)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from pympcc._jax import HAS_JAX

if not HAS_JAX:
    pytest.skip("JAX not installed", allow_module_level=True)

import jax
import jax.numpy as jnp

import pympcc
from pympcc import ParametricMPCC, solve_jax

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _branch_select_pmpcc() -> ParametricMPCC:
    """2-pair quadratic with θ shifting the objective targets:

        min ½‖x − θ‖²
        s.t. x0 ≥ 0 ⊥ x2 ≥ 0,  x1 ≥ 0 ⊥ x3 ≥ 0
    """
    return ParametricMPCC(
        n=4,
        n_comp=2,
        objective=lambda x, theta: 0.5 * jnp.sum((x - theta) ** 2),
        comp_G=lambda x, theta: x[:2],
        comp_H=lambda x, theta: x[2:],
    )


def _eq_box_pmpcc() -> ParametricMPCC:
    """4-var equality NLP with one trivial comp pair, θ-driven RHS:

        min ½(x-1)² + ½(y-1)² + ½(s² + t²)
        s.t. x + y = θ,  s ≥ 0 ⊥ t ≥ 0
    """
    return ParametricMPCC(
        n=4,
        n_comp=1,
        n_eq=1,
        objective=lambda z, theta: 0.5 * (
            (z[0] - 1) ** 2 + (z[1] - 1) ** 2 + z[2] ** 2 + z[3] ** 2
        ),
        eq_constraints=lambda z, theta: jnp.array([z[0] + z[1] - theta[0]]),
        comp_G=lambda z, theta: z[2:3],
        comp_H=lambda z, theta: z[3:4],
    )


def _biactive_pmpcc() -> ParametricMPCC:
    """Biactive optimum: min θ·s + θ·t  s.t. s ≥ 0 ⊥ t ≥ 0  with θ > 0.

    Optimum at s = t = 0 (biactive); IFT prerequisites fail.
    """
    return ParametricMPCC(
        n=2,
        n_comp=1,
        objective=lambda x, theta: theta[0] * x[0] + theta[0] * x[1],
        comp_G=lambda x, theta: x[:1],
        comp_H=lambda x, theta: x[1:],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParametricMPCCMaterialise:
    def test_baked_problem_solves(self):
        pmpcc = _branch_select_pmpcc()
        theta_np = np.array([2.0, 0.0, 0.0, 3.0])
        problem = pmpcc.materialise(theta_np, x0=theta_np)
        result = pympcc.solve(problem, strategy="scholtes", tnlp_refine=True)
        assert result.success
        np.testing.assert_allclose(result.x, theta_np, atol=1e-6)

    def test_validation_n_eq_requires_eq_constraints(self):
        with pytest.raises(ValueError, match="eq_constraints"):
            ParametricMPCC(
                n=2, n_comp=1, n_eq=1,
                objective=lambda x, t: 0.0,
                comp_G=lambda x, t: x[:1],
                comp_H=lambda x, t: x[1:],
            )


class TestSolveJaxForwardOnly:
    def test_returns_jax_array(self):
        pmpcc = _branch_select_pmpcc()
        theta = jnp.array([2.0, 0.0, 0.0, 3.0])
        x_star = solve_jax(pmpcc, theta, x0=np.array([2.0, 0.0, 0.0, 3.0]))
        assert isinstance(x_star, jnp.ndarray)
        np.testing.assert_allclose(np.asarray(x_star), np.array(theta), atol=1e-6)

    def test_matches_pympcc_solve(self):
        pmpcc = _branch_select_pmpcc()
        theta = np.array([2.0, 0.0, 0.0, 3.0])
        x_jax = np.asarray(solve_jax(pmpcc, jnp.asarray(theta), x0=theta))
        problem = pmpcc.materialise(theta, x0=theta)
        result = pympcc.solve(problem, strategy="scholtes")
        np.testing.assert_allclose(x_jax, result.x, atol=1e-8)


class TestEqualityClosedFormGradient:
    """For the eq-box problem, x* = θ/2, so dx*/dθ = 1/2."""

    @pytest.mark.parametrize("theta_val", [0.5, 1.0, 1.5])
    def test_grad_matches_closed_form(self, theta_val):
        pmpcc = _eq_box_pmpcc()
        # Loss = sum(x*); dL/dθ = sum(dx*/dθ) = (1/2 + 1/2 + 0 + 0) = 1.0
        def loss(theta):
            x = solve_jax(
                pmpcc,
                theta,
                x0=jnp.array([theta_val / 2, theta_val / 2, 0.0, 0.0]),
            )
            return jnp.sum(x)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            g = jax.grad(loss)(jnp.array([theta_val]))
        np.testing.assert_allclose(np.asarray(g), np.array([1.0]), atol=1e-5)


class TestBranchSelectGradientFD:
    """Sensitivity matches a finite-difference re-solve."""

    def test_objective_only_param(self):
        pmpcc = _branch_select_pmpcc()
        theta0 = np.array([2.0, 0.0, 0.0, 3.0])

        def loss(theta):
            x = solve_jax(pmpcc, theta, x0=theta0)
            return 0.5 * jnp.sum(x ** 2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            g_jax = np.asarray(jax.grad(loss)(jnp.asarray(theta0)))

        # FD reference: re-solve at θ ± ε e_k.
        eps = 1e-3
        g_fd = np.zeros_like(theta0)
        for k in range(theta0.size):
            theta_p = theta0.copy()
            theta_p[k] += eps
            theta_m = theta0.copy()
            theta_m[k] -= eps
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                x_p = np.asarray(
                    solve_jax(pmpcc, jnp.asarray(theta_p), x0=theta_p)
                )
                x_m = np.asarray(
                    solve_jax(pmpcc, jnp.asarray(theta_m), x0=theta_m)
                )
            l_p = 0.5 * float(np.sum(x_p ** 2))
            l_m = 0.5 * float(np.sum(x_m ** 2))
            g_fd[k] = (l_p - l_m) / (2 * eps)

        np.testing.assert_allclose(g_jax, g_fd, atol=1e-3, rtol=1e-3)

    def test_full_dx_dtheta_via_per_row_grad(self):
        # ``jax.jacrev`` internally ``vmap``s the bwd, which our NumPy/IPOPT
        # bwd cannot support (Phase-2).  Build the Jacobian one row at a
        # time via ``jax.grad`` instead.
        pmpcc = _branch_select_pmpcc()
        theta0 = np.array([2.0, 0.0, 0.0, 3.0])

        def x_i(theta, i):
            return solve_jax(pmpcc, theta, x0=theta0)[i]

        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for i in range(theta0.size):
                row = jax.grad(lambda th, i=i: x_i(th, i))(jnp.asarray(theta0))
                rows.append(np.asarray(row))
        J = np.stack(rows, axis=0)
        # For x0 = (2,0,0,3) the solution pins x1=0 (G-active) and x2=0
        # (H-active), so:  dx0/da = 1, dx3/dd = 1, others zero.
        expected = np.diag([1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(J, expected, atol=1e-4)


class TestSkippedPaths:
    def test_biactive_returns_zero_gradient(self):
        pmpcc = _biactive_pmpcc()

        def loss(theta):
            x = solve_jax(pmpcc, theta, x0=np.array([0.5, 0.5]))
            return jnp.sum(x)

        with pytest.warns(UserWarning, match="biactive|did not converge"):
            g = jax.grad(loss)(jnp.array([1.0]))
        np.testing.assert_allclose(np.asarray(g), 0.0, atol=1e-12)


class TestJitIncompatibility:
    """``solve_jax`` is not jittable — this is documented behaviour."""

    def test_jit_raises(self):
        pmpcc = _branch_select_pmpcc()
        theta0 = np.array([2.0, 0.0, 0.0, 3.0])

        @jax.jit
        def loss(theta):
            x = solve_jax(pmpcc, theta, x0=theta0)
            return jnp.sum(x)

        # Under jit, NumPy/IPOPT inside the fwd cannot trace.
        with pytest.raises(Exception):
            loss(jnp.asarray(theta0))
