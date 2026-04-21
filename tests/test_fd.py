"""
Tests for the finite-difference Jacobian fallback.

Covers:
- pympcc._fd unit tests (fd_gradient, fd_jacobian accuracy)
- Warning / error behaviour on MPCCProblem and StructuredMPCC
- Integration tests: solving real MPCC problems with FD Jacobians
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pympcc
from pympcc._fd import _DEFAULT_H, fd_gradient, fd_jacobian
from pympcc.models import StructuredMPCC


# ------------------------------------------------------------------ #
# 1. Unit tests — fd_gradient                                          #
# ------------------------------------------------------------------ #

class TestFdGradient:
    x0 = np.array([1.0, 2.0, -0.5])

    def test_quadratic_forward(self):
        f = lambda x: float(x @ x)
        grad = fd_gradient(f, n=3, mode="forward")
        exact = 2 * self.x0
        assert np.allclose(grad(self.x0), exact, atol=1e-5)

    def test_quadratic_central(self):
        f = lambda x: float(x @ x)
        grad = fd_gradient(f, n=3, mode="central")
        exact = 2 * self.x0
        assert np.allclose(grad(self.x0), exact, atol=1e-8)

    def test_nonlinear_forward(self):
        # f(x) = exp(x[0]) + sin(x[1])
        f = lambda x: float(np.exp(x[0]) + np.sin(x[1]))
        grad = fd_gradient(f, n=2, mode="forward")
        x = np.array([0.5, 1.0])
        exact = np.array([np.exp(0.5), np.cos(1.0)])
        assert np.allclose(grad(x), exact, atol=1e-5)

    def test_nonlinear_central(self):
        f = lambda x: float(np.exp(x[0]) + np.sin(x[1]))
        grad = fd_gradient(f, n=2, mode="central")
        x = np.array([0.5, 1.0])
        exact = np.array([np.exp(0.5), np.cos(1.0)])
        assert np.allclose(grad(x), exact, atol=1e-8)

    def test_output_shape(self):
        f = lambda x: float(x @ x)
        grad = fd_gradient(f, n=4)
        result = grad(np.ones(4))
        assert result.shape == (4,)
        assert result.dtype == float

    def test_custom_h(self):
        f = lambda x: float(x @ x)
        grad = fd_gradient(f, n=2, h=1e-5, mode="forward")
        x = np.array([1.0, 2.0])
        assert np.allclose(grad(x), 2 * x, atol=1e-4)


# ------------------------------------------------------------------ #
# 2. Unit tests — fd_jacobian                                          #
# ------------------------------------------------------------------ #

class TestFdJacobian:
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    x0 = np.array([1.0, -1.0, 0.5])

    def test_linear_forward(self):
        fn = lambda x: self.A @ x
        jac = fd_jacobian(fn, n_out=2, n=3, mode="forward")
        assert np.allclose(jac(self.x0), self.A, atol=1e-5)

    def test_linear_central(self):
        fn = lambda x: self.A @ x
        jac = fd_jacobian(fn, n_out=2, n=3, mode="central")
        assert np.allclose(jac(self.x0), self.A, atol=1e-8)

    def test_nonlinear_forward(self):
        # fn(x) = [x[0]^2 * x[1], sin(x[0]) + x[1]^2]
        fn = lambda x: np.array([x[0]**2 * x[1], np.sin(x[0]) + x[1]**2])
        jac = fd_jacobian(fn, n_out=2, n=2, mode="forward")
        x = np.array([1.0, 2.0])
        exact = np.array([
            [2*x[0]*x[1], x[0]**2],
            [np.cos(x[0]), 2*x[1]],
        ])
        assert np.allclose(jac(x), exact, atol=1e-5)

    def test_nonlinear_central(self):
        fn = lambda x: np.array([x[0]**2 * x[1], np.sin(x[0]) + x[1]**2])
        jac = fd_jacobian(fn, n_out=2, n=2, mode="central")
        x = np.array([1.0, 2.0])
        exact = np.array([
            [2*x[0]*x[1], x[0]**2],
            [np.cos(x[0]), 2*x[1]],
        ])
        assert np.allclose(jac(x), exact, atol=1e-8)

    def test_output_shape(self):
        fn = lambda x: self.A @ x
        jac = fd_jacobian(fn, n_out=2, n=3)
        result = jac(self.x0)
        assert result.shape == (2, 3)
        assert result.dtype == float

    def test_scalar_output(self):
        # n_out=1 edge case
        fn = lambda x: np.array([x[0]**2 + x[1]])
        jac = fd_jacobian(fn, n_out=1, n=2, mode="central")
        x = np.array([2.0, 3.0])
        exact = np.array([[2*x[0], 1.0]])
        assert np.allclose(jac(x), exact, atol=1e-8)


# ------------------------------------------------------------------ #
# 3. Warning / error behaviour — MPCCProblem                           #
# ------------------------------------------------------------------ #

def _base_simple_kwargs(**overrides):
    """Minimal 'simple' problem kwargs; overrides replace individual fields."""
    kw = dict(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        xl=np.zeros(2),
        objective=lambda x: (x[0]-2)**2 + (x[1]-1)**2,
        gradient=lambda x: np.array([2*(x[0]-2), 2*(x[1]-1)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )
    kw.update(overrides)
    return kw


class TestMPCCProblemFDWarningsErrors:
    def test_fd_sentinel_emits_warning(self):
        with pytest.warns(UserWarning, match="comp_G_jacobian"):
            pympcc.MPCCProblem(**_base_simple_kwargs(comp_G_jacobian="fd"))

    def test_warning_lists_all_fd_fields(self):
        with pytest.warns(UserWarning, match="gradient") as rec:
            pympcc.MPCCProblem(**_base_simple_kwargs(
                gradient="fd",
                comp_G_jacobian="fd",
                comp_H_jacobian="fd",
            ))
        msg = str(rec[0].message)
        assert "comp_G_jacobian" in msg
        assert "comp_H_jacobian" in msg

    def test_ineq_jacobian_fd_without_constraints_raises(self):
        with pytest.raises(ValueError, match="ineq_jacobian"):
            pympcc.MPCCProblem(**_base_simple_kwargs(
                n_ineq=0, ineq_jacobian="fd"
            ))

    def test_eq_jacobian_fd_without_constraints_raises(self):
        with pytest.raises(ValueError, match="eq_jacobian"):
            pympcc.MPCCProblem(**_base_simple_kwargs(
                n_eq=0, eq_jacobian="fd"
            ))

    def test_invalid_fd_mode_raises(self):
        with pytest.raises(ValueError, match="fd_mode"):
            pympcc.MPCCProblem(**_base_simple_kwargs(
                comp_G_jacobian="fd", fd_mode="bad"
            ))

    def test_no_warning_when_no_fd(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise
            pympcc.MPCCProblem(**_base_simple_kwargs())


# ------------------------------------------------------------------ #
# 4. Warning / error behaviour — StructuredMPCC                        #
# ------------------------------------------------------------------ #

class TestStructuredMPCCFDWarningsErrors:
    def _base_kwargs(self, **overrides):
        kw = dict(
            n=3, n_comp=1,
            x0=np.array([0.5, 0.5, 0.75]),
            xl=np.zeros(3),
            objective=lambda x: float(x @ x),
            gradient=lambda x: 2*x,
            A_eq=np.array([[1.0, 1.0, 0.0]]),
            b_eq=np.array([1.0]),
            n_nl_eq=1,
            eq_nl=lambda x: np.array([x[0]**2 + x[2] - 1.0]),
            jac_eq_nl=lambda x: np.array([[2*x[0], 0.0, 1.0]]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0, 0.0]]),
        )
        kw.update(overrides)
        return kw

    def test_fd_sentinel_emits_warning(self):
        with pytest.warns(UserWarning, match="jac_eq_nl"):
            StructuredMPCC(**self._base_kwargs(jac_eq_nl="fd"))

    def test_jac_eq_nl_fd_without_n_nl_eq_raises(self):
        with pytest.raises(ValueError, match="jac_eq_nl"):
            StructuredMPCC(**self._base_kwargs(
                n_nl_eq=0, eq_nl=None, jac_eq_nl="fd"
            ))

    def test_jac_ineq_nl_fd_without_n_nl_ineq_raises(self):
        with pytest.raises(ValueError, match="jac_ineq_nl"):
            StructuredMPCC(**self._base_kwargs(
                n_nl_ineq=0, ineq_nl=None, jac_ineq_nl="fd"
            ))

    def test_invalid_fd_mode_raises(self):
        with pytest.raises(ValueError, match="fd_mode"):
            StructuredMPCC(**self._base_kwargs(
                comp_G_jacobian="fd", fd_mode="bad"
            ))


# ------------------------------------------------------------------ #
# 5. Integration tests — MPCCProblem with FD Jacobians                 #
# ------------------------------------------------------------------ #

def _make_simple_fd(gradient_fd=False, comp_G_fd=False, comp_H_fd=False,
                    mode="forward"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            xl=np.zeros(2),
            objective=lambda x: (x[0]-2)**2 + (x[1]-1)**2,
            gradient="fd" if gradient_fd else lambda x: np.array([2*(x[0]-2), 2*(x[1]-1)]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian="fd" if comp_G_fd else lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian="fd" if comp_H_fd else lambda x: np.array([[0.0, 1.0]]),
            fd_mode=mode,
        )


@pytest.mark.parametrize("gradient_fd,comp_G_fd,comp_H_fd,mode", [
    (False, False, False, "forward"),   # baseline (exact)
    (False, True,  True,  "forward"),   # comp FD, forward
    (False, True,  True,  "central"),   # comp FD, central
    (True,  True,  True,  "forward"),   # all FD, forward
    (True,  True,  True,  "central"),   # all FD, central
])
def test_simple_problem_fd(gradient_fd, comp_G_fd, comp_H_fd, mode):
    problem = _make_simple_fd(gradient_fd, comp_G_fd, comp_H_fd, mode)
    result = pympcc.solve(problem, strategy="scholtes")
    assert result.success
    assert result.comp_residual < 1e-4
    assert abs(result.obj - 1.0) < 1e-3


def test_eq_jacobian_fd_bard1():
    """bard1 (n=5, n_comp=3, n_eq=1) with eq_jacobian='fd'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        problem = pympcc.MPCCProblem(
            n=5, n_comp=3, n_eq=1,
            x0=np.array([2.0, 2.0, 1.0, 1.0, 1.0]),
            xl=np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf]),
            objective=lambda x: (x[0]-5.0)**2 + (2.0*x[1]+1.0)**2,
            gradient=lambda x: np.array([2.0*(x[0]-5.0), 4.0*(2.0*x[1]+1.0), 0.0, 0.0, 0.0]),
            eq_constraints=lambda x: np.array([
                2.0*(x[1]-1.0) - 1.5*x[0] + x[2] - 0.5*x[3] + x[4]
            ]),
            eq_jacobian="fd",
            comp_G=lambda x: np.array([3.0*x[0]-x[1]-3.0, -x[0]+0.5*x[1]+4.0, -x[0]-x[1]+7.0]),
            comp_G_jacobian=lambda x: np.array([
                [ 3.0, -1.0, 0.0, 0.0, 0.0],
                [-1.0,  0.5, 0.0, 0.0, 0.0],
                [-1.0, -1.0, 0.0, 0.0, 0.0],
            ]),
            comp_H=lambda x: np.array([x[2], x[3], x[4]]),
            comp_H_jacobian=lambda x: np.eye(3, 5, k=2),
        )
    result = pympcc.solve(problem, strategy="scholtes")
    assert result.success
    assert result.comp_residual < 1e-4
    assert abs(result.obj - 17.0) < 1e-2


# ------------------------------------------------------------------ #
# 6. Integration tests — StructuredMPCC with FD Jacobians              #
# ------------------------------------------------------------------ #

def test_structured_mpcc_jac_eq_nl_fd():
    """'mixed' problem with jac_eq_nl='fd' and comp_G_jacobian='fd'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = StructuredMPCC(
            n=3, n_comp=1,
            x0=np.array([0.5, 0.5, 0.75]),
            xl=np.zeros(3),
            objective=lambda x: float(x @ x),
            gradient=lambda x: 2*x,
            A_eq=np.array([[1.0, 1.0, 0.0]]),
            b_eq=np.array([1.0]),
            n_nl_eq=1,
            eq_nl=lambda x: np.array([x[0]**2 + x[2] - 1.0]),
            jac_eq_nl="fd",
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian="fd",
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0, 0.0]]),
        )
    result = pympcc.solve(model, strategy="scholtes")
    assert result.success
    assert result.comp_residual < 1e-4
    assert abs(result.obj - 1.0) < 1e-3


# ------------------------------------------------------------------ #
# 7. Custom step size                                                   #
# ------------------------------------------------------------------ #

def test_custom_fd_h():
    """Coarse step fd_h=1e-5 still allows construction and convergence."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        problem = pympcc.MPCCProblem(
            **_base_simple_kwargs(
                comp_G_jacobian="fd",
                comp_H_jacobian="fd",
                fd_h=1e-5,
            )
        )
    result = pympcc.solve(problem, strategy="scholtes")
    assert result.success
