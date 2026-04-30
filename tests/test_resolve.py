"""Tests for MPCCSolver.resolve() — stateful warm hot-start (§6.5)."""
from __future__ import annotations

import numpy as np
import pytest

import pympcc


# ======================================================================= #
# Problem builder — a parametric MPCC family used to exercise resolve().    #
#                                                                           #
#     min   (x0 - a)^2 + (x1 - b)^2                                          #
#     s.t.  x >= 0,  x0 * x1 = 0                                             #
#                                                                           #
# Global optimum: (a, 0) when a > b, else (0, b).  Sweeping (a, b) lets us  #
# verify that warm-restarts track the true optimum across parameter steps.  #
# ======================================================================= #


def _make_parametric(a: float, b: float, x0: np.ndarray | None = None
                     ) -> pympcc.MPCCProblem:
    if x0 is None:
        x0 = np.array([0.5, 0.5])
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=x0,
        xl=np.zeros(2),
        objective=lambda x, a=a, b=b: (x[0] - a) ** 2 + (x[1] - b) ** 2,
        gradient=lambda x, a=a, b=b: np.array([2 * (x[0] - a), 2 * (x[1] - b)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


# ----------------------------------------------------------------------- #
# Sanity                                                                    #
# ----------------------------------------------------------------------- #


def test_resolve_requires_prior_solve():
    """resolve() before solve() is a usage error, not a silent cold start."""
    solver = pympcc.MPCCSolver(_make_parametric(2.0, 1.0))
    with pytest.raises(RuntimeError, match="prior solve"):
        solver.resolve(_make_parametric(2.5, 1.0))


def test_resolve_basic_param_sweep_scholtes():
    """Sweeping (a, b) returns to the parametric optimum at every step."""
    solver = pympcc.MPCCSolver(_make_parametric(2.0, 1.0), strategy="scholtes")
    r0 = solver.solve()
    assert r0.success
    assert np.isclose(r0.x[0], 2.0, atol=1e-3) and np.isclose(r0.x[1], 0.0, atol=1e-3)
    assert r0.warmstart_savings_iter is None  # cold baseline
    assert r0.n_ipopt_iter_total is not None and r0.n_ipopt_iter_total > 0

    for a in (2.1, 2.2, 2.3):
        r = solver.resolve(_make_parametric(a, 1.0))
        assert r.success
        assert np.isclose(r.x[0], a, atol=1e-3)
        assert np.isclose(r.x[1], 0.0, atol=1e-3)
        assert r.warmstart_savings_iter is not None
        assert r.n_ipopt_iter_total is not None and r.n_ipopt_iter_total > 0


def test_resolve_basic_param_sweep_direct():
    """Direct strategy also accepts warm-restarts (single-shot NLP)."""
    solver = pympcc.MPCCSolver(_make_parametric(2.0, 1.0), strategy="direct")
    r0 = solver.solve()
    # Direct may report status=3 (small step) at LICQ-failing points but
    # x* should still match the global optimum.
    assert np.isclose(r0.x[0], 2.0, atol=1e-3)
    r1 = solver.resolve(_make_parametric(2.5, 1.0))
    assert np.isclose(r1.x[0], 2.5, atol=1e-3)
    assert r1.warmstart_savings_iter is not None


# ----------------------------------------------------------------------- #
# Iter savings                                                              #
# ----------------------------------------------------------------------- #


def test_resolve_saves_iterations_on_small_perturbation():
    """A tiny parameter step should converge faster than the cold baseline."""
    solver = pympcc.MPCCSolver(
        _make_parametric(2.0, 1.0), strategy="scholtes")
    r0 = solver.solve()
    cold_iters = r0.n_ipopt_iter_total
    # Perturb a by 1e-4 — warm restart should see almost no work.
    r1 = solver.resolve(_make_parametric(2.0001, 1.0))
    assert r1.success
    assert r1.n_ipopt_iter_total <= cold_iters
    assert r1.warmstart_savings_iter == cold_iters - r1.n_ipopt_iter_total


# ----------------------------------------------------------------------- #
# Structure-change cold fallback                                            #
# ----------------------------------------------------------------------- #


def test_resolve_cold_fallback_on_dimension_change():
    """Changing n forces a cold rebuild + a UserWarning."""
    solver = pympcc.MPCCSolver(_make_parametric(2.0, 1.0))
    r0 = solver.solve()

    # Build a problem with one extra (free) variable but the same comp pair.
    p3 = pympcc.MPCCProblem(
        n=3, n_comp=1,
        x0=np.array([0.5, 0.5, 0.5]),
        xl=np.zeros(3),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2 + (x[2] - 0.3) ** 2,
        gradient=lambda x: np.array([2 * (x[0] - 2.0), 2 * (x[1] - 1.0), 2 * (x[2] - 0.3)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0, 0.0]]),
    )
    with pytest.warns(UserWarning, match="structure changed"):
        r1 = solver.resolve(p3)
    assert r1.success
    assert r1.x.shape == (3,)
    # Cold rebuild → savings field is None (new baseline established).
    assert r1.warmstart_savings_iter is None


def test_resolve_cold_fallback_on_n_comp_change():
    """Adding a comp pair triggers cold fallback."""
    solver = pympcc.MPCCSolver(_make_parametric(2.0, 1.0))
    solver.solve()
    # Same n=2 but two comp pairs with the same variable indices duplicated —
    # an artificial structural change that flips the signature.
    p_twocomp = pympcc.MPCCProblem(
        n=2, n_comp=2,
        x0=np.array([0.5, 0.5]),
        xl=np.zeros(2),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2 * (x[0] - 2.0), 2 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0], x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0], [1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1], x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    with pytest.warns(UserWarning, match="structure changed"):
        r1 = solver.resolve(p_twocomp)
    assert r1.success


# ----------------------------------------------------------------------- #
# Warm-start opt-outs                                                       #
# ----------------------------------------------------------------------- #


def test_resolve_warm_x0_disabled_uses_problem_x0():
    """warm_x0=False respects the new problem's own x0."""
    solver = pympcc.MPCCSolver(_make_parametric(2.0, 1.0))
    solver.solve()
    new_p = _make_parametric(2.5, 1.0, x0=np.array([0.1, 0.1]))
    # x0 mutation is detectable: capture it before resolve() and check it
    # wasn't overwritten by the previous solve's x*.
    pre_x0 = new_p.x0.copy()
    r1 = solver.resolve(new_p, warm_x0=False)
    assert r1.success
    assert np.allclose(new_p.x0, pre_x0)


def test_resolve_warm_dual_disabled():
    """warm_dual=False still seeds x but doesn't forward multipliers."""
    solver = pympcc.MPCCSolver(_make_parametric(2.0, 1.0))
    solver.solve()
    r1 = solver.resolve(_make_parametric(2.5, 1.0), warm_dual=False)
    assert r1.success
    assert np.isclose(r1.x[0], 2.5, atol=1e-3)


# ----------------------------------------------------------------------- #
# Multi-step continuation                                                   #
# ----------------------------------------------------------------------- #


def test_resolve_chain_three_steps_smoothing():
    """Smoothing strategy chains three resolves without losing accuracy."""
    solver = pympcc.MPCCSolver(_make_parametric(2.0, 1.0), strategy="smoothing")
    solver.solve()
    for a in (2.1, 2.2, 2.3, 2.4):
        r = solver.resolve(_make_parametric(a, 1.0))
        assert r.success
        assert np.isclose(r.x[0], a, atol=1e-3)
        assert r.comp_residual < 1e-4


def test_resolve_x0_clipped_to_new_bounds():
    """Bound tightening: x_seed is clipped onto the new box."""
    solver = pympcc.MPCCSolver(_make_parametric(5.0, 1.0))
    r0 = solver.solve()
    assert np.isclose(r0.x[0], 5.0, atol=1e-3)
    # New problem with xu[0]=3.0 — the previous x*=5 must be clipped to 3.
    new_p = _make_parametric(5.0, 1.0)
    new_p.xu = np.array([3.0, np.inf])
    r1 = solver.resolve(new_p)
    assert r1.success
    assert r1.x[0] <= 3.0 + 1e-6
    assert np.isclose(r1.x[0], 3.0, atol=1e-3)


# ----------------------------------------------------------------------- #
# Result fields are populated for cold solves too                           #
# ----------------------------------------------------------------------- #


def test_solve_populates_n_ipopt_iter_total():
    """Cold solves populate n_ipopt_iter_total (savings is None)."""
    solver = pympcc.MPCCSolver(_make_parametric(2.0, 1.0))
    r = solver.solve()
    assert r.n_ipopt_iter_total is not None
    assert r.n_ipopt_iter_total > 0
    assert r.warmstart_savings_iter is None


def test_solve_populates_iter_total_direct():
    """Direct strategy: iter total comes from the strategy side-channel."""
    solver = pympcc.MPCCSolver(_make_parametric(2.0, 1.0), strategy="direct")
    r = solver.solve()
    assert r.n_ipopt_iter_total is not None
    assert r.n_ipopt_iter_total > 0
