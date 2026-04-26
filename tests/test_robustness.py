"""Edge-case and robustness tests for MPCCProblem validation and solver safeguards."""
from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc.strategies.direct import DirectStrategy

# --------------------------------------------------------------------------- #
# Minimal problem factory                                                       #
# --------------------------------------------------------------------------- #

def _base_kwargs() -> dict:
    """Keyword arguments for a valid 2-variable, 1-comp problem."""
    return dict(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        xl=np.zeros(2),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2 * (x[0] - 2.0), 2 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


# --------------------------------------------------------------------------- #
# Construction-time NaN / Inf guards                                            #
# --------------------------------------------------------------------------- #

def test_nan_objective_raises():
    kw = _base_kwargs()
    kw["objective"] = lambda x: float("nan")
    with pytest.raises(ValueError, match="objective.*non-finite"):
        pympcc.MPCCProblem(**kw)


def test_inf_objective_raises():
    kw = _base_kwargs()
    kw["objective"] = lambda x: float("inf")
    with pytest.raises(ValueError, match="objective.*non-finite"):
        pympcc.MPCCProblem(**kw)


def test_nan_comp_G_raises():
    kw = _base_kwargs()
    kw["comp_G"] = lambda x: np.array([float("nan")])
    with pytest.raises(ValueError, match="comp_G.*non-finite"):
        pympcc.MPCCProblem(**kw)


def test_inf_comp_H_raises():
    kw = _base_kwargs()
    kw["comp_H"] = lambda x: np.array([float("inf")])
    with pytest.raises(ValueError, match="comp_H.*non-finite"):
        pympcc.MPCCProblem(**kw)


def test_nan_dense_jacobian_raises():
    kw = _base_kwargs()
    kw["comp_G_jacobian"] = lambda x: np.array([[float("nan"), 0.0]])
    with pytest.raises(ValueError, match="comp_G_jacobian.*non-finite"):
        pympcc.MPCCProblem(**kw)


def test_nan_sparse_jacobian_raises():
    kw = _base_kwargs()
    kw["comp_G_jacobian"] = lambda x: np.array([float("nan")])
    kw["comp_G_jacobian_sparsity"] = (np.array([0]), np.array([0]))
    with pytest.raises(ValueError, match="comp_G_jacobian.*non-finite"):
        pympcc.MPCCProblem(**kw)


def test_duplicate_sparse_jacobian_entries_raise():
    kw = _base_kwargs()
    kw["comp_G_jacobian"] = lambda x: np.array([1.0, 2.0])
    kw["comp_G_jacobian_sparsity"] = (np.array([0, 0]), np.array([0, 0]))
    with pytest.raises(ValueError, match="duplicate"):
        pympcc.MPCCProblem(**kw)


def test_callback_cache_reuses_values_at_same_x():
    calls = {
        "g": 0,
        "h": 0,
        "jg": 0,
        "jh": 0,
        "ineq": 0,
        "jineq": 0,
        "eq": 0,
        "jeq": 0,
    }

    def comp_g(x):
        calls["g"] += 1
        return np.array([x[0]])

    def comp_h(x):
        calls["h"] += 1
        return np.array([x[1]])

    def jac_g(x):
        calls["jg"] += 1
        return np.array([[1.0, 0.0]])

    def jac_h(x):
        calls["jh"] += 1
        return np.array([[0.0, 1.0]])

    def ineq(x):
        calls["ineq"] += 1
        return np.array([x[0] + x[1] - 2.0])

    def jac_ineq(x):
        calls["jineq"] += 1
        return np.array([[1.0, 1.0]])

    def eq(x):
        calls["eq"] += 1
        return np.array([x[0] - x[1]])

    def jac_eq(x):
        calls["jeq"] += 1
        return np.array([[1.0, -1.0]])

    problem = pympcc.MPCCProblem(
        **{
            **_base_kwargs(),
            "comp_G": comp_g,
            "comp_G_jacobian": jac_g,
            "comp_H": comp_h,
            "comp_H_jacobian": jac_h,
            "n_ineq": 1,
            "ineq_constraints": ineq,
            "ineq_jacobian": jac_ineq,
            "n_eq": 1,
            "eq_constraints": eq,
            "eq_jacobian": jac_eq,
        }
    )
    for key in calls:
        calls[key] = 0

    strategy = DirectStrategy(problem, {})
    cache = strategy._new_callback_cache()
    x = problem.x0.copy()

    strategy._eval_comp_values(x, cache)
    strategy._eval_comp_values(x, cache)
    strategy._eval_comp_jac_raw(x, cache)
    strategy._build_comp_jacobians(x, cache)
    strategy._eval_standard_con_values(x, cache)
    strategy._eval_standard_con_values(x, cache)
    strategy._build_std_jac_flat(x, cache)
    strategy._build_std_jac_flat(x, cache)

    assert calls == {
        "g": 1,
        "h": 1,
        "jg": 1,
        "jh": 1,
        "ineq": 1,
        "jineq": 1,
        "eq": 1,
        "jeq": 1,
    }

    strategy._eval_comp_values(x + 1.0, cache)
    assert calls["g"] == 2
    assert calls["h"] == 2


def test_nan_ineq_constraint_raises():
    kw = _base_kwargs()
    kw["n_ineq"] = 1
    kw["ineq_constraints"] = lambda x: np.array([float("nan")])
    kw["ineq_jacobian"] = lambda x: np.zeros((1, 2))
    with pytest.raises(ValueError, match="ineq_constraints.*non-finite"):
        pympcc.MPCCProblem(**kw)


# --------------------------------------------------------------------------- #
# n_comp validation                                                             #
# --------------------------------------------------------------------------- #

def test_n_comp_zero_raises():
    kw = _base_kwargs()
    kw["n_comp"] = 0
    # comp_G / comp_H still needed for validation; use empty arrays
    kw["comp_G"] = lambda x: np.empty(0)
    kw["comp_G_jacobian"] = lambda x: np.empty((0, 2))
    kw["comp_H"] = lambda x: np.empty(0)
    kw["comp_H_jacobian"] = lambda x: np.empty((0, 2))
    with pytest.raises(ValueError, match="n_comp"):
        pympcc.MPCCProblem(**kw)


# --------------------------------------------------------------------------- #
# Mid-loop IPOPT failure warning                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strategy", ["scholtes", "smoothing", "lin_fukushima"])
def test_ipopt_failure_emits_warning(strategy):
    """Forcing IPOPT to hit max_iter=1 must emit a UserWarning, not silently continue."""
    problem = pympcc.MPCCProblem(**_base_kwargs())
    with pytest.warns(UserWarning, match="MAX_ITER"):
        pympcc.solve(
            problem,
            strategy=strategy,
            ipopt_options={"max_iter": 1},
            max_iter=2,       # two outer iterations so at least one inner solve fails
            epsilon_0=0.5,
        )


def test_ipopt_failure_emits_warning_augmented_lagrangian():
    """AL builds one NLP and calls it repeatedly; inner failures must still warn."""
    problem = pympcc.MPCCProblem(**_base_kwargs())
    with pytest.warns(UserWarning, match="MAX_ITER"):
        pympcc.solve(
            problem,
            strategy="augmented_lagrangian",
            ipopt_options={"max_iter": 1},
            max_iter=2,
        )


@pytest.mark.parametrize(
    ("strategy", "bad_options", "match"),
    [
        ("scholtes", {"max_iter": 0}, "max_iter"),
        ("smoothing", {"epsilon_0": 0.0}, "epsilon_0"),
        ("lin_fukushima", {"reduction": 1.0}, "reduction"),
        ("slack", {"epsilon_min": 2.0}, "epsilon_min"),
        ("augmented_lagrangian", {"rho_0": 0.0}, "rho_0"),
        ("augmented_lagrangian", {"rho_0": 10.0, "rho_max": 1.0}, "rho_max"),
        ("augmented_lagrangian", {"eta": -1.0}, "eta"),
    ],
)
def test_invalid_strategy_options_raise(strategy, bad_options, match):
    problem = pympcc.MPCCProblem(**_base_kwargs())
    with pytest.raises(ValueError, match=match):
        pympcc.MPCCSolver(problem, strategy=strategy, **bad_options)
