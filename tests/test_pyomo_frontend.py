"""Tests for :mod:`pympcc.frontend.pyomo`.

Pyomo is an optional dependency (`pip install pympcc[pyomo]`); the entire
file is skipped when Pyomo is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pyomo = pytest.importorskip("pyomo")
pyo = pytest.importorskip("pyomo.environ")
mpec_module = pytest.importorskip("pyomo.mpec")

import pympcc  # noqa: E402
from pympcc.frontend.pyomo import apply_solution, from_pyomo  # noqa: E402

Complementarity = mpec_module.Complementarity
complements = mpec_module.complements


# --------------------------------------------------------------------------- #
# Pure-NLP path (no Complementarity blocks)
# --------------------------------------------------------------------------- #


def test_pure_nlp_rejected():
    """A model without Complementarity should be rejected with a clear error."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=2.0)
    m.obj = pyo.Objective(expr=(m.x - 1) ** 2)

    with pytest.raises(ValueError, match="no active Complementarity"):
        from_pyomo(m)


# --------------------------------------------------------------------------- #
# Complementarity path
# --------------------------------------------------------------------------- #


def test_simple_mpcc_x_perp_y():
    """min (x-2)^2 + (y-1)^2 s.t. x>=0 perp y>=0.

    Optimum: x=2, y=0, obj=1.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=0.5)
    m.y = pyo.Var(bounds=(0, None), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - 2) ** 2 + (m.y - 1) ** 2)
    m.cc = Complementarity(expr=complements(m.x >= 0, m.y >= 0))

    out = from_pyomo(m)
    assert out.problem.n_comp >= 1

    result = pympcc.solve(out.problem, strategy="scholtes")
    assert result.success
    assert result.x[out.var_index["x"]] == pytest.approx(2.0, abs=1e-5)
    assert result.x[out.var_index["y"]] == pytest.approx(0.0, abs=1e-5)
    assert result.obj == pytest.approx(1.0, abs=1e-4)


def test_clone_preserves_original_model():
    """from_pyomo(model, clone=True) must not mutate the user's model."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=0.5)
    m.y = pyo.Var(bounds=(0, None), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - 2) ** 2 + (m.y - 1) ** 2)
    m.cc = Complementarity(expr=complements(m.x >= 0, m.y >= 0))

    cc_before = list(m.component_objects(Complementarity, active=True))
    from_pyomo(m, clone=True)
    cc_after = list(m.component_objects(Complementarity, active=True))

    assert len(cc_before) == 1
    assert len(cc_after) == 1


def test_clone_false_mutates_in_place():
    """clone=False applies mpec.nl directly; Complementarity is rewritten."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=0.5)
    m.y = pyo.Var(bounds=(0, None), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - 2) ** 2 + (m.y - 1) ** 2)
    m.cc = Complementarity(expr=complements(m.x >= 0, m.y >= 0))

    from_pyomo(m, clone=False)
    # mpec.nl deactivates the original Complementarity block.
    cc_active = [
        c
        for c in m.component_objects(Complementarity, active=True)
        if c.active
    ]
    assert cc_active == []


# --------------------------------------------------------------------------- #
# apply_solution helper
# --------------------------------------------------------------------------- #


def test_apply_solution_writes_back_to_original_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=0.5)
    m.y = pyo.Var(bounds=(0, None), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - 2) ** 2 + (m.y - 1) ** 2)
    m.cc = Complementarity(expr=complements(m.x >= 0, m.y >= 0))

    out = from_pyomo(m)
    result = pympcc.solve(out.problem, strategy="scholtes")

    apply_solution(m, result.x, out.var_index)
    assert pyo.value(m.x) == pytest.approx(2.0, abs=1e-5)
    assert pyo.value(m.y) == pytest.approx(0.0, abs=1e-5)


def test_apply_solution_skips_unknown_names():
    """Names absent from the model are silently skipped."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    apply_solution(m, np.array([0.0, 7.0]), {"x": 1, "nonexistent": 0})
    assert pyo.value(m.x) == pytest.approx(7.0)


# --------------------------------------------------------------------------- #
# Indexed variables
# --------------------------------------------------------------------------- #


def test_indexed_variables_get_qualified_names():
    """Indexed Vars should appear in var_index with their qualified names."""
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=[0, 1])
    m.x = pyo.Var(m.I, bounds=(0, None), initialize=1.0)
    m.y = pyo.Var(bounds=(0, None), initialize=0.5)
    m.obj = pyo.Objective(
        expr=sum((m.x[i] - (i + 1)) ** 2 for i in m.I) + (m.y - 1) ** 2
    )
    m.cc = Complementarity(expr=complements(m.x[0] >= 0, m.y >= 0))

    out = from_pyomo(m)
    assert "x[0]" in out.var_index
    assert "x[1]" in out.var_index
    assert "y" in out.var_index

    result = pympcc.solve(out.problem, strategy="scholtes")
    apply_solution(m, result.x, out.var_index)
    # x[1] is unconstrained beyond bounds → drives toward 2.
    assert pyo.value(m.x[1]) == pytest.approx(2.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# transformed_model field
# --------------------------------------------------------------------------- #


def test_transformed_model_is_attached():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=0.5)
    m.y = pyo.Var(bounds=(0, None), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - 2) ** 2 + (m.y - 1) ** 2)
    m.cc = Complementarity(expr=complements(m.x >= 0, m.y >= 0))

    out = from_pyomo(m)
    assert out.transformed_model is not None
    # mpec.nl introduces a 'cc.bv' binding variable on the transformed model.
    assert any("bv" in name for name in out.var_index)
