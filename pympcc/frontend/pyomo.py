"""Pyomo frontend for pympcc.

Builds a :class:`pympcc.MPCCProblem` from a :class:`pyomo.ConcreteModel`,
including models that use :class:`pyomo.mpec.Complementarity` blocks.

Approach
--------
The frontend applies the ``mpec.nl`` transformation (which converts each
``Complementarity`` block into a pair of constraints linked through AMPL's
bound-type-5 ``cvar`` mapping), writes the model out as an AMPL ``.nl``
file via Pyomo's NL writer, then parses the result with the existing
:func:`pympcc.frontend.ampl.from_nl` reader.

Complementarity is therefore encoded the same way MacMPEC encodes it,
and every existing ``.nl``-driven test in :mod:`pympcc.frontend.ampl`
exercises the Pyomo path too.

Usage
-----
.. code-block:: python

    import pyomo.environ as pyo
    from pyomo.mpec import Complementarity, complements
    import pympcc
    from pympcc.frontend.pyomo import from_pyomo, apply_solution

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=0.5)
    m.y = pyo.Var(bounds=(0, None), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - 2)**2 + (m.y - 1)**2)
    m.cc  = Complementarity(expr=complements(m.x >= 0, m.y >= 0))

    pmpcc = from_pyomo(m)
    result = pympcc.solve(pmpcc.problem, strategy="scholtes")
    apply_solution(m, result.x, pmpcc.var_index)

Limitations
-----------
* Tempfile roundtrip; not suitable for hot-loop reconstruction
  (see roadmap §6.5 stateful warm hot-start).
* Variable-to-x ordering is dictated by Pyomo's NL writer, not the user's
  declaration order in the model.  ``PyomoMPCC.var_index`` maps each
  Pyomo qualified variable name back to its index in
  ``problem.x0`` / ``result.x``.
* Variables introduced by the ``mpec.nl`` transformation (per-block
  binding variables, typically named ``cc.bv``) appear in ``problem``.
  They are part of the lifted problem and should be ignored by the user.
* Pyomo is an optional dependency; install with
  ``pip install pympcc[pyomo]``.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any

from .ampl import from_nl

__all__ = ["from_pyomo", "apply_solution", "PyomoMPCC"]


_PYOMO_INSTALL_HINT = (
    "Pyomo is required for pympcc.frontend.pyomo; "
    "install with `pip install pympcc[pyomo]` (or `pip install pyomo`)."
)


def _require_pyomo():
    try:
        import pyomo.environ as pyo
        from pyomo.environ import TransformationFactory
        from pyomo.mpec import Complementarity
    except ImportError as exc:
        raise ImportError(_PYOMO_INSTALL_HINT) from exc
    return pyo, TransformationFactory, Complementarity


@dataclass
class PyomoMPCC:
    """Result of :func:`from_pyomo`.

    Attributes
    ----------
    problem
        Numeric :class:`pympcc.MPCCProblem` ready for :func:`pympcc.solve`.
    var_index
        Maps Pyomo qualified variable name (e.g. ``"x"``, ``"y[0]"``) to
        its index in ``problem.x0`` / ``result.x``.  Includes the binding
        variables introduced by the ``mpec.nl`` transformation.
    con_index
        Maps Pyomo qualified constraint name to its index in the NL file's
        constraint list.  Useful for diagnostics; typically not needed by
        end users.
    transformed_model
        The (possibly cloned) Pyomo model after the ``mpec.nl``
        transformation has been applied.  When ``from_pyomo(model)`` is
        called with the default ``clone=True``, this is a copy and the
        user's original ``model`` is left untouched.
    """

    problem: Any
    var_index: dict[str, int]
    con_index: dict[str, int]
    transformed_model: Any = None


def from_pyomo(model, *, clone: bool = True) -> PyomoMPCC:
    """Build an :class:`pympcc.MPCCProblem` from a Pyomo model.

    Parameters
    ----------
    model
        A Pyomo ``ConcreteModel`` containing ``Var``, ``Objective``,
        ``Constraint``, and optionally :class:`pyomo.mpec.Complementarity`
        blocks.
    clone
        When ``True`` (default), the model is cloned before any
        transformation, so the user's original model is left untouched.
        Set to ``False`` to apply the ``mpec.nl`` transformation in place
        (useful when the model is already a temporary copy).

    Returns
    -------
    PyomoMPCC
        Carries the numeric problem plus name-to-index mappings.

    Raises
    ------
    ImportError
        When Pyomo is not installed.
    """
    pyo, TransformationFactory, Complementarity = _require_pyomo()

    if clone:
        model = model.clone()

    cc_blocks = list(model.component_objects(Complementarity, active=True))
    if not cc_blocks:
        raise ValueError(
            "pympcc.frontend.pyomo.from_pyomo: model has no active "
            "Complementarity blocks; pympcc only solves problems with "
            "complementarity constraints.  For pure NLPs use IPOPT "
            "(via cyipopt) or another NLP solver directly."
        )
    TransformationFactory("mpec.nl").apply_to(model)

    with tempfile.TemporaryDirectory() as td:
        nl_path = os.path.join(td, "model.nl")
        _, smap_id = model.write(nl_path, format="nl")
        problem = from_nl(nl_path)

    var_index, con_index = _build_label_maps(model, smap_id)
    return PyomoMPCC(
        problem=problem,
        var_index=var_index,
        con_index=con_index,
        transformed_model=model,
    )


def apply_solution(model, x, var_index: dict[str, int]) -> None:
    """Write ``x`` back into ``model`` by qualified variable name.

    Variables in ``var_index`` that the model does not expose
    (for example, the ``mpec.nl`` binding variables when the caller
    passes the *original* untransformed model) are skipped silently.

    Parameters
    ----------
    model
        The Pyomo model whose ``Var`` values to update.  Either the
        original or the ``transformed_model`` returned by
        :func:`from_pyomo` works.
    x
        Solution vector — typically ``result.x`` from :func:`pympcc.solve`.
    var_index
        Name-to-index mapping from :class:`PyomoMPCC.var_index`.
    """
    for name, idx in var_index.items():
        var = model.find_component(name)
        if var is None:
            continue
        try:
            var.set_value(float(x[idx]))
        except (AttributeError, TypeError, ValueError):
            continue


def _build_label_maps(model, smap_id) -> tuple[dict[str, int], dict[str, int]]:
    """Walk Pyomo's symbol map to build {qualified_name: nl_index} dicts.

    Pyomo's NL writer assigns symbols ``v0, v1, …`` to variables and
    ``c0, c1, …`` to constraints in NL-file order.  We invert the map to
    produce ``{qualified_name: index}`` for both.
    """
    smap = model.solutions.symbol_map[smap_id]
    var_index: dict[str, int] = {}
    con_index: dict[str, int] = {}
    for symbol, obj in smap.bySymbol.items():
        if not symbol or not hasattr(obj, "getname"):
            continue
        if symbol.startswith("v"):
            try:
                idx = int(symbol[1:])
            except ValueError:
                continue
            var_index[obj.getname(fully_qualified=True)] = idx
        elif symbol.startswith("c"):
            try:
                idx = int(symbol[1:])
            except ValueError:
                continue
            con_index[obj.getname(fully_qualified=True)] = idx
    return var_index, con_index
