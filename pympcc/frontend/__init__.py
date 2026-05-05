"""User-facing frontends that build :class:`pympcc.MPCCProblem` from
external problem specifications.

Currently provides:

* :mod:`pympcc.frontend.ampl` — reader for AMPL ``.nl`` files (text format),
  including the ``cvar`` complementarity-variable suffix used by MacMPEC.
* :mod:`pympcc.frontend.pyomo` — reader for Pyomo ``ConcreteModel`` (with
  optional ``Complementarity`` blocks).  Requires the ``pyomo`` extra
  (``pip install pympcc[pyomo]``); the module imports lazily so the
  Pyomo dependency is only loaded when the frontend is used.
"""

from . import ampl, pyomo
from .ampl import from_nl
from .pyomo import PyomoMPCC, apply_solution, from_pyomo

__all__ = [
    "ampl",
    "pyomo",
    "from_nl",
    "from_pyomo",
    "apply_solution",
    "PyomoMPCC",
]
