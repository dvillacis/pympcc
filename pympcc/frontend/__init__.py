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

__all__ = ["ampl", "pyomo"]
