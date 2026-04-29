"""User-facing frontends that build :class:`pympcc.MPCCProblem` from
external problem specifications.

Currently provides:

* :mod:`pympcc.frontend.ampl` — reader for AMPL ``.nl`` files (text format),
  including the ``cvar`` complementarity-variable suffix used by MacMPEC.
"""

from . import ampl

__all__ = ["ampl"]
