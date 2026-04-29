"""
pympcc — Python solver for Mathematical Programs with Complementarity Constraints

Quickstart
----------
>>> import numpy as np
>>> import pympcc
>>>
>>> problem = pympcc.MPCCProblem(
...     n=2, n_comp=1,
...     x0=np.array([0.5, 0.5]),
...     objective=lambda x: (x[0] - 2)**2 + (x[1] - 1)**2,
...     gradient=lambda x: np.array([2*(x[0]-2), 2*(x[1]-1)]),
...     comp_G=lambda x: np.array([x[0]]),
...     comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
...     comp_H=lambda x: np.array([x[1]]),
...     comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
... )
>>> result = pympcc.solve(problem, strategy='scholtes')
>>> result.x          # optimal solution
>>> result.obj        # objective value
>>> result.success    # convergence flag
"""

from ._autoscale import autoscale_comp_pairs
from ._diagnostics import active_sets, classify_cq
from ._presolve import PresolveMap, presolve
from ._sosc import sosc_check
from ._stationarity import (
    classify_stationarity,
    compute_kkt_residual,
    verify_b_stationarity,
)
from ._tnlp import TNLPResult
from .models import StructuredMPCC
from .multistart import MultiStartResult, multistart
from .problem import MPCCProblem
from .result import IPOPTStatus, IterationInfo, MPCCResult, unscale_multipliers
from .solver import MPCCSolver, solve

__all__ = [
    "MPCCProblem",
    "StructuredMPCC",
    "MPCCResult",
    "IterationInfo",
    "IPOPTStatus",
    "MPCCSolver",
    "solve",
    "active_sets",
    "classify_cq",
    "sosc_check",
    "classify_stationarity",
    "compute_kkt_residual",
    "verify_b_stationarity",
    "unscale_multipliers",
    "presolve",
    "PresolveMap",
    "autoscale_comp_pairs",
    "multistart",
    "MultiStartResult",
    "TNLPResult",
]
__version__ = "0.4.2"
