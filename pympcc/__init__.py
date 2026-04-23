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

from ._stationarity import classify_stationarity, compute_kkt_residual
from .models import StructuredMPCC
from .problem import MPCCProblem
from .result import IPOPTStatus, IterationInfo, MPCCResult
from .solver import MPCCSolver, solve

__all__ = [
    "MPCCProblem",
    "StructuredMPCC",
    "MPCCResult",
    "IterationInfo",
    "IPOPTStatus",
    "MPCCSolver",
    "solve",
    "classify_stationarity",
    "compute_kkt_residual",
]
__version__ = "0.1.0"
