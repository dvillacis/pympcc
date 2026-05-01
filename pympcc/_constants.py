"""Centralised numerical tolerances used across pympcc.

Single source of truth for the magic numbers that previously sat
inline in `_presolve.py`, `_diagnostics.py`, `solver.py`, `_tnlp.py`,
and `strategies/_base.py`.

Values are empirical defaults tuned against the MacMPEC reference
suite and the published reference implementations.  They are not
user-configurable from this module — callers that need different
values pass explicit keyword arguments to the public functions
(e.g. `pympcc.classify_cq(..., tol=...)`) which still default to the
same numeric value.  Keeping the symbol here lets a future call site
import the same baseline without re-deriving it.

See `ROADMAP.md` §7.2.5 for the audit context.
"""
from __future__ import annotations

from typing import Final

# -- Active set / stationarity ----------------------------------- #

#: Default biactive / active-set tolerance for stationarity, SOSC,
#: and CQ classification (kwarg default for the public API).
BIACTIVE_TOL: Final[float] = 1e-6

#: Floor used by the per-pair adaptive biactive threshold
#: ``max(sqrt(comp) * (1 + BIACTIVE_TOL_FLOOR), BIACTIVE_TOL_FLOOR)``.
#: A flat 1e-6 cutoff misclassifies near-biactive points where both
#: G_i and H_i are tiny but non-zero; the adaptive form catches them.
BIACTIVE_TOL_FLOOR: Final[float] = 1e-6

# -- IPOPT / strategy tolerances --------------------------------- #

#: Default value for IPOPT's ``tol`` option.
IPOPT_DEFAULT_TOL: Final[float] = 1e-8

#: Cleanup-phase inner-tol floor.  IPOPT spins chasing noise below
#: this; the cleanup polish floors here.
CLEANUP_TOL_FLOOR: Final[float] = 1e-6

#: KKT-residual threshold used by the safeguard-KKT-termination
#: check during ε-continuation.
KKT_TERMINATION_TOL: Final[float] = 1e-6

#: Hard floor on the inner IPOPT tol when ε-continuation is in
#: ``matched`` mode; below this IPOPT cannot reliably converge.
INNER_TOL_FLOOR: Final[float] = 1e-10

# -- Diagnostics ------------------------------------------------- #

#: Vector norm below which a multiplier / Lagrangian gradient is
#: treated as identically zero in CQ classification.
ZERO_NORM_TOL: Final[float] = 1e-10

# -- Presolve ---------------------------------------------------- #

#: Comp-pair value below which a pair with structurally empty
#: Jacobian sparsity row is treated as dead (presolve B1 / B2).
DEAD_VAL_TOL: Final[float] = 1e-12

#: Tolerance for a structurally empty constraint row to be treated
#: as feasible at ``x0``.  Violations emit a UserWarning and fall
#: back to identity.
EMPTY_ROW_TOL: Final[float] = 1e-9

#: ``|∂f/∂x_j|`` at probe points must stay below this for variable
#: ``j`` to be flagged as effectively free in dead-column detection.
FREE_VAR_GRAD_TOL: Final[float] = 1e-10

#: Bounds must improve by more than this on a single FBBT iteration
#: to count as progress.
FBBT_TOL: Final[float] = 1e-9

#: Linearity probe tolerance: a row is judged linear when the
#: residual ``g(x0+δ) - g(x0) - J(x0)·δ`` falls below
#: ``LINEARITY_TOL * scale``.
LINEARITY_TOL: Final[float] = 1e-9

# -- Cancellation guards ----------------------------------------- #

#: Divide-by-near-zero floor for ratio diagnostics.  Not a
#: tolerance — a safety net so the ratio is well-defined when the
#: denominator collapses.
EPS_DIV_GUARD: Final[float] = 1e-12

__all__ = [
    "BIACTIVE_TOL",
    "BIACTIVE_TOL_FLOOR",
    "IPOPT_DEFAULT_TOL",
    "CLEANUP_TOL_FLOOR",
    "KKT_TERMINATION_TOL",
    "INNER_TOL_FLOOR",
    "ZERO_NORM_TOL",
    "DEAD_VAL_TOL",
    "EMPTY_ROW_TOL",
    "FREE_VAR_GRAD_TOL",
    "FBBT_TOL",
    "LINEARITY_TOL",
    "EPS_DIV_GUARD",
]
