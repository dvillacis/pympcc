from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

__all__ = ["IPOPTStatus", "IterationInfo", "MPCCResult", "unscale_multipliers"]


class IPOPTStatus(IntEnum):
    """IPOPT solver return codes (integer values match cyipopt's status field)."""
    SOLVED        =    0   # Solve_Succeeded
    ACCEPTABLE    =    1   # Solved_To_Acceptable_Level
    INFEASIBLE    =    2   # Infeasible_Problem_Detected
    SMALL_STEP    =    3   # Search_Direction_Becomes_Too_Small
    DIVERGING     =    4   # Diverging_Iterates
    USER_STOP     =    5   # User_Requested_Stop
    MAX_ITER      =   -1   # Maximum_Iterations_Exceeded
    RESTORE_FAIL  =   -2   # Restoration_Failed
    STEP_ERROR    =   -3   # Error_In_Step_Computation
    MAX_TIME      =   -4   # Maximum_CpuTime_Exceeded
    NO_DOF        =  -10   # Not_Enough_Degrees_Of_Freedom
    BAD_PROBLEM   =  -11   # Invalid_Problem_Definition
    BAD_OPTION    =  -12   # Invalid_Option
    BAD_NUMBER    =  -13   # Invalid_Number_Detected
    EXCEPTION     = -100   # Unrecoverable_Exception
    NO_MEMORY     = -102   # Insufficient_Memory
    INTERNAL_ERR  = -199   # Internal_Error


@dataclass
class IterationInfo:
    """Diagnostic snapshot for one NLP solve in an iterative strategy."""

    epsilon: float
    x: np.ndarray
    obj: float
    status: int
    message: str
    comp_residual: float       # max_i |G_i * H_i|
    comp_residual_mean: float  # mean_i |G_i * H_i|
    n_ipopt_iter: int          # IPOPT iterations in this NLP solve
    iter_time: float           # wall-clock seconds for this NLP solve
    kkt_residual: Optional[float] = None  # MPCC stationarity residual (∞-norm)
    restoration_iter_count: int = 0       # IPOPT iterations in feasibility restoration
    entered_restoration: bool = False      # True if any iter ran in restoration mode


@dataclass
class MPCCResult:
    """
    Result returned by :func:`pympcc.solve` or :meth:`pympcc.MPCCSolver.solve`.

    Attributes
    ----------
    x : ndarray
        Solution vector.
    obj : float
        Objective value at the solution.
    status : int
        IPOPT exit code of the final NLP solve.
        Common values: 0 = Solve_Succeeded, 1 = Solved_To_Acceptable_Level,
        -1 = Maximum_Iterations_Exceeded, 2 = Infeasible_Problem_Detected.
    message : str
        Human-readable status from IPOPT.
    G : ndarray, shape (n_comp,)
        G(x*) values at the solution.
    H : ndarray, shape (n_comp,)
        H(x*) values at the solution.
    comp_residual : float
        Complementarity feasibility measure: ``max_i |G_i(x*) * H_i(x*)|``.
    comp_residual_mean : float
        Mean complementarity residual: ``mean_i |G_i(x*) * H_i(x*)|``.
    success : bool
        ``True`` when IPOPT status is 0 (Solve_Succeeded) or
        1 (Solved_To_Acceptable_Level).
    strategy : str
        Name of the reformulation strategy used.
    solve_time : float or None
        Sum of wall-clock seconds spent inside each ``nlp.solve()`` call.
        Excludes strategy setup (NLP construction, sparsity computation),
        callback invocations, and post-solve stationarity classification.
        ``None`` when not measured (e.g. results constructed programmatically).
    history : list[IterationInfo]
        Per-iteration diagnostics for iterative strategies (Scholtes,
        smoothing).  Empty for the direct strategy.
    """

    x: np.ndarray
    obj: float
    status: int
    message: str
    G: np.ndarray
    H: np.ndarray
    comp_residual: float
    comp_residual_mean: float
    success: bool
    strategy: str
    history: list[IterationInfo] = field(default_factory=list)
    solve_time: Optional[float] = None
    mult_g: Optional[np.ndarray] = field(default=None)
    stationarity: str = "unknown"
    kkt_residual: Optional[float] = None
    # Active-set cleanup phase (post-continuation polish solve).
    # Populated only when the strategy was invoked with cleanup={True,"auto"}.
    cleanup_status: Optional[int] = None          # IPOPT status of cleanup NLP
    cleanup_n_iter: Optional[int] = None          # IPOPT iters in cleanup NLP
    cleanup_obj: Optional[float] = None           # objective at cleanup solution
    cleanup_active_set: Optional[tuple] = None    # (I_G_active, I_H_active)
    cleanup_accepted: Optional[bool] = None       # True iff cleanup result replaces continuation
    # Complementarity-pair diagonal scaling that was active during the solve.
    # ``None`` when the problem was solved without rescaling.  Multipliers
    # ``mpcc_mult_G`` / ``mpcc_mult_H`` returned via ``mult_g`` are in
    # *scaled* space; multiply by these vectors to recover original-space
    # KKT duals.  See :func:`pympcc.unscale_multipliers`.
    comp_G_scale: Optional[np.ndarray] = None
    comp_H_scale: Optional[np.ndarray] = None
    # Constraint-qualification diagnostics (populated when the solver was
    # invoked with ``diagnostics=True``).  See ``pympcc._diagnostics``.
    cq: Optional[str] = None                       # "MPCC-LICQ" | "MPCC-MFCQ" | "none" | "unknown"
    cq_active_set_sizes: Optional[dict] = None     # {"g","h","G","H","biactive","xL","xU"}
    cq_rank_deficit: Optional[int] = None          # 0 ⇔ LICQ
    # B-stationarity diagnostics (populated alongside ``cq``).  See
    # ``pympcc._stationarity.verify_b_stationarity``.
    b_stationary: Optional[str] = None             # "B-stationary" | "not B-stationary" | "intractable" | "unknown"
    b_stationary_witness: Optional[tuple] = None   # branch chars where descent was found
    b_stationary_min_descent: Optional[float] = None

    def unscale_comp_multipliers(
        self,
        mpcc_mult_G: np.ndarray,
        mpcc_mult_H: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert scaled-space comp multipliers back to original space.

        If the problem was solved with ``comp_G_scale``/``comp_H_scale``
        active, IPOPT's multipliers correspond to the *scaled* constraint
        ``s · G(x) ≥ 0``.  Original-space duals satisfy ``μ = s · μ̃``
        (chain rule).  Returns the input unchanged if no scaling was used.
        """
        s_G = self.comp_G_scale if self.comp_G_scale is not None else 1.0
        s_H = self.comp_H_scale if self.comp_H_scale is not None else 1.0
        return s_G * mpcc_mult_G, s_H * mpcc_mult_H

    def __repr__(self) -> str:  # pragma: no cover
        kkt_str = (f", kkt_residual={self.kkt_residual:.3e}"
                   if self.kkt_residual is not None else "")
        return (
            f"MPCCResult(strategy={self.strategy!r}, success={self.success}, "
            f"obj={self.obj:.6g}, comp_residual={self.comp_residual:.3e}, "
            f"comp_residual_mean={self.comp_residual_mean:.3e}, "
            f"stationarity={self.stationarity!r}{kkt_str}, status={self.status})"
        )


def unscale_multipliers(
    result: "MPCCResult",
    mpcc_mult_G: np.ndarray,
    mpcc_mult_H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Module-level alias for :meth:`MPCCResult.unscale_comp_multipliers`."""
    return result.unscale_comp_multipliers(mpcc_mult_G, mpcc_mult_H)
