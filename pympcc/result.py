from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal, Optional

import numpy as np

__all__ = ["IPOPTStatus", "IterationInfo", "MPCCResult", "unscale_multipliers"]

PairStatus = Literal["G_active", "H_active", "biactive", "inactive"]


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
    # Per-pair complementarity status.  Always populated by MPCCSolver.  Each
    # entry is one of: "G_active" (G_i≈0, H_i>0), "H_active" (H_i≈0, G_i>0),
    # "biactive" (both≈0), "inactive" (both>0 — infeasible, should not occur).
    per_pair_status: Optional[list] = None
    # MPCC-stationarity multipliers μ_G = −λ_G, μ_H = −λ_H (literature
    # sign convention).  Populated when ``tnlp_refine=True`` from the TNLP
    # active-set re-solve; None otherwise to avoid confusion with approximate
    # relaxed-NLP multipliers.
    mult_comp_G_mpcc: Optional[np.ndarray] = None
    mult_comp_H_mpcc: Optional[np.ndarray] = None
    # TNLP active-set refinement result (§2.6).  Populated when the solver is
    # called with ``tnlp_refine=True``.
    tnlp_refined: Optional[Any] = None  # TNLPResult, typed as Any to avoid circular import
    # MPCC second-order sufficient conditions (§2.3).  Populated when the
    # solver is invoked with ``diagnostics=True``.
    # ``True``  — reduced Hessian is PD on the critical cone.
    # ``False`` — at least one non-positive eigenvalue (not a strict local min).
    # ``None``  — check skipped (non-converged, biactive pairs, or Hessian
    #             unavailable); see ``sosc_skipped_reason``.
    sosc: Optional[bool] = None
    sosc_min_eigenvalue: Optional[float] = None
    sosc_skipped_reason: Optional[str] = None

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

    def to_json(self) -> str:
        """Serialize the result to a JSON string.

        Arrays are converted to lists.  Fields that are ``None`` are included
        as JSON ``null``.  The ``history`` list is omitted (use
        ``summary(verbosity=2)`` for human-readable history).

        Returns
        -------
        str
            JSON-encoded result suitable for logging, benchmarking, or storage.
        """
        def _arr(a):
            return a.tolist() if isinstance(a, np.ndarray) else a

        d: dict = {
            "obj": self.obj,
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "strategy": self.strategy,
            "comp_residual": self.comp_residual,
            "comp_residual_mean": self.comp_residual_mean,
            "stationarity": self.stationarity,
            "kkt_residual": self.kkt_residual,
            "x": _arr(self.x),
            "G": _arr(self.G),
            "H": _arr(self.H),
            "per_pair_status": self.per_pair_status,
            "mult_comp_G_mpcc": _arr(self.mult_comp_G_mpcc),
            "mult_comp_H_mpcc": _arr(self.mult_comp_H_mpcc),
            "cq": self.cq,
            "cq_rank_deficit": self.cq_rank_deficit,
            "b_stationary": self.b_stationary,
            "solve_time": self.solve_time,
        }
        if self.tnlp_refined is not None:
            tnlp = self.tnlp_refined
            d["tnlp_refined"] = {
                "success": tnlp.success,
                "status": tnlp.status,
                "obj": tnlp.obj,
                "stationarity": tnlp.stationarity,
                "kkt_residual": tnlp.kkt_residual,
                "mult_comp_G": _arr(tnlp.mult_comp_G),
                "mult_comp_H": _arr(tnlp.mult_comp_H),
                "n_iter": tnlp.n_iter,
                "solve_time": tnlp.solve_time,
            }
        return json.dumps(d)

    def to_dataframe(self):
        """Return a per-complementarity-pair :class:`pandas.DataFrame`.

        Columns: ``pair``, ``G``, ``H``, ``GH``, ``status``, and (when
        available from TNLP refinement) ``mu_G``, ``mu_H``.

        Requires ``pandas``.

        Returns
        -------
        pandas.DataFrame
            One row per complementarity pair.

        Raises
        ------
        ImportError
            When ``pandas`` is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for MPCCResult.to_dataframe(). "
                "Install it with: pip install pandas"
            ) from exc

        rows = []
        for i, (g, h) in enumerate(zip(self.G, self.H)):
            row: dict = {
                "pair": i,
                "G": float(g),
                "H": float(h),
                "GH": float(g * h),
                "status": (self.per_pair_status[i]
                           if self.per_pair_status is not None else None),
            }
            if self.mult_comp_G_mpcc is not None:
                row["mu_G"] = float(self.mult_comp_G_mpcc[i])
            if self.mult_comp_H_mpcc is not None:
                row["mu_H"] = float(self.mult_comp_H_mpcc[i])
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # pragma: no cover
        kkt_str = (f", kkt_residual={self.kkt_residual:.3e}"
                   if self.kkt_residual is not None else "")
        return (
            f"MPCCResult(strategy={self.strategy!r}, success={self.success}, "
            f"obj={self.obj:.6g}, comp_residual={self.comp_residual:.3e}, "
            f"comp_residual_mean={self.comp_residual_mean:.3e}, "
            f"stationarity={self.stationarity!r}{kkt_str}, status={self.status})"
        )

    def summary(self, verbosity: int = 1) -> str:
        """Return a formatted summary of the result.

        Parameters
        ----------
        verbosity : int
            ``0`` — single-line headline.
            ``1`` — multi-section block (default); shows objective, residuals,
            stationarity, populated diagnostics, and performance.
            ``2`` — also includes the per-iteration history table.
        """
        try:
            status_name = IPOPTStatus(self.status).name
        except ValueError:
            status_name = "UNKNOWN"

        if verbosity <= 0:
            kkt = (f", kkt={self.kkt_residual:.2e}"
                   if self.kkt_residual is not None else "")
            return (
                f"MPCCResult({self.strategy}): success={self.success}, "
                f"obj={self.obj:.6g}, comp={self.comp_residual:.2e}{kkt}"
            )

        lines: list[str] = []
        lines.append(
            f"MPCCResult — strategy={self.strategy!r}, success={self.success}"
        )
        lines.append(f"  Status: {self.status} ({status_name}) — {self.message}")

        lines.append("  Solution:")
        lines.append(f"    objective       = {self.obj: .6e}")
        lines.append(
            f"    comp_residual   = {self.comp_residual:.3e} (max), "
            f"{self.comp_residual_mean:.3e} (mean)"
        )
        if self.kkt_residual is not None:
            lines.append(f"    KKT residual    = {self.kkt_residual:.3e}")

        stat_lines: list[str] = []
        if self.stationarity and self.stationarity != "unknown":
            stat_lines.append(f"    class           = {self.stationarity}")
        if self.cq is not None:
            cq_extra = ""
            if self.cq_rank_deficit is not None:
                cq_extra = f" (rank deficit {self.cq_rank_deficit})"
            stat_lines.append(f"    CQ              = {self.cq}{cq_extra}")
        if self.b_stationary is not None:
            bs_extra = ""
            if self.b_stationary_min_descent is not None:
                bs_extra = f" (min descent {self.b_stationary_min_descent:.3e})"
            stat_lines.append(f"    B-stationary    = {self.b_stationary}{bs_extra}")
        if stat_lines:
            lines.append("  Stationarity:")
            lines.extend(stat_lines)

        if self.cq_active_set_sizes is not None:
            sizes = self.cq_active_set_sizes
            parts = [f"|I_{k}|={v}" for k, v in sizes.items()]
            lines.append("  Active set:")
            lines.append("    " + ", ".join(parts))

        perf_lines: list[str] = []
        if self.solve_time is not None:
            perf_lines.append(f"    solve_time      = {self.solve_time:.3f}s")
        if self.history:
            n_outer = len(self.history)
            n_ipopt = sum(it.n_ipopt_iter for it in self.history)
            n_restoration = sum(it.restoration_iter_count for it in self.history)
            restored = any(it.entered_restoration for it in self.history)
            iters_str = f"    outer iters     = {n_outer} (IPOPT iters: {n_ipopt}"
            if restored:
                iters_str += f", restoration: {n_restoration}"
            iters_str += ")"
            perf_lines.append(iters_str)
        if perf_lines:
            lines.append("  Performance:")
            lines.extend(perf_lines)

        if self.cleanup_status is not None:
            lines.append("  Cleanup:")
            lines.append(
                f"    accepted={self.cleanup_accepted}, "
                f"status={self.cleanup_status}, "
                f"obj={self.cleanup_obj:.6e}, "
                f"n_iter={self.cleanup_n_iter}"
            )

        if self.per_pair_status is not None:
            from collections import Counter
            counts = Counter(self.per_pair_status)
            parts = []
            for key in ("G_active", "H_active", "biactive", "inactive"):
                if counts[key]:
                    parts.append(f"{key}={counts[key]}")
            lines.append("  Complementarity pairs: " + ", ".join(parts))

        if self.tnlp_refined is not None:
            tnlp = self.tnlp_refined
            lines.append("  TNLP Refinement:")
            lines.append(
                f"    success={tnlp.success}, "
                f"status={tnlp.status}, "
                f"obj={tnlp.obj:.6e}, "
                f"stationarity={tnlp.stationarity!r}, "
                f"n_iter={tnlp.n_iter}"
            )
            if tnlp.kkt_residual is not None:
                lines.append(f"    KKT residual    = {tnlp.kkt_residual:.3e}")

        if verbosity >= 2 and self.history:
            lines.append("  Per-iteration history:")
            lines.append(
                "    k    epsilon      obj           comp_max     "
                "kkt          n_iter   time"
            )
            for k, it in enumerate(self.history):
                kkt_cell = (f"{it.kkt_residual:.3e}"
                            if it.kkt_residual is not None else "    -    ")
                lines.append(
                    f"    {k:<4d} {it.epsilon:.3e}   {it.obj: .3e}   "
                    f"{it.comp_residual:.3e}   {kkt_cell}   "
                    f"{it.n_ipopt_iter:<8d} {it.iter_time:.3f}s"
                )

        return "\n".join(lines)


def unscale_multipliers(
    result: "MPCCResult",
    mpcc_mult_G: np.ndarray,
    mpcc_mult_H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Module-level alias for :meth:`MPCCResult.unscale_comp_multipliers`."""
    return result.unscale_comp_multipliers(mpcc_mult_G, mpcc_mult_H)
