"""Safeguard option-resolution mixin for ε-continuation strategies.

Owns :meth:`_init_safeguards` (translating a merged options dict into
``self.safeguard_*`` attributes consumed by ``_run_epsilon_continuation``)
and :meth:`_validate_augmented_lagrangian_options` (a static helper used
by :class:`AugmentedLagrangianStrategy`).

The runtime safeguard *checks* (rollback, adaptive ε, KKT termination,
plateau detection) live inside :meth:`_run_epsilon_continuation` on the
continuation mixin — this mixin only handles construction-time option
parsing and storage.

Mixed into :class:`BaseStrategy`; never instantiated directly.
"""
from __future__ import annotations

from .._constants import (
    INNER_TOL_FLOOR as _INNER_TOL_FLOOR,
)
from .._constants import (
    KKT_TERMINATION_TOL as _KKT_TERMINATION_TOL,
)


class SafeguardsMixin:
    """Translate option dicts into ``self.safeguard_*`` attributes."""

    def _init_safeguards(self, opts: dict) -> None:
        """Install safeguard attributes on ``self`` from a merged options dict.

        Strategies built on :meth:`_run_epsilon_continuation` call this once
        after ``_validate_continuation_options``.  When ``opts["safeguards"] ==
        "all"``, the three boolean safeguards are forced on and the inner-tol
        mode is set to ``"quadratic"`` (one-line opt-in for callers).
        """
        all_on = (opts.get("safeguards") == "all")

        rb_on  = bool(opts.get("safeguard_rollback", False))         or all_on
        ad_on  = bool(opts.get("safeguard_adaptive_eps", False))      or all_on
        kk_on  = bool(opts.get("safeguard_kkt_termination", False))   or all_on
        pl_on  = bool(opts.get("safeguard_plateau", False))           or all_on
        mode   = opts.get("inner_tol_mode", "linear")
        if mode not in ("linear", "quadratic", "matched"):
            raise ValueError(
                "inner_tol_mode must be 'linear', 'quadratic', or 'matched',"
                f" got {mode!r}"
            )

        theta    = float(opts.get("comp_eps_ratio_theta", 10.0))
        lam_jump = float(opts.get("rollback_lambda_jump", 1e3))
        rb_cap   = int(opts.get("rollback_max_count", 3))
        eps_hold = float(opts.get("eps_hold_factor", 1.0))
        kkt_tol  = float(opts.get("kkt_tol", _KKT_TERMINATION_TOL))
        rest_th  = int(opts.get("restoration_iter_threshold", 5))
        blowup   = float(opts.get("pre_post_blowup_factor", 10.0))
        tol_fac  = float(opts.get("inner_tol_factor", 0.1))
        tol_floor = float(opts.get("inner_tol_floor", _INNER_TOL_FLOOR))
        pl_tobj  = float(opts.get("plateau_tol_obj", 1e-4))
        pl_tcomp = float(opts.get("plateau_tol_comp", 1e-3))
        pl_win   = int(opts.get("plateau_window", 2))
        pl_targ  = float(opts.get("plateau_comp_target", 1e-4))

        if theta <= 0.0:
            raise ValueError("comp_eps_ratio_theta must be > 0")
        if lam_jump <= 1.0:
            raise ValueError("rollback_lambda_jump must be > 1")
        if rb_cap < 1:
            raise ValueError("rollback_max_count must be >= 1")
        if eps_hold <= 0.0:
            raise ValueError("eps_hold_factor must be > 0")
        if kkt_tol <= 0.0:
            raise ValueError("kkt_tol must be > 0")
        if rest_th < 1:
            raise ValueError("restoration_iter_threshold must be >= 1")
        if blowup <= 1.0:
            raise ValueError("pre_post_blowup_factor must be > 1")
        if tol_fac <= 0.0:
            raise ValueError("inner_tol_factor must be > 0")
        if tol_floor <= 0.0:
            raise ValueError("inner_tol_floor must be > 0")
        if pl_tobj <= 0.0:
            raise ValueError("plateau_tol_obj must be > 0")
        if pl_tcomp <= 0.0:
            raise ValueError("plateau_tol_comp must be > 0")
        if pl_win < 1:
            raise ValueError("plateau_window must be >= 1")
        if pl_targ <= 0.0:
            raise ValueError("plateau_comp_target must be > 0")

        self.safeguard_rollback         = rb_on
        self.safeguard_adaptive_eps     = ad_on
        self.safeguard_kkt_termination  = kk_on
        self.safeguard_plateau          = pl_on
        self.kkt_tol                    = kkt_tol
        self.inner_tol_mode             = mode
        self.comp_eps_ratio_theta       = theta
        self.rollback_lambda_jump       = lam_jump
        self.rollback_max_count         = rb_cap
        self.eps_hold_factor            = eps_hold
        self.restoration_iter_threshold = rest_th
        self.pre_post_blowup_factor     = blowup
        self.inner_tol_factor           = tol_fac
        self.inner_tol_floor            = tol_floor
        self.plateau_tol_obj            = pl_tobj
        self.plateau_tol_comp           = pl_tcomp
        self.plateau_window             = pl_win
        self.plateau_comp_target        = pl_targ

    @staticmethod
    def _validate_augmented_lagrangian_options(
        *,
        rho_0: float,
        rho_max: float,
        tau: float,
        eta: float,
        max_iter: int,
        comp_tol: float,
        stagnation_iters: int,
    ) -> None:
        """Validate augmented-Lagrangian scalar options."""
        if int(max_iter) != max_iter or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1")
        if rho_0 <= 0.0:
            raise ValueError("rho_0 must be > 0")
        if rho_max < rho_0:
            raise ValueError("rho_max must be >= rho_0")
        if tau <= 1.0:
            raise ValueError("tau must be > 1")
        if eta < 0.0:
            raise ValueError("eta must be >= 0")
        if comp_tol <= 0.0:
            raise ValueError("comp_tol must be > 0")
        if int(stagnation_iters) != stagnation_iters or stagnation_iters < 1:
            raise ValueError("stagnation_iters must be an integer >= 1")


__all__ = ["SafeguardsMixin"]
