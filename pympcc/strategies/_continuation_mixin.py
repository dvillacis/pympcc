"""ε-continuation outer-loop mixin for ε-relaxation MPCC strategies.

Owns the auto-ε₀ resolution helpers, the option-validation guard, and
the main outer loop :meth:`_run_epsilon_continuation` that every
relaxation-family strategy (Scholtes, Smoothing, Lin-Fukushima, Slack,
NCP-base) calls from its :meth:`solve`.

The continuation orchestrator that runs *before* solve —
``_init_continuation_options`` — stays on :class:`BaseStrategy` itself
because it directly calls ``BaseStrategy.__init__`` (a chain that's
clearer with the parent class staying as the orchestrator).

Mixed into :class:`BaseStrategy`; never instantiated directly.  The
loop calls many sibling helpers (``_timed_solve``, ``_comp_residual``,
``_eval_comp_values``, …) that live on ``BaseStrategy`` and resolve
through normal MRO.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .._constants import (
    EPS_DIV_GUARD as _EPS_DIV_GUARD,
)
from .._constants import (
    INNER_TOL_FLOOR as _INNER_TOL_FLOOR,
)
from .._constants import (
    IPOPT_DEFAULT_TOL as _IPOPT_DEFAULT_TOL,
)
from .._constants import (
    KKT_TERMINATION_TOL as _KKT_TERMINATION_TOL,
)
from ..result import IterationInfo


class ContinuationMixin:
    """Auto-ε₀, option validation, and the ε-continuation outer loop."""

    @staticmethod
    def _resolve_auto_epsilon_0(
        problem,
        *,
        theta: float = 1.0,
        lo: float = 1e-3,
        hi: float = 1.0,
    ) -> tuple[float, float]:
        """Pick ``ε₀ = clip(theta * max|G(x₀)·H(x₀)|, lo, hi)``.

        Returns ``(epsilon_0, raw_residual)`` so callers can log both.
        Uses the raw user-provided ``comp_G`` / ``comp_H`` (pre-scaling);
        the resolver runs once at strategy ``__init__``.
        """
        G = np.asarray(problem.comp_G(problem.x0))
        H = np.asarray(problem.comp_H(problem.x0))
        raw = float(np.max(np.abs(G * H))) if G.size else 0.0
        eps0 = float(np.clip(theta * raw, lo, hi))
        return eps0, raw

    def _maybe_resolve_auto_epsilon_0(self, opts: dict) -> dict:
        """Replace ``opts['epsilon_0'] == 'auto'`` with a concrete float.

        Stashes ``self._auto_eps0_origin = (raw_residual, resolved_eps0)``
        so :meth:`_log_auto_eps0` can print one diagnostic line; left as
        ``None`` when the user passed a numeric ``epsilon_0``.
        """
        self._auto_eps0_origin: tuple[float, float] | None = None
        if opts.get("epsilon_0") == "auto":
            eps0, raw = self._resolve_auto_epsilon_0(self.problem)
            self._auto_eps0_origin = (raw, eps0)
            opts = {**opts, "epsilon_0": eps0}
        return opts

    def _log_auto_eps0(self) -> None:
        """One-line report when ε₀ was resolved from the ``"auto"`` sentinel.

        Quietly does nothing for numeric ``epsilon_0``, or when the user
        hasn't requested output (no per-iteration callback installed —
        which is the same gate ``verbose=True`` uses to wire up
        :func:`_default_verbose_callback`).
        """
        origin = getattr(self, "_auto_eps0_origin", None)
        if origin is None or self.callback is None:
            return
        raw, eps0 = origin
        clipped = " (clipped)" if eps0 != raw else ""
        print(f"auto-ε₀: max|G·H|(x0) = {raw:.2e} → ε₀ = {eps0:.2e}{clipped}")

    @staticmethod
    def _validate_continuation_options(
        *,
        epsilon_0: float,
        reduction: float,
        max_iter: int,
        epsilon_min: float,
        comp_tol: float | None,
    ) -> None:
        """Validate common continuation-strategy scalar options."""
        if int(max_iter) != max_iter or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1")
        if epsilon_0 <= 0.0:
            raise ValueError("epsilon_0 must be > 0")
        if not (0.0 < reduction < 1.0):
            raise ValueError("reduction must satisfy 0 < reduction < 1")
        if epsilon_min < 0.0:
            raise ValueError("epsilon_min must be >= 0")
        if epsilon_min >= epsilon_0:
            raise ValueError("epsilon_min must be smaller than epsilon_0")
        if comp_tol is not None and comp_tol <= 0.0:
            raise ValueError("comp_tol must be > 0 when provided")

    def _run_epsilon_continuation(
        self,
        nlp,
        x0: np.ndarray,
        eps_ref: list,
        make_iteration,
    ) -> tuple[np.ndarray, dict, float, list[IterationInfo]]:
        """
        Shared outer loop for epsilon-continuation strategies.

        Handles the common mechanics: epsilon updates, warm-starting, adaptive
        IPOPT tolerance, timing, history storage, callback invocation, and
        early stopping on ``comp_tol``.

        Optional safeguards (off by default; enabled via
        :data:`SAFEGUARD_DEFAULTS` keys forwarded as ``strategy_options``):

        * **rollback** — snapshot ``(x, warm_dual, eps)`` before each inner
          solve.  Reject and restore the snapshot when the inner solver
          fails, fails to track ε (``comp_residual > θ·ε``), or makes the
          dual multipliers explode.  After a reject ε is held (``×
          eps_hold_factor``) instead of being reduced.  After
          ``rollback_max_count`` consecutive rejects the loop terminates.
        * **adaptive ε reduction** — only apply the user's ``reduction``
          when the last solve was clean; otherwise back off to
          ``sqrt(reduction)``.
        * **inner-tol coupling** — switch the inner-solve tol from the
          linear ``ε·1e-2`` to the quadratic ``ε**1.5`` Leyffer schedule.
        * **MPCC-KKT termination** — break early once
          ``info.kkt_residual ≤ kkt_tol``.
        """
        self._log_auto_eps0()
        history: list[IterationInfo] = []
        x = np.asarray(x0, dtype=float).copy()
        eps = self.epsilon_0
        last_info: dict = {}
        # Hot-start seed (§6.5).  When :meth:`MPCCSolver.resolve` injects a
        # warm dual from a previous solve, consume it on the very first
        # inner solve and arm IPOPT's ``warm_start_init_point`` immediately
        # rather than waiting until the second outer iteration.
        warm_dual: dict = {}
        if self._initial_warm_dual:
            warm_dual = dict(self._initial_warm_dual)
            self._initial_warm_dual = None  # one-shot
        total_time: float = 0.0
        # Best feasible incumbent across the outer loop (§6.3).  Tracks the
        # accepted iterate with the smallest comp_residual seen so far; on
        # time-limit termination we restore this rather than returning the
        # in-flight (possibly partial) iterate.
        best_x: np.ndarray | None = None
        best_info: dict | None = None
        best_comp: float = float("inf")
        self._time_limit_hit = False
        wall_t0 = time.perf_counter()
        # Exposed so ``_maybe_run_cleanup`` can compute remaining budget
        # against the same outer-solve start.
        self._wall_t0 = wall_t0

        rollback_on    = getattr(self, "safeguard_rollback", False)
        adaptive_on    = getattr(self, "safeguard_adaptive_eps", False)
        kkt_term_on    = getattr(self, "safeguard_kkt_termination", False)
        plateau_on     = getattr(self, "safeguard_plateau", False)
        kkt_tol        = getattr(self, "kkt_tol", _KKT_TERMINATION_TOL)
        tol_mode       = getattr(self, "inner_tol_mode", "linear")
        theta          = getattr(self, "comp_eps_ratio_theta", 10.0)
        lam_jump_max   = getattr(self, "rollback_lambda_jump", 1e3)
        rollback_cap   = getattr(self, "rollback_max_count", 3)
        eps_hold       = getattr(self, "eps_hold_factor", 1.0)
        rest_threshold = getattr(self, "restoration_iter_threshold", 5)
        blowup_factor  = getattr(self, "pre_post_blowup_factor", 10.0)
        tol_factor     = getattr(self, "inner_tol_factor", 0.1)
        tol_floor      = getattr(self, "inner_tol_floor", _INNER_TOL_FLOOR)
        plateau_tol_obj  = getattr(self, "plateau_tol_obj", 1e-4)
        plateau_tol_comp = getattr(self, "plateau_tol_comp", 1e-3)
        plateau_window   = getattr(self, "plateau_window", 2)
        plateau_target   = getattr(self, "plateau_comp_target", 1e-4)

        user_tol      = self.ipopt_options.get("tol", _IPOPT_DEFAULT_TOL)
        inner_max_iter = int(self.ipopt_options.get("max_iter", 3000))

        rollback_run   = 0
        prev_mult_inf  = None
        last_good_eps: float | None = None  # most recent ε of an accepted iterate
        warm_init_armed_at: int | None = None  # outer index where we toggled warm_start_init_point
        # When :meth:`MPCCSolver.resolve` seeded ``warm_dual`` we want IPOPT
        # to honour it on the very first inner solve, so arm
        # ``warm_start_init_point`` before entering the loop.
        if warm_dual:
            nlp.add_option("warm_start_init_point", "yes")
            warm_init_armed_at = 0
        plateau_streak = 0
        prev_obj_acc: float | None = None
        prev_comp_acc: float | None = None

        k = 0
        while k < self.max_iter:
            if (self.time_limit is not None
                    and time.perf_counter() - wall_t0 >= self.time_limit):
                self._time_limit_hit = True
                break
            eps_ref[0] = eps
            # Zero per-solve diagnostics (n_ipopt_iter + restoration counters).
            if hasattr(nlp, "reset_iter_counters"):
                nlp.reset_iter_counters()
            else:
                nlp.n_ipopt_iter = 0
            if (warm_init_armed_at is None and k >= 1 and self.dual_warmstart):
                nlp.add_option("warm_start_init_point", "yes")
                warm_init_armed_at = k

            # Inner solver tolerance.  ``linear`` and ``quadratic`` floor
            # at ``user_tol`` (typically 1e-8) — IPOPT keeps chasing
            # precision even when ε can no longer support it, which
            # spins MAX_ITER on degenerate Scholtes NLPs.  ``matched``
            # tracks ε directly with no user_tol floor: the inner tol
            # loosens as ε shrinks, so IPOPT terminates at a precision
            # the relaxation can actually deliver.
            if tol_mode == "quadratic":
                inner_tol = max(user_tol, eps ** 1.5)
            elif tol_mode == "matched":
                inner_tol = max(tol_floor, tol_factor * eps)
            else:
                inner_tol = max(user_tol, eps * 1e-2)
            nlp.add_option("tol", inner_tol)

            # Bound the inner IPOPT solve by the remaining outer budget.
            # ``time_limit`` is wall-clock; IPOPT's ``max_cpu_time`` is CPU
            # time of the IPOPT thread, which approximates wall-clock for
            # single-threaded solves (the default).  Without this, a single
            # hard inner NLP can run minutes past the user's budget — the
            # outer-loop check only fires *between* iterations.
            if self.time_limit is not None:
                _remaining = self.time_limit - (time.perf_counter() - wall_t0)
                if _remaining <= 0:
                    self._time_limit_hit = True
                    break
                nlp.add_option("max_cpu_time", float(_remaining))

            # Snapshot for rollback (before inner solve overwrites x/warm_dual).
            if rollback_on:
                snap_x        = x.copy()
                snap_warm     = dict(warm_dual)
                snap_eps      = eps
                snap_prev_inf = prev_mult_inf
                # Pre-NLP comp residual: lets us detect post-NLP iterates
                # that are dramatically worse than the warm-start (e.g.
                # IPOPT MAX_ITER returns a degraded iterate).
                snap_pre_comp = self._comp_residual(snap_x)

            x, last_info, iter_time = self._timed_solve(nlp, x, warm_dual)
            total_time += iter_time
            if self.dual_warmstart:
                # Only seed the next inner solve when the previous one
                # actually produced multipliers; a backend or IPOPT failure
                # mode that omits ``mult_g`` would otherwise pass ``None``
                # into cyipopt's ``lagrange=`` and crash there.
                _mg = last_info.get("mult_g")
                if _mg is not None and len(_mg):
                    warm_dual = {
                        "lagrange": _mg,
                        "zl":       last_info.get("mult_x_L"),
                        "zu":       last_info.get("mult_x_U"),
                    }

            info = make_iteration(eps, x, last_info, nlp.n_ipopt_iter, iter_time)
            # Restoration-phase diagnostics — populated regardless of rollback
            # safeguards so callers can always inspect history[k].
            info.restoration_iter_count = int(getattr(nlp, "restoration_iter_count", 0))
            info.entered_restoration    = bool(getattr(nlp, "entered_restoration", False))
            history.append(info)
            if self.callback is not None:
                self.callback(len(history) - 1, history[-1])

            status_ok = last_info["status"] in (0, 1, 3)

            # ---------------- rollback decision ----------------
            rejected = False
            if rollback_on:
                tracked_eps = info.comp_residual <= theta * eps
                mult_g = last_info.get("mult_g")
                cur_inf = (float(np.max(np.abs(mult_g)))
                           if mult_g is not None and len(mult_g) else 0.0)
                mult_jump_ok = (
                    prev_mult_inf is None
                    or prev_mult_inf <= 0.0
                    or cur_inf <= lam_jump_max * prev_mult_inf
                )
                # Persistent restoration is a strong "ε too tight" signal —
                # force a rollback even when the other heuristics would
                # have accepted the iterate.
                restoration_excess = (
                    info.entered_restoration
                    and info.restoration_iter_count >= rest_threshold
                )
                # Post-NLP iterate dramatically worse than pre-NLP warm
                # start.  Catches MAX_ITER cases where the inner solver
                # returns a degraded iterate that tracked_eps alone may
                # accept (e.g. when ε is very small but pre_comp was
                # already smaller).
                post_blew_up = (
                    snap_pre_comp > 0.0
                    and info.comp_residual > blowup_factor * snap_pre_comp
                    and info.comp_residual > eps
                )
                rejected = (
                    not (status_ok and tracked_eps and mult_jump_ok)
                    or restoration_excess
                    or post_blew_up
                )

            if rejected:
                # Restore snapshot and back off ε.  ``eps_hold_factor > 1``
                # genuinely expands ε so the next attempt is less aggressive;
                # cap at ``last_good_eps`` so we never exceed the most recent
                # accepted value (otherwise a bad rollback could undo earlier
                # progress).  When no iterate has been accepted yet, fall
                # back to ``self.epsilon_0`` as the ceiling.
                x          = snap_x
                warm_dual  = snap_warm
                ceiling    = last_good_eps if last_good_eps is not None else self.epsilon_0
                eps        = min(snap_eps * eps_hold, ceiling)
                prev_mult_inf = snap_prev_inf
                plateau_streak = 0
                rollback_run += 1
                if rollback_run >= rollback_cap:
                    break
                if eps < self.epsilon_min:
                    break
                # If backoff didn't actually move ε (eps_hold==1 with no
                # ceiling change), we'd repeat the same NLP — break early.
                if eps <= snap_eps:
                    break
                k += 1
                continue

            rollback_run = 0
            last_good_eps = eps
            mult_g = last_info.get("mult_g")
            if mult_g is not None and len(mult_g):
                prev_mult_inf = float(np.max(np.abs(mult_g)))

            # Update best incumbent only on a clean (status_ok) accepted
            # iterate.  Tie-break by smaller comp_residual; ε itself is not
            # a tie-break because the accepted iterate already passed the
            # tracked_eps gate when rollback is on.
            if status_ok and info.comp_residual < best_comp:
                best_comp = info.comp_residual
                best_x = x.copy()
                best_info = dict(last_info)

            # ---------------- termination tests ----------------
            if (kkt_term_on
                    and info.kkt_residual is not None
                    and info.kkt_residual <= kkt_tol
                    and status_ok):
                break

            if (self.comp_tol is not None
                    and info.comp_residual < self.comp_tol
                    and status_ok):
                break

            # Plateau termination: stop once both obj and comp_residual
            # have stalled across consecutive accepted iterates and comp
            # is already below the target.  Catches the "obj converged
            # at outer iter k but loop keeps shrinking ε for nothing"
            # pattern that wastes most of the wall time on poorly-scaled
            # large problems.
            if plateau_on and prev_obj_acc is not None and prev_comp_acc is not None:
                d_obj = (abs(info.obj - prev_obj_acc)
                         / max(abs(info.obj), _EPS_DIV_GUARD))
                d_comp = (abs(info.comp_residual - prev_comp_acc)
                          / max(prev_comp_acc, _EPS_DIV_GUARD))
                comp_below_target = info.comp_residual <= plateau_target
                if (d_obj < plateau_tol_obj
                        and d_comp < plateau_tol_comp
                        and comp_below_target):
                    plateau_streak += 1
                    if plateau_streak >= plateau_window:
                        break
                else:
                    plateau_streak = 0
            prev_obj_acc = info.obj
            prev_comp_acc = info.comp_residual

            # ---------------- ε update ----------------
            if adaptive_on:
                clean = (
                    status_ok
                    and info.comp_residual <= theta * eps
                    and info.n_ipopt_iter < max(1, inner_max_iter // 2)
                )
                eps *= self.reduction if clean else math.sqrt(self.reduction)
            else:
                eps *= self.reduction

            if eps < self.epsilon_min:
                break
            k += 1

        # On time-limit termination, prefer the best feasible incumbent.
        # We never overwrite when the loop ran to completion — even if the
        # final iterate happens to have a slightly larger comp_residual,
        # it carries fully-converged multipliers users may rely on.
        if self._time_limit_hit and best_x is not None and best_info is not None:
            x = best_x
            last_info = best_info

        return x, last_info, total_time, history


__all__ = ["ContinuationMixin"]
