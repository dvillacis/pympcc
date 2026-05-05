"""Active-set cleanup-pass mixin for ε-continuation strategies.

Owns the optional polish phase that runs after the continuation loop
converges.  Pin every comp pair on the side IPOPT settled on, drop the
``G·H ≤ ε`` rows, and resolve the resulting smooth NLP.  Accepted only
when the polished iterate doesn't worsen the objective beyond
``cleanup_obj_worsen_tol``.

Methods owned by this mixin:

* :meth:`_init_cleanup` — translate options dict to ``self.cleanup_*``
* :meth:`_infer_active_set` — magnitude / multiplier rule pinning
* :meth:`_build_cleanup_nlp` — assemble the smooth pinned NLP
* :meth:`_build_cleanup_hessian` — universal cleanup Lagrangian Hessian
* :meth:`_build_jax_cleanup_hessian` — JAX path used by the default
* :meth:`_maybe_run_cleanup` — top-level dispatcher

Mixed into :class:`BaseStrategy`; never instantiated directly.  Methods
call sibling methods (``_eval_comp_values``, ``_build_nlp``,
``_timed_solve``, ``_decode_msg``) that live on ``BaseStrategy`` itself
and are resolved through normal MRO.
"""
from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .._constants import (
    CLEANUP_TOL_FLOOR as _CLEANUP_TOL_FLOOR,
)
from .._constants import (
    IPOPT_DEFAULT_TOL as _IPOPT_DEFAULT_TOL,
)
from ..result import MPCCResult

if TYPE_CHECKING:
    from ..problem import MPCCProblem


class CleanupMixin:
    """Active-set polish phase for ε-continuation strategies."""

    # Attributes set by ``BaseStrategy.__init__``; declared here so
    # type-checking the mixin in isolation succeeds.
    problem: "MPCCProblem"
    ipopt_options: dict
    time_limit: float | None
    _time_limit_hit: bool
    _wall_t0: float | None

    # Helpers from ``BaseStrategy`` consumed by the cleanup pass.
    _new_callback_cache: Callable[..., dict]
    _make_jac_structure: Callable[..., tuple[np.ndarray, np.ndarray]]
    _build_nlp: Callable[..., Any]
    _eval_comp_values: Callable[..., tuple[np.ndarray, np.ndarray]]
    _timed_solve: Callable[..., tuple[np.ndarray, dict, float]]
    _decode_msg: Callable[..., str]

    def _init_cleanup(self, opts: dict, user_kwargs: dict | None = None) -> None:
        """Install active-set cleanup attributes on ``self``.

        Strategies built on :meth:`_run_epsilon_continuation` call this once
        after ``_init_safeguards``.  The meta key ``opts["safeguards"] ==
        "all"`` also flips ``cleanup`` to ``"auto"`` when the user has *not*
        explicitly supplied a ``cleanup`` value, so that ``--safeguards``
        becomes the single switch to opt into both the continuation guards
        and the polish phase.  Passing ``cleanup=False`` explicitly always
        suppresses cleanup regardless of ``safeguards``.

        *user_kwargs* is the raw ``**kwargs`` dict before merging with
        defaults; it is used to detect whether the caller explicitly set
        ``cleanup``.  When omitted (legacy callers) the old behaviour is
        preserved.
        """
        cleanup_raw = opts.get("cleanup", False)
        user_set_cleanup = user_kwargs is not None and "cleanup" in user_kwargs
        if opts.get("safeguards") == "all" and not user_set_cleanup and cleanup_raw is False:
            cleanup_raw = "auto"
        if cleanup_raw not in (False, True, "auto"):
            raise ValueError(
                f"cleanup must be False, True, or 'auto', got {cleanup_raw!r}"
            )

        bi_tol = opts.get("cleanup_biactive_tol", None)
        if bi_tol is not None and float(bi_tol) <= 0.0:
            raise ValueError("cleanup_biactive_tol must be > 0 when provided")

        manual = opts.get("cleanup_active_set", None)
        if manual is not None:
            if not (isinstance(manual, tuple) and len(manual) == 2):
                raise ValueError(
                    "cleanup_active_set must be a 2-tuple "
                    "(I_G_active, I_H_active) of integer arrays"
                )

        worsen = float(opts.get("cleanup_obj_worsen_tol", 1e-3))
        max_it = int(opts.get("cleanup_max_iter", 300))
        cleanup_tol_raw = opts.get("cleanup_tol", None)
        if worsen < 0.0:
            raise ValueError("cleanup_obj_worsen_tol must be >= 0")
        if max_it < 1:
            raise ValueError("cleanup_max_iter must be >= 1")
        if cleanup_tol_raw is not None and float(cleanup_tol_raw) <= 0.0:
            raise ValueError("cleanup_tol must be > 0 when provided")

        self.cleanup                = cleanup_raw
        self.cleanup_biactive_tol   = float(bi_tol) if bi_tol is not None else None
        self.cleanup_active_set     = manual
        self.cleanup_obj_worsen_tol = worsen
        self.cleanup_max_iter       = max_it
        self.cleanup_tol            = (float(cleanup_tol_raw)
                                       if cleanup_tol_raw is not None else None)

    def _infer_active_set(
        self,
        G: np.ndarray,
        H: np.ndarray,
        mpcc_mult_G: np.ndarray | None,
        mpcc_mult_H: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partition ``range(n_comp)`` into ``(I_G_active, I_H_active)``.

        Every index is pinned on exactly one side: ``G_i = 0`` or ``H_i = 0``.
        Default rule is the magnitude rule (pin the smaller side); ties
        and biactive pairs (both within ``cleanup_biactive_tol``) are
        broken by the multiplier rule when MPCC multipliers are
        provided.  When ``self.cleanup_active_set`` is set, that
        partition is returned verbatim (no inference).
        """
        if self.cleanup_active_set is not None:
            I_G, I_H = self.cleanup_active_set
            return (
                np.asarray(I_G, dtype=np.intp).ravel(),
                np.asarray(I_H, dtype=np.intp).ravel(),
            )

        n_c = len(G)
        if n_c == 0:
            empty = np.empty(0, dtype=np.intp)
            return empty, empty

        if self.cleanup_biactive_tol is not None:
            bi_tol = self.cleanup_biactive_tol
        else:
            comp = float(np.max(np.abs(G * H)))
            bi_tol = max(np.sqrt(comp), 1e-4)

        # Magnitude rule: pin the smaller of G_i, H_i.
        pin_G = G < H

        # Multiplier rule for truly biactive pairs (both small).
        if mpcc_mult_G is not None and mpcc_mult_H is not None:
            biactive = (G <= bi_tol) & (H <= bi_tol)
            if np.any(biactive):
                mu_G_bi = np.abs(mpcc_mult_G[biactive])
                mu_H_bi = np.abs(mpcc_mult_H[biactive])
                # Pin the side with the smaller multiplier (the side the
                # solver considers less binding).
                pin_G[biactive] = mu_G_bi <= mu_H_bi

        I_G = np.where(pin_G)[0].astype(np.intp)
        I_H = np.where(~pin_G)[0].astype(np.intp)
        return I_G, I_H

    def _build_cleanup_nlp(
        self,
        I_G_active: np.ndarray,
        I_H_active: np.ndarray,
    ):
        """Build the smooth active-set-fixed cleanup NLP in original ``x``.

        Layout: ``[g, h, G, H]`` where ``G_i`` (resp. ``H_i``) row has
        ``cl=cu=0`` for pinned indices and ``cl=0, cu=+∞`` otherwise.
        Mirrors :class:`DirectStrategy` minus the ``G·H`` rows.

        Returns ``(nlp, cache)`` ready for a single ``_timed_solve`` call.
        L-BFGS Hessian is forced (cleanup uses a different Lagrangian
        than continuation, so any JAX-built Hessian from the strategy
        does not apply here).
        """
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp
        _INF = 2e19

        cl = np.concatenate([
            np.full(n_g, -_INF), np.zeros(n_h),
            np.zeros(n_c), np.zeros(n_c),
        ])
        cu = np.concatenate([
            np.zeros(n_g), np.zeros(n_h),
            np.full(n_c, _INF), np.full(n_c, _INF),
        ])
        if len(I_G_active):
            cu[n_g + n_h + I_G_active] = 0.0
        if len(I_H_active):
            cu[n_g + n_h + n_c + I_H_active] = 0.0

        cache = self._new_callback_cache()

        def constraints(x):
            std_parts = self._eval_standard_con_values(x, cache)
            G, H = self._eval_comp_values(x, cache)
            return np.concatenate([*std_parts, G, H])

        jac_structure = self._make_jac_structure([
            (n_g, p.ineq_jacobian_sparsity),
            (n_h, p.eq_jacobian_sparsity),
            (n_c, p.comp_G_jacobian_sparsity),
            (n_c, p.comp_H_jacobian_sparsity),
        ]) if p.is_sparse else None

        if p.is_sparse:
            def jacobian(x):
                std_flat = self._build_std_jac_flat(x, cache)
                vG_raw, vH_raw = self._eval_comp_jac_raw(x, cache)
                v_G = vG_raw.ravel() if vG_raw.ndim == 2 else vG_raw
                v_H = vH_raw.ravel() if vH_raw.ndim == 2 else vH_raw
                return np.concatenate([std_flat, v_G, v_H])
        else:
            def jacobian(x):
                _, jac_rows = self._build_standard_constraints(x, cache)
                _, _, JG, JH = self._build_comp_jacobians(x, cache)
                jac_rows.extend([JG, JH])
                return np.vstack(jac_rows)

        hess_fn, hess_sp = self._build_cleanup_hessian()
        nlp = self._build_nlp(cl, cu, constraints, jacobian, jac_structure,
                              hess_fn=hess_fn, hess_sparsity=hess_sp)
        if hess_fn is None:
            # No exact Hessian available for this strategy — fall back to L-BFGS
            # and floor the inherited tol at 1e-6 to avoid IPOPT spinning on
            # quasi-Newton dual residuals it can't drive below the analytical
            # tolerance.
            nlp.add_option("hessian_approximation", "limited-memory")
            cu_tol_floor = _CLEANUP_TOL_FLOOR
        else:
            cu_tol_floor = 0.0
        nlp.add_option("max_iter", int(self.cleanup_max_iter))
        if self.cleanup_tol is not None:
            cu_tol = float(self.cleanup_tol)
        else:
            cu_tol = max(float(self.ipopt_options.get("tol", _IPOPT_DEFAULT_TOL)),
                         cu_tol_floor)
        nlp.add_option("tol", cu_tol)
        return nlp, cache

    def _build_cleanup_hessian(self):
        """Return ``(hess_fn, hess_sparsity)`` for the cleanup NLP, or
        ``(None, None)`` to use L-BFGS.

        The cleanup constraint layout is ``[g, h, G, H]`` (size
        ``n_g + n_h + 2·n_c``), so ``hess_fn(x, lam_cu, obj_factor)`` must
        accept multipliers in that layout.  The cleanup Lagrangian is
        the same across every strategy::

            L_cu = obj_factor·f + λ_g·g + λ_h·h + λ_G·G + λ_H·H

        so the universal default builds it via JAX autodiff whenever
        ``problem.use_jax_hessian`` is set; otherwise returns
        ``(None, None)``.  Strategies whose full Lagrangian is a strict
        superset of ``L_cu`` (Scholtes, lin_fukushima) override to wrap
        their existing Hessian with zero-padded multipliers — cheaper
        than a second JAX compile when a manual Hessian is supplied.
        """
        if not self._has_jax_hessian():
            return None, None
        return self._build_jax_cleanup_hessian()

    def _build_jax_cleanup_hessian(self):  # pragma: no cover
        """Build the cleanup Lagrangian Hessian via JAX (universal).

        Used by every strategy whose full Lagrangian is *not* a strict
        superset of ``L_cu`` (smoothing, slack, augmented_lagrangian),
        and as the JAX fallback for Scholtes/lin_fukushima when no
        manual Hessian is supplied.
        """
        import jax.numpy as jnp

        from .._jax import jax_hessian_lagrangian
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp
        m_cu = n_g + n_h + 2 * n_c  # cleanup layout: [g, h, G, H]

        def lagrangian_cu(x, lam, obj_factor):
            val = obj_factor * p.objective(x)
            if n_g:
                val = val + jnp.dot(lam[:n_g], p.ineq_constraints(x))
            if n_h:
                val = val + jnp.dot(lam[n_g:n_g + n_h], p.eq_constraints(x))
            off = n_g + n_h
            val = val + jnp.dot(lam[off:off + n_c], p.comp_G(x))
            val = val + jnp.dot(lam[off + n_c:off + 2 * n_c], p.comp_H(x))
            return val

        return jax_hessian_lagrangian(
            lagrangian_cu, p.n, p.x0, m_cu, p.jax_sparsity_tol,
        )

    def _maybe_run_cleanup(
        self,
        result: MPCCResult,
        last_info: dict,
        x_orig: np.ndarray,
        mpcc_mult_G: np.ndarray | None,
        mpcc_mult_H: np.ndarray | None,
    ) -> MPCCResult:
        """Run the active-set cleanup pass when enabled and patch ``result``.

        Cleanup is skipped silently when:

        * ``self.cleanup`` is ``False``;
        * ``self.cleanup == "auto"`` and the continuation result is not
          usable (``status not in {0,1,3}``) or ``comp_residual > 1e-2``.

        When cleanup runs, the diagnostic fields ``cleanup_status``,
        ``cleanup_n_iter``, ``cleanup_obj``, ``cleanup_active_set`` and
        ``cleanup_accepted`` are populated regardless of whether the
        polished iterate is accepted.  The top-level ``x``, ``obj``,
        ``G``, ``H``, ``comp_residual``, ``mult_g`` and ``stationarity``
        fields are replaced *only* when cleanup converges and does not
        worsen the objective by more than ``cleanup_obj_worsen_tol``.
        """
        from .._stationarity import (  # local import: avoid cycle
            classify_stationarity,
            compute_kkt_residual,
        )

        mode = self.cleanup
        if mode is False:
            return result
        if mode == "auto":
            if not result.success or result.comp_residual > 1e-2:
                return result

        p = self.problem
        # Skip cleanup entirely when the continuation already exhausted
        # the user-supplied wall-clock budget — the cleanup NLP can be
        # expensive and we shouldn't run past the budget.
        if self._time_limit_hit:
            result.cleanup_status   = -998
            result.cleanup_accepted = False
            return result
        try:
            # Evaluate G,H in the same (possibly scaled) space the multipliers
            # live in; otherwise the magnitude/multiplier rule mixes spaces
            # when comp_G_scale/comp_H_scale are active.
            G_init, H_init = self._eval_comp_values(x_orig, self._new_callback_cache())
            I_G, I_H = self._infer_active_set(G_init, H_init,
                                              mpcc_mult_G, mpcc_mult_H)
            nlp, _cache = self._build_cleanup_nlp(I_G, I_H)
            # Bound the cleanup solve by remaining wall-clock budget so it
            # cannot run past ``self.time_limit``.
            if self.time_limit is not None and self._wall_t0 is not None:
                _remaining = self.time_limit - (time.perf_counter() - self._wall_t0)
                if _remaining <= 0:
                    self._time_limit_hit = True
                    result.cleanup_status   = -998
                    result.cleanup_accepted = False
                    return result
                nlp.add_option("max_cpu_time", float(_remaining))
            x_cu, info_cu, _t = self._timed_solve(nlp, x_orig, {})
        except (ArithmeticError, AttributeError, TypeError, ValueError,
                RuntimeError, np.linalg.LinAlgError) as exc:  # pragma: no cover
            warnings.warn(
                f"pympcc cleanup raised {type(exc).__name__}: {exc}. "
                "Returning continuation result.",
                UserWarning, stacklevel=2,
            )
            result.cleanup_status   = -999
            result.cleanup_accepted = False
            return result

        cleanup_status = int(info_cu["status"])
        cleanup_obj    = float(info_cu["obj_val"])
        result.cleanup_status      = cleanup_status
        result.cleanup_n_iter      = int(getattr(nlp, "n_ipopt_iter", 0))
        result.cleanup_obj         = cleanup_obj
        result.cleanup_active_set  = (I_G, I_H)

        cleanup_success = cleanup_status in (0, 1, 3)
        worsen = (cleanup_obj - result.obj) / (abs(result.obj) + 1.0)
        accept = cleanup_success and (worsen <= self.cleanup_obj_worsen_tol)
        result.cleanup_accepted = accept

        if not accept:
            return result

        # Promote cleanup iterate.  Use _eval_comp_values so result.G/H stay
        # in the same (scaled) space that the rest of the solver and the
        # stored multipliers use — without this, accepting cleanup silently
        # flips result.G/H/comp_residual to unscaled space.
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp
        G_cu, H_cu = self._eval_comp_values(x_cu, self._new_callback_cache())
        result.x                  = x_cu
        result.obj                = cleanup_obj
        result.G                  = G_cu
        result.H                  = H_cu
        result.comp_residual      = float(np.max(np.abs(G_cu * H_cu)))
        result.comp_residual_mean = float(np.mean(np.abs(G_cu * H_cu)))
        result.status             = cleanup_status
        result.message            = self._decode_msg(info_cu["status_msg"])
        result.success            = cleanup_success
        result.mult_g             = info_cu.get("mult_g")

        # Cleanup layout: [g, h, G, H].  The G and H multiplier blocks
        # carry both the original inequality multipliers and (where the
        # row was pinned to equality) the equality multipliers — sign
        # is unambiguous for stationarity classification because every
        # pair has exactly one pinned side.
        mult_g = info_cu.get("mult_g")
        if mult_g is not None:
            lam_G = mult_g[n_g + n_h           : n_g + n_h + n_c]
            lam_H = mult_g[n_g + n_h + n_c     : n_g + n_h + 2 * n_c]
            # Cleanup uses scaled constraints (s_G·G, s_H·H) via
            # _eval_comp_values.  Scale multipliers to original problem space
            # so compute_kkt_residual (which uses unscaled Jacobians) is correct.
            mu_G_orig = lam_G * np.asarray(p.comp_G_scale) if p.comp_G_scale is not None else lam_G
            mu_H_orig = lam_H * np.asarray(p.comp_H_scale) if p.comp_H_scale is not None else lam_H
            result.kkt_residual = compute_kkt_residual(
                result, p,
                mpcc_mult_G=mu_G_orig,
                mpcc_mult_H=mu_H_orig,
                mult_x_L=info_cu.get("mult_x_L"),
                mult_x_U=info_cu.get("mult_x_U"),
            )
            # Every pair is pinned on exactly one side, so the biactive
            # set in the cleaned MPCC is empty by construction.  In that
            # regime ``classify_stationarity`` returns ``"S-stationary"``
            # vacuously when the solve converged.
            result.stationarity = classify_stationarity(result, p)
        return result


__all__ = ["CleanupMixin"]
