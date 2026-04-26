"""Lin-Fukushima regularization strategy for MPCC."""
from __future__ import annotations

import numpy as np

from .._stationarity import classify_stationarity, compute_kkt_residual
from ..result import IterationInfo, MPCCResult
from ._base import CLEANUP_DEFAULTS, SAFEGUARD_DEFAULTS, BaseStrategy

_INF = 2e19

_DEFAULTS = dict(epsilon_0=1.0, reduction=0.1, max_iter=20, epsilon_min=1e-8,
                 dual_warmstart=True, comp_tol=None,
                 **SAFEGUARD_DEFAULTS, **CLEANUP_DEFAULTS)


class LinFukushimaStrategy(BaseStrategy):
    """
    Lin-Fukushima regularization strategy (Lin & Fukushima, 2003).

    Replaces the complementarity conditions with::

        G_i(x) * H_i(x) ≤ epsilon          (Scholtes upper bound)
        G_i(x) + H_i(x) ≥ epsilon          (Lin-Fukushima lower bound)

    keeping G(x) ≥ 0 and H(x) ≥ 0 as explicit constraints, and solves a
    sequence of relaxed NLPs with ``epsilon → 0``.

    The additional lower bound on ``G + H`` prevents both variables from
    simultaneously approaching zero, which is the source of MFCQ failure in
    the plain Scholtes relaxation.  As a result, MPCC-MFCQ holds at every
    feasible point of each regularized NLP (for ``epsilon > 0``), giving
    IPOPT stronger convergence guarantees on degenerate problems.

    Parameters (passed as **strategy_options** to :class:`MPCCSolver`)
    ------------------------------------------------------------------
    epsilon_0 : float
        Initial regularization parameter (default 1.0).
    reduction : float
        Multiplicative factor applied to epsilon each outer iteration
        (default 0.1).
    max_iter : int
        Maximum number of outer iterations (default 20).
    epsilon_min : float
        Outer loop terminates when ``epsilon < epsilon_min`` (default 1e-8).

    References
    ----------
    Lin, G.-H., & Fukushima, M. (2003). New relaxation method for
    mathematical programs with complementarity constraints.
    *Journal of Optimization Theory and Applications*, 118(1), 81–116.
    """

    name = "lin_fukushima"
    _VALID_OPTIONS: frozenset = frozenset(_DEFAULTS)

    def _build_jax_hessian(self):  # pragma: no cover
        """Build exact Lagrangian Hessian via JAX autodiff."""
        import jax.numpy as jnp

        from .._jax import jax_hessian_lagrangian
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp
        m = n_g + n_h + 4 * n_c  # [g, h, G, H, G*H, G+H]

        def lagrangian(x, lam, obj_factor):
            val = obj_factor * p.objective(x)
            if n_g:
                val = val + jnp.dot(lam[:n_g], p.ineq_constraints(x))
            if n_h:
                val = val + jnp.dot(lam[n_g:n_g + n_h], p.eq_constraints(x))
            G = p.comp_G(x)
            H = p.comp_H(x)
            off = n_g + n_h
            val = val + jnp.dot(lam[off:off + n_c], G)
            val = val + jnp.dot(lam[off + n_c:off + 2 * n_c], H)
            val = val + jnp.dot(lam[off + 2 * n_c:off + 3 * n_c], G * H)
            val = val + jnp.dot(lam[off + 3 * n_c:], G + H)
            return val

        return jax_hessian_lagrangian(lagrangian, p.n, p.x0, m, p.jax_sparsity_tol)

    def __init__(self, problem, ipopt_options: dict, **kwargs) -> None:
        super().__init__(problem, ipopt_options,
                         backend=kwargs.pop("backend", "ipopt"),
                         solver_options=kwargs.pop("solver_options", None),
                         callback=kwargs.pop("callback", None))
        opts = {**_DEFAULTS, **kwargs}
        opts = self._maybe_resolve_auto_epsilon_0(opts)
        self._validate_continuation_options(
            epsilon_0=opts["epsilon_0"],
            reduction=opts["reduction"],
            max_iter=opts["max_iter"],
            epsilon_min=opts["epsilon_min"],
            comp_tol=opts["comp_tol"],
        )
        self.epsilon_0: float = opts["epsilon_0"]
        self.reduction: float = opts["reduction"]
        self.max_iter: int = opts["max_iter"]
        self.epsilon_min: float = opts["epsilon_min"]
        self.dual_warmstart: bool = bool(opts["dual_warmstart"])
        self.comp_tol: float | None = opts["comp_tol"]
        self._init_safeguards(opts)
        self._init_cleanup(opts)

    def solve(self) -> MPCCResult:
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp

        # Constraint layout: [g, h, G, H, G*H - ε, G+H - ε]
        #
        # ε is absorbed into both constraint functions so bounds are static,
        # allowing the NLP object to be reused across outer iterations.
        #
        #   g_ineq:       −∞ ≤ g(x)       ≤ 0
        #   h_eq:          0 ≤ h(x)        ≤ 0
        #   G:             0 ≤ G(x)        ≤ +∞
        #   H:             0 ≤ H(x)        ≤ +∞
        #   G*H - ε:      −∞ ≤ G*H - ε    ≤ 0   (was: G*H ≤ ε)
        #   G+H - ε:       0 ≤ G+H - ε    ≤ +∞  (was: G+H ≥ ε)
        cl = np.concatenate([
            np.full(n_g, -_INF),
            np.zeros(n_h),
            np.zeros(n_c),
            np.zeros(n_c),
            np.full(n_c, -_INF),
            np.zeros(n_c),          # G+H - ε ≥ 0  (was: G+H ≥ ε)
        ])
        cu = np.concatenate([
            np.zeros(n_g),
            np.zeros(n_h),
            np.full(n_c, _INF),
            np.full(n_c, _INF),
            np.zeros(n_c),          # G*H - ε ≤ 0  (was: G*H ≤ ε)
            np.full(n_c, _INF),
        ])

        eps_ref = [self.epsilon_0]
        cache = self._new_callback_cache()

        def constraints(x):
            std_parts = self._eval_standard_con_values(x, cache)
            G, H = self._eval_comp_values(x, cache)
            e = eps_ref[0]
            return np.concatenate([*std_parts, G, H, G * H - e, G + H - e])

        union_maps = self._make_union_maps(
            p.comp_G_jacobian_sparsity, p.comp_H_jacobian_sparsity
        )
        gh_sp = union_maps[0] if union_maps is not None else None

        # G*H and G+H blocks share the same union sparsity pattern
        jac_structure = self._make_jac_structure([
            (n_g, p.ineq_jacobian_sparsity),
            (n_h, p.eq_jacobian_sparsity),
            (n_c, p.comp_G_jacobian_sparsity),
            (n_c, p.comp_H_jacobian_sparsity),
            (n_c, gh_sp),   # G*H
            (n_c, gh_sp),   # G+H
        ]) if p.is_sparse else None

        # Pre-allocate flat output buffer and alias the two tails as union buffers.
        # Layout: [std_flat | v_G | v_H | gh_vals | gph_vals]
        # Both kernel calls write directly into _jac_flat_buf via views —
        # zero copies and zero extra allocation on the return path.
        _ones = np.ones(n_c)
        _union_buf1: np.ndarray | None
        _union_buf2: np.ndarray | None
        if p.is_sparse and gh_sp is not None:
            assert jac_structure is not None  # set above when p.is_sparse
            assert p.comp_G_jacobian_sparsity is not None and p.comp_H_jacobian_sparsity is not None
            _nnz_G    = len(p.comp_G_jacobian_sparsity[0])
            _nnz_H    = len(p.comp_H_jacobian_sparsity[0])
            _nnz_gh   = len(gh_sp[0])
            _nnz_tot  = len(jac_structure[0])
            _off_G    = _nnz_tot - _nnz_G - _nnz_H - 2 * _nnz_gh
            _off_H    = _off_G + _nnz_G
            _off_gh   = _off_H + _nnz_H
            _off_gh2  = _off_gh + _nnz_gh
            _jac_flat_buf = np.empty(_nnz_tot)
            _union_buf1   = _jac_flat_buf[_off_gh:_off_gh2]   # G*H block
            _union_buf2   = _jac_flat_buf[_off_gh2:]           # G+H block
        else:
            _jac_flat_buf = None
            _union_buf1 = np.empty(len(gh_sp[0])) if gh_sp is not None else None
            _union_buf2 = np.empty(len(gh_sp[0])) if gh_sp is not None else None
        _gh_buf  = np.empty((n_c, p.n)) if not p.is_sparse else None
        _gph_buf = np.empty((n_c, p.n)) if not p.is_sparse else None

        if p.is_sparse:
            def jacobian(x):
                G, H = self._eval_comp_values(x, cache)
                vG_raw, vH_raw = self._eval_comp_jac_raw(x, cache)
                v_G = vG_raw.ravel() if vG_raw.ndim == 2 else vG_raw
                v_H = vH_raw.ravel() if vH_raw.ndim == 2 else vH_raw
                if union_maps is not None:
                    (r_u, _), map1, map2 = union_maps
                    self._eval_weighted_union(
                        v_G, v_H, H, G, r_u, map1, map2, _union_buf1)
                    self._eval_weighted_union(
                        v_G, v_H, _ones, _ones, r_u, map1, map2, _union_buf2)
                    std_flat = self._build_std_jac_flat(x, cache)
                    _jac_flat_buf[:_off_G]         = std_flat
                    _jac_flat_buf[_off_G:_off_H]   = v_G
                    _jac_flat_buf[_off_H:_off_gh]  = v_H
                    # _jac_flat_buf[_off_gh:] already written by kernels via views
                    return _jac_flat_buf
                else:
                    std_flat = self._build_std_jac_flat(x, cache)
                    JG = (vG_raw if vG_raw.ndim == 2
                          else self._to_dense_block(vG_raw, p.comp_G_jacobian_sparsity, p.n_comp, p.n))
                    JH = (vH_raw if vH_raw.ndim == 2
                          else self._to_dense_block(vH_raw, p.comp_H_jacobian_sparsity, p.n_comp, p.n))
                    gh_vals  = (H[:, None] * JG + G[:, None] * JH).ravel()
                    gph_vals = (JG + JH).ravel()
                    return np.concatenate([std_flat, v_G, v_H, gh_vals, gph_vals])
        else:
            def jacobian(x):
                _, jac_rows = self._build_standard_constraints(x, cache)
                G, H, JG, JH = self._build_comp_jacobians(x, cache)
                self._weighted_row_sum(H, JG, G, JH, _gh_buf)
                np.add(JG, JH, out=_gph_buf)     # ∂(G+H)/∂x — in-place add
                jac_rows.extend([JG, JH, _gh_buf, _gph_buf])
                return np.vstack(jac_rows)

        hess_fn, hess_sparsity = None, None
        if self._has_manual_hessian():
            p = self.problem
            hess_fn = p.lagrangian_hessian
            hess_sparsity = p.lagrangian_hessian_sparsity
        elif self._has_jax_hessian():  # pragma: no cover
            hess_fn, hess_sparsity = self._build_jax_hessian()

        # Build the NLP once — bounds are static, ε enters via eps_ref closure.
        nlp = self._build_nlp(cl, cu, constraints, jacobian, jac_structure,
                              hess_fn=hess_fn, hess_sparsity=hess_sparsity)

        def make_iteration(eps, x, last_info, n_ipopt_iter, iter_time):
            G, H = self._eval_comp_values(x, cache)
            _off = n_g + n_h
            _lam_G   = last_info["mult_g"][_off             : _off + n_c]
            _lam_H   = last_info["mult_g"][_off + n_c       : _off + 2 * n_c]
            _lam_GH  = last_info["mult_g"][_off + 2 * n_c   : _off + 3 * n_c]
            _lam_GPH = last_info["mult_g"][_off + 3 * n_c   : _off + 4 * n_c]
            _kkt = self._compute_kkt_iter(
                x, last_info["mult_g"],
                mpcc_mult_G=_lam_G + H * _lam_GH + _lam_GPH,
                mpcc_mult_H=_lam_H + G * _lam_GH + _lam_GPH,
                mult_x_L=last_info.get("mult_x_L"),
                mult_x_U=last_info.get("mult_x_U"),
            )
            return IterationInfo(
                epsilon=eps,
                x=x.copy(),
                obj=float(last_info["obj_val"]),
                status=last_info["status"],
                message=self._decode_msg(last_info["status_msg"]),
                comp_residual=float(np.max(np.abs(G * H))),
                comp_residual_mean=float(np.mean(np.abs(G * H))),
                n_ipopt_iter=n_ipopt_iter,
                iter_time=iter_time,
                kkt_residual=_kkt,
            )

        x, last_info, total_time, history = self._run_epsilon_continuation(
            nlp, p.x0, eps_ref, make_iteration
        )

        G, H = self._eval_comp_values(x, cache)

        result = MPCCResult(
            x=x,
            obj=float(last_info["obj_val"]),
            status=last_info["status"],
            message=self._decode_msg(last_info["status_msg"]),
            G=G,
            H=H,
            comp_residual=float(np.max(np.abs(G * H))),
            comp_residual_mean=float(np.mean(np.abs(G * H))),
            # Status 3 ("Search Direction Becomes Too Small") is expected at
            # MPCC solutions where LICQ fails; the solution is still valid.
            success=last_info["status"] in (0, 1, 3),
            strategy=self.name,
            solve_time=total_time,
            history=history,
            mult_g=last_info.get("mult_g"),
        )
        result.stationarity = classify_stationarity(result, self.problem)
        # Lin-Fukushima layout: [g, h, G, H, G*H-ε, G+H-ε].  MPCC multipliers:
        # μ_G = λ_G + H ⊙ λ_GH + λ_GPH,  μ_H = λ_H + G ⊙ λ_GH + λ_GPH.
        _off = n_g + n_h
        lam_G   = last_info["mult_g"][_off             : _off + n_c]
        lam_H   = last_info["mult_g"][_off + n_c       : _off + 2 * n_c]
        lam_GH  = last_info["mult_g"][_off + 2 * n_c   : _off + 3 * n_c]
        lam_GPH = last_info["mult_g"][_off + 3 * n_c   : _off + 4 * n_c]
        mpcc_mult_G = lam_G + result.H * lam_GH + lam_GPH
        mpcc_mult_H = lam_H + result.G * lam_GH + lam_GPH
        result.kkt_residual = compute_kkt_residual(
            result, self.problem,
            mpcc_mult_G=mpcc_mult_G,
            mpcc_mult_H=mpcc_mult_H,
            mult_x_L=last_info.get("mult_x_L"),
            mult_x_U=last_info.get("mult_x_U"),
        )
        result = self._maybe_run_cleanup(result, last_info, x, mpcc_mult_G, mpcc_mult_H)
        return result
