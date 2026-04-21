"""Lin-Fukushima regularization strategy for MPCC."""
from __future__ import annotations

import numpy as np

from ._base import BaseStrategy
from ..result import IterationInfo, MPCCResult
from .._stationarity import classify_stationarity

_INF = 2e19

_DEFAULTS = dict(epsilon_0=1.0, reduction=0.1, max_iter=20, epsilon_min=1e-8,
                 dual_warmstart=True)


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

    def _build_jax_hessian(self):
        """Build exact Lagrangian Hessian via JAX autodiff."""
        from .._jax import jax_hessian_lagrangian
        import jax.numpy as jnp
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
        super().__init__(problem, ipopt_options, callback=kwargs.pop("callback", None))
        opts = {**_DEFAULTS, **kwargs}
        self.epsilon_0: float = opts["epsilon_0"]
        self.reduction: float = opts["reduction"]
        self.max_iter: int = opts["max_iter"]
        self.epsilon_min: float = opts["epsilon_min"]
        self.dual_warmstart: bool = bool(opts["dual_warmstart"])

    def solve(self) -> MPCCResult:
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp

        # Constraint layout: [g, h, G, H, G*H, G+H]
        #
        # Both the upper bound on G*H and the lower bound on G+H depend on ε,
        # so both cl and cu must be rebuilt each outer iteration.
        #
        # Fixed parts (ε-independent):
        #   g_ineq:  −∞ ≤ g(x) ≤ 0
        #   h_eq:     0 ≤ h(x) ≤ 0
        #   G:        0 ≤ G(x) ≤ +∞
        #   H:        0 ≤ H(x) ≤ +∞
        #
        # ε-dependent parts:
        #   G*H:  −∞ ≤ G*H ≤ ε
        #   G+H:   ε ≤ G+H ≤ +∞

        def make_cl(eps: float) -> np.ndarray:
            return np.concatenate([
                np.full(n_g, -_INF),
                np.zeros(n_h),
                np.zeros(n_c),           # G >= 0
                np.zeros(n_c),           # H >= 0
                np.full(n_c, -_INF),     # G*H (no lower bound)
                np.full(n_c, eps),       # G+H >= eps
            ])

        def make_cu(eps: float) -> np.ndarray:
            return np.concatenate([
                np.zeros(n_g),
                np.zeros(n_h),
                np.full(n_c, _INF),
                np.full(n_c, _INF),
                np.full(n_c, eps),       # G*H <= eps
                np.full(n_c, _INF),
            ])

        def constraints(x):
            std_parts = self._eval_standard_con_values(x)
            G = np.asarray(p.comp_G(x))
            H = np.asarray(p.comp_H(x))
            return np.concatenate([*std_parts, G, H, G * H, G + H])

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

        # Pre-allocate reusable buffers (setup cost, not per-callback cost)
        _union_buf1 = np.empty(len(gh_sp[0])) if gh_sp is not None else None
        _union_buf2 = np.empty(len(gh_sp[0])) if gh_sp is not None else None
        _ones = np.ones(n_c)
        _gh_buf  = np.empty((n_c, p.n)) if not p.is_sparse else None
        _gph_buf = np.empty((n_c, p.n)) if not p.is_sparse else None

        if p.is_sparse:
            def jacobian(x):
                std_flat = self._build_std_jac_flat(x)
                G = np.asarray(p.comp_G(x))
                H = np.asarray(p.comp_H(x))
                vG_raw = np.asarray(p.comp_G_jacobian(x), dtype=float)
                vH_raw = np.asarray(p.comp_H_jacobian(x), dtype=float)
                v_G = vG_raw.ravel() if vG_raw.ndim == 2 else vG_raw
                v_H = vH_raw.ravel() if vH_raw.ndim == 2 else vH_raw
                if union_maps is not None:
                    (r_u, _), map1, map2 = union_maps
                    gh_vals  = self._eval_weighted_union(
                        v_G, v_H, H, G, r_u, map1, map2, _union_buf1)
                    gph_vals = self._eval_weighted_union(
                        v_G, v_H, _ones, _ones, r_u, map1, map2, _union_buf2)
                else:
                    JG = (vG_raw if vG_raw.ndim == 2
                          else self._to_dense_block(vG_raw, p.comp_G_jacobian_sparsity, p.n_comp, p.n))
                    JH = (vH_raw if vH_raw.ndim == 2
                          else self._to_dense_block(vH_raw, p.comp_H_jacobian_sparsity, p.n_comp, p.n))
                    gh_vals  = (H[:, None] * JG + G[:, None] * JH).ravel()
                    gph_vals = (JG + JH).ravel()
                return np.concatenate([std_flat, v_G, v_H, gh_vals, gph_vals])
        else:
            def jacobian(x):
                _, jac_rows = self._build_standard_constraints(x)
                G, H, JG, JH = self._build_comp_jacobians(x)
                self._weighted_row_sum(H, JG, G, JH, _gh_buf)
                np.add(JG, JH, out=_gph_buf)     # ∂(G+H)/∂x — in-place add
                jac_rows.extend([JG, JH, _gh_buf, _gph_buf])
                return np.vstack(jac_rows)

        hess_fn, hess_sparsity = None, None
        if self._has_jax_hessian():
            hess_fn, hess_sparsity = self._build_jax_hessian()

        history: list[IterationInfo] = []
        x = p.x0.copy()
        eps = self.epsilon_0
        last_info: dict = {}
        warm_dual: dict = {}

        for _ in range(self.max_iter):
            nlp = self._build_nlp(make_cl(eps), make_cu(eps), constraints, jacobian,
                                  jac_structure,
                                  hess_fn=hess_fn, hess_sparsity=hess_sparsity)
            if self.dual_warmstart and warm_dual:
                nlp.add_option("warm_start_init_point", "yes")
                x, last_info = nlp.solve(x, **warm_dual)
            else:
                x, last_info = nlp.solve(x)
            if self.dual_warmstart:
                warm_dual = {
                    "lagrange": last_info["mult_g"],
                    "zl":       last_info["mult_x_L"],
                    "zu":       last_info["mult_x_U"],
                }

            G = np.asarray(p.comp_G(x))
            H = np.asarray(p.comp_H(x))
            history.append(IterationInfo(
                epsilon=eps,
                x=x.copy(),
                obj=float(last_info["obj_val"]),
                status=last_info["status"],
                message=self._decode_msg(last_info["status_msg"]),
                comp_residual=float(np.max(np.abs(G * H))),
                comp_residual_mean=float(np.mean(np.abs(G * H))),
                n_ipopt_iter=nlp.n_ipopt_iter,
            ))
            if self.callback is not None:
                self.callback(len(history) - 1, history[-1])

            eps *= self.reduction
            if eps < self.epsilon_min:
                break

        G = np.asarray(p.comp_G(x))
        H = np.asarray(p.comp_H(x))

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
            history=history,
            mult_g=last_info.get("mult_g"),
        )
        result.stationarity = classify_stationarity(result, self.problem)
        return result
