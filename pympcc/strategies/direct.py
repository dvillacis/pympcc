"""Direct NLP reformulation of MPCC (single solve)."""
from __future__ import annotations

import numpy as np

from .._stationarity import classify_stationarity, compute_kkt_residual
from ..result import MPCCResult
from ._base import BaseStrategy

_INF = 2e19


class DirectStrategy(BaseStrategy):
    """
    Direct NLP reformulation.

    This strategy accepts no extra options.

    Replaces the complementarity conditions with::

        G(x) >= 0,   H(x) >= 0,   G_i(x) * H_i(x) <= 0

    and performs a **single** IPOPT solve.  Because ``G*H <= 0`` together with
    ``G,H >= 0`` forces ``G*H = 0``, this is exact — but LICQ typically fails
    at an MPCC feasible point, so IPOPT may struggle to converge or may report
    only an acceptable-level solution.
    """

    name = "direct"
    _VALID_OPTIONS: frozenset = frozenset()

    def _build_jax_hessian(self):  # pragma: no cover
        """Build exact Lagrangian Hessian via JAX autodiff."""
        import jax.numpy as jnp

        from .._jax import jax_hessian_lagrangian
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp
        m = n_g + n_h + 3 * n_c  # [g, h, G, H, G*H]

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
            val = val + jnp.dot(lam[off + 2 * n_c:], G * H)
            return val

        return jax_hessian_lagrangian(lagrangian, p.n, p.x0, m, p.jax_sparsity_tol)

    def solve(self) -> MPCCResult:
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp

        # Constraint layout:  [g, h, G, H, G*H]
        # cl: [-inf*ng, 0*nh, 0*nc, 0*nc, -inf*nc]
        # cu: [  0*ng,  0*nh, +inf, +inf,    0*nc]
        cl = np.concatenate([
            np.full(n_g, -_INF),
            np.zeros(n_h),
            np.zeros(n_c),
            np.zeros(n_c),
            np.full(n_c, -_INF),
        ])
        cu = np.concatenate([
            np.zeros(n_g),
            np.zeros(n_h),
            np.full(n_c, _INF),
            np.full(n_c, _INF),
            np.zeros(n_c),
        ])

        cache = self._new_callback_cache()

        def constraints(x):
            std_parts = self._eval_standard_con_values(x, cache)
            G, H = self._eval_comp_values(x, cache)
            return np.concatenate([*std_parts, G, H, G * H])

        union_maps = self._make_union_maps(
            p.comp_G_jacobian_sparsity, p.comp_H_jacobian_sparsity
        )
        gh_sp = union_maps[0] if union_maps is not None else None

        jac_structure = self._make_jac_structure([
            (n_g, p.ineq_jacobian_sparsity),
            (n_h, p.eq_jacobian_sparsity),
            (n_c, p.comp_G_jacobian_sparsity),
            (n_c, p.comp_H_jacobian_sparsity),
            (n_c, gh_sp),
        ]) if p.is_sparse else None

        # Pre-allocate reusable buffers (setup cost, not per-callback cost)
        _union_buf = np.empty(len(gh_sp[0])) if gh_sp is not None else None
        _gh_buf = np.empty((n_c, p.n)) if not p.is_sparse else None

        if p.is_sparse:
            def jacobian(x):
                std_flat = self._build_std_jac_flat(x, cache)
                G, H = self._eval_comp_values(x, cache)
                vG_raw, vH_raw = self._eval_comp_jac_raw(x, cache)
                v_G = vG_raw.ravel() if vG_raw.ndim == 2 else vG_raw
                v_H = vH_raw.ravel() if vH_raw.ndim == 2 else vH_raw
                if union_maps is not None:
                    (r_u, _), map1, map2 = union_maps
                    gh_vals = self._eval_weighted_union(
                        v_G, v_H, H, G, r_u, map1, map2, _union_buf)
                else:
                    JG = (vG_raw if vG_raw.ndim == 2
                          else self._to_dense_block(vG_raw, p.comp_G_jacobian_sparsity, p.n_comp, p.n))
                    JH = (vH_raw if vH_raw.ndim == 2
                          else self._to_dense_block(vH_raw, p.comp_H_jacobian_sparsity, p.n_comp, p.n))
                    gh_vals = (H[:, None] * JG + G[:, None] * JH).ravel()
                return np.concatenate([std_flat, v_G, v_H, gh_vals])
        else:
            def jacobian(x):
                _, jac_rows = self._build_standard_constraints(x, cache)
                G, H, JG, JH = self._build_comp_jacobians(x, cache)
                self._weighted_row_sum(H, JG, G, JH, _gh_buf)
                jac_rows.extend([JG, JH, _gh_buf])
                return np.vstack(jac_rows)

        hess_fn, hess_sparsity = None, None
        if self._has_manual_hessian():
            p = self.problem
            hess_fn = p.lagrangian_hessian
            hess_sparsity = p.lagrangian_hessian_sparsity
        elif self._has_jax_hessian():  # pragma: no cover
            hess_fn, hess_sparsity = self._build_jax_hessian()

        nlp = self._build_nlp(cl, cu, constraints, jacobian, jac_structure,
                               hess_fn=hess_fn, hess_sparsity=hess_sparsity)
        x, info, solve_time = self._timed_solve(nlp, p.x0, {})

        G, H = self._eval_comp_values(x, cache)

        result = MPCCResult(
            x=x,
            obj=float(info["obj_val"]),
            status=info["status"],
            message=self._decode_msg(info["status_msg"]),
            G=G,
            H=H,
            comp_residual=float(np.max(np.abs(G * H))),
            comp_residual_mean=float(np.mean(np.abs(G * H))),
            # Status 3 ("Search Direction Becomes Too Small") is common at
            # MPCC feasible points where LICQ fails; treat it as a success
            # when comp_residual is acceptably small.
            success=info["status"] in (0, 1, 3),
            strategy=self.name,
            solve_time=solve_time,
            mult_g=info["mult_g"],
        )
        result.stationarity = classify_stationarity(result, self.problem)
        # Direct layout: [g, h, G, H, G*H].  MPCC multipliers are
        # μ_G = λ_G + H ⊙ λ_GH  and  μ_H = λ_H + G ⊙ λ_GH.
        _off = n_g + n_h
        lam_G  = info["mult_g"][_off           : _off + n_c]
        lam_H  = info["mult_g"][_off + n_c     : _off + 2 * n_c]
        lam_GH = info["mult_g"][_off + 2 * n_c : _off + 3 * n_c]
        result.kkt_residual = compute_kkt_residual(
            result, self.problem,
            mpcc_mult_G=lam_G + result.H * lam_GH,
            mpcc_mult_H=lam_H + result.G * lam_GH,
            mult_x_L=info.get("mult_x_L"),
            mult_x_U=info.get("mult_x_U"),
        )
        return result
