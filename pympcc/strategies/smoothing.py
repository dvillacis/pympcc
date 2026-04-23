"""Fischer-Burmeister smoothing strategy for MPCC."""
from __future__ import annotations

import numpy as np

from .._kernels import eval_phi_eps_weighted_union as _phi_eps_wu
from .._stationarity import classify_stationarity, compute_kkt_residual
from ..result import IterationInfo, MPCCResult
from ._base import BaseStrategy

_INF = 2e19

_DEFAULTS = dict(epsilon_0=1.0, reduction=0.1, max_iter=20, epsilon_min=1e-8,
                 dual_warmstart=True, comp_tol=None)


class SmoothingStrategy(BaseStrategy):
    """
    Fischer-Burmeister smoothing strategy.

    Replaces the complementarity conditions with the smoothed
    Fischer-Burmeister (FB) equation::

        phi_eps(G_i, H_i) = G_i + H_i - sqrt(G_i^2 + H_i^2 + eps^2) = 0

    keeping G(x) >= 0 and H(x) >= 0 as explicit inequality constraints.
    Solves a sequence of smooth NLPs with ``eps -> 0``.
    Each NLP is warm-started from the previous solution.

    As ``eps -> 0``, ``phi_0(a, b) = 0`` is equivalent to
    ``a >= 0, b >= 0, a*b = 0`` (the exact Fischer-Burmeister reformulation).

    Parameters (passed as **strategy_options** to :class:`MPCCSolver`)
    ------------------------------------------------------------------
    epsilon_0 : float
        Initial smoothing parameter (default 1.0).
    reduction : float
        Multiplicative factor applied to eps each outer iteration
        (default 0.1).
    max_iter : int
        Maximum number of outer iterations (default 20).
    epsilon_min : float
        Outer loop terminates when ``eps < epsilon_min`` (default 1e-8).

    References
    ----------
    Chen, B., & Harker, P. T. (1993). A non-interior-point continuation
    method for linear complementarity problems.
    *SIAM Journal on Matrix Analysis and Applications*, 14(4), 1168–1190.
    """

    name = "smoothing"
    _VALID_OPTIONS: frozenset = frozenset(_DEFAULTS)

    def _build_jax_hessian(self, eps_ref: list):  # pragma: no cover
        """
        Build exact Lagrangian Hessian via JAX autodiff.

        ``eps_ref`` is a one-element list so the compiled callback always reads
        the current smoothing parameter at call time without JAX recompilation.
        ``eps`` is passed as a JAX array so JAX traces it dynamically.
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp
        m = n_g + n_h + 3 * n_c  # [g, h, G, H, phi_eps]

        def lagrangian(x, lam, obj_factor, eps):
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
            phi = G + H - jnp.sqrt(G ** 2 + H ** 2 + eps ** 2)
            val = val + jnp.dot(lam[off + 2 * n_c:], phi)
            return val

        x0_j   = jnp.asarray(p.x0, dtype=float)
        lam0_j = jnp.ones(m, dtype=float)
        H_dense = np.asarray(
            jax.hessian(
                lambda x: lagrangian(x, lam0_j, 1.0, jnp.asarray(1.0))
            )(x0_j),
            dtype=float,
        )
        mask = np.tril(np.abs(H_dense)) > p.jax_sparsity_tol
        rows, cols = np.where(mask)
        rows = rows.astype(np.intp)
        cols = cols.astype(np.intp)
        r_j, c_j = jnp.array(rows), jnp.array(cols)

        @jax.jit
        def _hess_sparse(x, lam, obj_factor, eps):
            H = jax.hessian(
                lambda xk: lagrangian(xk, lam, obj_factor, eps)
            )(x)
            return H[r_j, c_j]

        def hess_fn(x, lam, obj_factor):
            return np.asarray(
                _hess_sparse(
                    jnp.asarray(x, dtype=float),
                    jnp.asarray(lam, dtype=float),
                    jnp.asarray(obj_factor, dtype=float),
                    jnp.asarray(eps_ref[0], dtype=float),
                ),
                dtype=float,
            )

        return hess_fn, (rows, cols)

    def __init__(self, problem, ipopt_options: dict, **kwargs) -> None:
        super().__init__(problem, ipopt_options,
                         backend=kwargs.pop("backend", "ipopt"),
                         solver_options=kwargs.pop("solver_options", None),
                         callback=kwargs.pop("callback", None))
        opts = {**_DEFAULTS, **kwargs}
        self.epsilon_0: float = opts["epsilon_0"]
        self.reduction: float = opts["reduction"]
        self.max_iter: int = opts["max_iter"]
        self.epsilon_min: float = opts["epsilon_min"]
        self.dual_warmstart: bool = bool(opts["dual_warmstart"])
        self.comp_tol: float | None = opts["comp_tol"]

    # ------------------------------------------------------------------ #
    # Fischer-Burmeister functions                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _phi(G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
        """phi_eps(G, H) — shape (n_comp,)."""
        r = np.sqrt(G**2 + H**2 + eps**2)
        return G + H - r

    @staticmethod
    def _phi_jac(
        G: np.ndarray, H: np.ndarray,
        JG: np.ndarray, JH: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Jacobian of phi_eps w.r.t. x — shape (n_comp, n)."""
        r = np.sqrt(G**2 + H**2 + eps**2)   # (n_comp,)
        alpha = 1.0 - G / r                  # (n_comp,)
        beta  = 1.0 - H / r                  # (n_comp,)
        return alpha[:, None] * JG + beta[:, None] * JH

    # ------------------------------------------------------------------ #

    def solve(self) -> MPCCResult:
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp

        # Constraint layout: [g, h, G, H, phi_eps(G,H)]
        # Bounds are fixed across iterations; only the constraint function
        # changes (phi depends on eps via the closure).
        cl = np.concatenate([
            np.full(n_g, -_INF),
            np.zeros(n_h),
            np.zeros(n_c),        # G >= 0
            np.zeros(n_c),        # H >= 0
            np.zeros(n_c),        # phi_eps = 0
        ])
        cu = np.concatenate([
            np.zeros(n_g),
            np.zeros(n_h),
            np.full(n_c, _INF),
            np.full(n_c, _INF),
            np.zeros(n_c),        # phi_eps = 0  (equality)
        ])

        # eps is mutated each iteration via a one-element list (closure trick).
        # Must be created before _build_jax_hessian so the hess_fn closure
        # captures the same list that solve() updates.
        eps_ref = [self.epsilon_0]

        hess_fn, hess_sparsity = None, None
        if self._has_jax_hessian():  # pragma: no cover
            hess_fn, hess_sparsity = self._build_jax_hessian(eps_ref)

        def constraints(x):
            std_parts = self._eval_standard_con_values(x)
            G = np.asarray(p.comp_G(x))
            H = np.asarray(p.comp_H(x))
            phi = self._phi(G, H, eps_ref[0])
            return np.concatenate([*std_parts, G, H, phi])

        union_maps = self._make_union_maps(
            p.comp_G_jacobian_sparsity, p.comp_H_jacobian_sparsity
        )
        gh_sp = union_maps[0] if union_maps is not None else None

        # phi_eps support = union(supp(JG), supp(JH))
        jac_structure = self._make_jac_structure([
            (n_g, p.ineq_jacobian_sparsity),
            (n_h, p.eq_jacobian_sparsity),
            (n_c, p.comp_G_jacobian_sparsity),
            (n_c, p.comp_H_jacobian_sparsity),
            (n_c, gh_sp),
        ]) if p.is_sparse else None

        # Pre-allocate flat output buffer and alias the tail as the union buffer.
        # When the sparse hot path is active (p.is_sparse and union_maps is not
        # None), the kernel writes phi_vals directly into _jac_flat_buf via the
        # view — zero copies on the return path.
        _union_buf: np.ndarray | None
        if p.is_sparse and gh_sp is not None:
            assert jac_structure is not None  # set above when p.is_sparse
            assert p.comp_G_jacobian_sparsity is not None and p.comp_H_jacobian_sparsity is not None
            _nnz_G   = len(p.comp_G_jacobian_sparsity[0])
            _nnz_H   = len(p.comp_H_jacobian_sparsity[0])
            _nnz_gh  = len(gh_sp[0])
            _nnz_tot = len(jac_structure[0])
            _off_G   = _nnz_tot - _nnz_G - _nnz_H - _nnz_gh
            _off_H   = _off_G + _nnz_G
            _off_gh  = _off_H + _nnz_H
            _jac_flat_buf = np.empty(_nnz_tot)
            _union_buf    = _jac_flat_buf[_off_gh:]   # view — kernel writes here
        else:
            _jac_flat_buf = None
            _union_buf    = np.empty(len(gh_sp[0])) if gh_sp is not None else None
        _gh_buf = np.empty((n_c, p.n)) if not p.is_sparse else None

        if p.is_sparse:
            def jacobian(x):
                G = np.asarray(p.comp_G(x))
                H = np.asarray(p.comp_H(x))
                vG_raw = np.asarray(p.comp_G_jacobian(x), dtype=float)
                vH_raw = np.asarray(p.comp_H_jacobian(x), dtype=float)
                v_G = vG_raw.ravel() if vG_raw.ndim == 2 else vG_raw
                v_H = vH_raw.ravel() if vH_raw.ndim == 2 else vH_raw
                if union_maps is not None:
                    (r_u, _), map1, map2 = union_maps
                    # Fused kernel: zero (n_comp,) temporaries.
                    _phi_eps_wu(v_G, v_H, G, H, eps_ref[0], r_u, map1, map2, _union_buf)
                    # Assemble output directly into the pre-allocated flat buffer.
                    std_flat = self._build_std_jac_flat(x)
                    _jac_flat_buf[:_off_G]        = std_flat
                    _jac_flat_buf[_off_G:_off_H]  = v_G
                    _jac_flat_buf[_off_H:_off_gh] = v_H
                    # _jac_flat_buf[_off_gh:] already written by kernel via view
                    return _jac_flat_buf
                else:
                    std_flat = self._build_std_jac_flat(x)
                    r_norm = np.sqrt(G ** 2 + H ** 2 + eps_ref[0] ** 2)
                    alpha  = 1.0 - G / r_norm
                    beta   = 1.0 - H / r_norm
                    JG = (vG_raw if vG_raw.ndim == 2
                          else self._to_dense_block(vG_raw, p.comp_G_jacobian_sparsity, p.n_comp, p.n))
                    JH = (vH_raw if vH_raw.ndim == 2
                          else self._to_dense_block(vH_raw, p.comp_H_jacobian_sparsity, p.n_comp, p.n))
                    phi_vals = (alpha[:, None] * JG + beta[:, None] * JH).ravel()
                    return np.concatenate([std_flat, v_G, v_H, phi_vals])
        else:
            def jacobian(x):
                _, jac_rows = self._build_standard_constraints(x)
                G, H, JG, JH = self._build_comp_jacobians(x)
                r_norm = np.sqrt(G ** 2 + H ** 2 + eps_ref[0] ** 2)
                alpha = 1.0 - G / r_norm
                beta  = 1.0 - H / r_norm
                self._weighted_row_sum(alpha, JG, beta, JH, _gh_buf)
                jac_rows.extend([JG, JH, _gh_buf])
                return np.vstack(jac_rows)

        # Build the NLP once — bounds are static, ε enters via eps_ref closure.
        nlp = self._build_nlp(cl, cu, constraints, jacobian, jac_structure,
                              hess_fn=hess_fn, hess_sparsity=hess_sparsity)

        history: list[IterationInfo] = []
        x = p.x0.copy()
        eps = self.epsilon_0
        last_info: dict = {}
        warm_dual: dict = {}
        total_time: float = 0.0

        for k in range(self.max_iter):
            eps_ref[0] = eps
            nlp.n_ipopt_iter = 0
            if k == 1 and self.dual_warmstart:
                nlp.add_option("warm_start_init_point", "yes")
            # Loose tolerance early (relaxed NLP is intermediate, not the final answer).
            # Floor is the user's requested tol so we never over-solve final iterations.
            nlp.add_option("tol", max(self.ipopt_options.get("tol", 1e-8), eps * 1e-2))
            x, last_info, iter_time = self._timed_solve(nlp, x, warm_dual)
            total_time += iter_time
            if self.dual_warmstart:
                warm_dual = {
                    "lagrange": last_info["mult_g"],
                    "zl":       last_info["mult_x_L"],
                    "zu":       last_info["mult_x_U"],
                }

            G = np.asarray(p.comp_G(x))
            H = np.asarray(p.comp_H(x))
            _off = n_g + n_h
            _lam_G   = last_info["mult_g"][_off           : _off + n_c]
            _lam_H   = last_info["mult_g"][_off + n_c     : _off + 2 * n_c]
            _lam_phi = last_info["mult_g"][_off + 2 * n_c : _off + 3 * n_c]
            _r_norm  = np.sqrt(G**2 + H**2 + eps**2)
            _kkt = self._compute_kkt_iter(
                x, last_info["mult_g"],
                mpcc_mult_G=_lam_G + (1.0 - G / _r_norm) * _lam_phi,
                mpcc_mult_H=_lam_H + (1.0 - H / _r_norm) * _lam_phi,
                mult_x_L=last_info.get("mult_x_L"),
                mult_x_U=last_info.get("mult_x_U"),
            )
            history.append(IterationInfo(
                epsilon=eps,
                x=x.copy(),
                obj=float(last_info["obj_val"]),
                status=last_info["status"],
                message=self._decode_msg(last_info["status_msg"]),
                comp_residual=float(np.max(np.abs(G * H))),
                comp_residual_mean=float(np.mean(np.abs(G * H))),
                n_ipopt_iter=nlp.n_ipopt_iter,
                iter_time=iter_time,
                kkt_residual=_kkt,
            ))
            if self.callback is not None:
                self.callback(len(history) - 1, history[-1])

            if (self.comp_tol is not None
                    and history[-1].comp_residual < self.comp_tol
                    and last_info["status"] in (0, 1, 3)):
                break
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
            solve_time=total_time,
            history=history,
            mult_g=last_info.get("mult_g"),
        )
        result.stationarity = classify_stationarity(result, self.problem)
        # Smoothing layout: [g, h, G, H, φ_ε=0].  MPCC multipliers:
        # μ_G = λ_G + α ⊙ λ_φ,  μ_H = λ_H + β ⊙ λ_φ
        # where α = 1 − G/r, β = 1 − H/r, r = sqrt(G²+H²+ε²).
        _off = n_g + n_h
        lam_G   = last_info["mult_g"][_off           : _off + n_c]
        lam_H   = last_info["mult_g"][_off + n_c     : _off + 2 * n_c]
        lam_phi = last_info["mult_g"][_off + 2 * n_c : _off + 3 * n_c]
        _r_norm = np.sqrt(result.G ** 2 + result.H ** 2 + eps_ref[0] ** 2)
        _alpha  = 1.0 - result.G / _r_norm
        _beta   = 1.0 - result.H / _r_norm
        result.kkt_residual = compute_kkt_residual(
            result, self.problem,
            mpcc_mult_G=lam_G + _alpha * lam_phi,
            mpcc_mult_H=lam_H + _beta  * lam_phi,
            mult_x_L=last_info.get("mult_x_L"),
            mult_x_U=last_info.get("mult_x_U"),
        )
        return result
