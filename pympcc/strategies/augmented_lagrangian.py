"""Augmented Lagrangian strategy for MPCC."""
from __future__ import annotations

import numpy as np

from ._base import BaseStrategy
from ..result import IterationInfo, MPCCResult
from .._stationarity import classify_stationarity
from .._kernels import scatter_add as _scatter_add

_INF = 2e19

_DEFAULTS = dict(
    rho_0=10.0,
    rho_max=1e6,
    tau=10.0,
    eta=0.25,
    max_iter=20,
    comp_tol=1e-8,
    dual_warmstart=True,
)


class AugmentedLagrangianStrategy(BaseStrategy):
    """
    Augmented Lagrangian strategy for MPCC (PHR method).

    Moves the complementarity condition into the objective via the
    Hestenes-Powell-Rockafellar (PHR) penalty for the inequality
    ``G_i(x) * H_i(x) ≤ 0``::

        min  f(x) + (1/2ρ) Σ_i [max(0, μ_i + ρ G_i H_i)² − μ_i²]
        s.t. G(x) ≥ 0,  H(x) ≥ 0,  g(x) ≤ 0,  h(x) = 0

    The complementarity pair products appear only in the objective;
    the NLP constraints are limited to G/H non-negativity and the
    standard inequality/equality constraints.  This keeps MFCQ intact
    at every inner-NLP feasible point.

    After each inner solve the multiplier estimates are updated::

        μ_i ← max(0, μ_i + ρ G_i(x*) H_i(x*))

    and the penalty is grown by ``tau`` if the complementarity residual
    has not decreased by a factor of at least ``eta``::

        ρ ← min(tau · ρ, rho_max)

    The outer loop terminates when ``comp_residual < comp_tol`` or
    ``max_iter`` iterations have been executed.

    Parameters (passed as **strategy_options** to :class:`MPCCSolver`)
    ------------------------------------------------------------------
    rho_0 : float
        Initial penalty parameter (default 10.0).
    rho_max : float
        Upper cap on the penalty (default 1e6).
    tau : float
        Penalty growth factor applied when convergence stalls (default 10.0).
    eta : float
        Progress threshold: ``ρ`` is grown if the new complementarity
        residual exceeds ``eta`` times the previous one (default 0.25).
    max_iter : int
        Maximum number of outer iterations (default 20).
    comp_tol : float
        Outer loop terminates early when ``comp_residual < comp_tol``
        (default 1e-8).
    dual_warmstart : bool
        Warm-start IPOPT dual variables between outer iterations
        (default ``True``).

    Notes
    -----
    The ``epsilon`` field of each :class:`IterationInfo` entry stores the
    penalty parameter ``ρ`` at that iteration (not a relaxation parameter).

    References
    ----------
    Luo, Z.-Q., Pang, J.-S., & Ralph, D. (1996). *Mathematical Programs
    with Equilibrium Constraints*. Cambridge University Press.

    Huang, X.-X., Yang, X.-Q., & Zhu, D.-L. (2006). A sequential smooth
    penalization approach to mathematical programming with equilibrium
    constraints. *Numerical Functional Analysis and Optimization*, 27(1).
    """

    name = "augmented_lagrangian"

    def _build_jax_hessian(self, mu_ref: list, rho_ref: list):
        """
        Build exact Lagrangian Hessian via JAX autodiff.

        ``mu_ref`` and ``rho_ref`` are one-element lists updated each outer
        iteration.  Both are passed as JAX arrays so JAX traces them
        dynamically without recompilation.
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp
        m = n_g + n_h + 2 * n_c  # [g, h, G, H]

        def lagrangian(x, lam, obj_factor, mu, rho):
            G = p.comp_G(x)
            H = p.comp_H(x)
            lam_eff = jnp.maximum(0.0, mu + rho * G * H)
            penalty = (0.5 / rho) * jnp.sum(lam_eff ** 2 - mu ** 2)
            val = obj_factor * (p.objective(x) + penalty)
            if n_g:
                val = val + jnp.dot(lam[:n_g], p.ineq_constraints(x))
            if n_h:
                val = val + jnp.dot(lam[n_g:n_g + n_h], p.eq_constraints(x))
            off = n_g + n_h
            val = val + jnp.dot(lam[off:off + n_c], G)
            val = val + jnp.dot(lam[off + n_c:], H)
            return val

        x0_j   = jnp.asarray(p.x0, dtype=float)
        lam0_j = jnp.ones(m, dtype=float)
        mu0_j  = jnp.zeros(n_c, dtype=float)
        rho0_j = jnp.asarray(self.rho_0, dtype=float)
        H_dense = np.asarray(
            jax.hessian(
                lambda x: lagrangian(x, lam0_j, 1.0, mu0_j, rho0_j)
            )(x0_j),
            dtype=float,
        )
        mask = np.tril(np.abs(H_dense)) > p.jax_sparsity_tol
        rows, cols = np.where(mask)
        rows = rows.astype(np.intp)
        cols = cols.astype(np.intp)
        r_j, c_j = jnp.array(rows), jnp.array(cols)

        @jax.jit
        def _hess_sparse(x, lam, obj_factor, mu, rho):
            H = jax.hessian(
                lambda xk: lagrangian(xk, lam, obj_factor, mu, rho)
            )(x)
            return H[r_j, c_j]

        def hess_fn(x, lam, obj_factor):
            return np.asarray(
                _hess_sparse(
                    jnp.asarray(x, dtype=float),
                    jnp.asarray(lam, dtype=float),
                    jnp.asarray(obj_factor, dtype=float),
                    jnp.asarray(mu_ref[0], dtype=float),
                    jnp.asarray(rho_ref[0], dtype=float),
                ),
                dtype=float,
            )

        return hess_fn, (rows, cols)

    def __init__(self, problem, ipopt_options: dict, **kwargs) -> None:
        super().__init__(problem, ipopt_options, callback=kwargs.pop("callback", None))
        opts = {**_DEFAULTS, **kwargs}
        self.rho_0: float = opts["rho_0"]
        self.rho_max: float = opts["rho_max"]
        self.tau: float = opts["tau"]
        self.eta: float = opts["eta"]
        self.max_iter: int = opts["max_iter"]
        self.comp_tol: float = opts["comp_tol"]
        self.dual_warmstart: bool = bool(opts["dual_warmstart"])

    def solve(self) -> MPCCResult:
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp

        # ------------------------------------------------------------------ #
        # Constraint layout:  [g, h, G, H]
        # No complementarity constraint — it lives in the objective.
        # ------------------------------------------------------------------ #
        cl = np.concatenate([
            np.full(n_g, -_INF),
            np.zeros(n_h),
            np.zeros(n_c),        # G ≥ 0
            np.zeros(n_c),        # H ≥ 0
        ])
        cu = np.concatenate([
            np.zeros(n_g),
            np.zeros(n_h),
            np.full(n_c, _INF),
            np.full(n_c, _INF),
        ])

        def constraints(x):
            std_parts = self._eval_standard_con_values(x)
            G = np.asarray(p.comp_G(x))
            H = np.asarray(p.comp_H(x))
            return np.concatenate([*std_parts, G, H])

        jac_structure = self._make_jac_structure([
            (n_g, p.ineq_jacobian_sparsity),
            (n_h, p.eq_jacobian_sparsity),
            (n_c, p.comp_G_jacobian_sparsity),
            (n_c, p.comp_H_jacobian_sparsity),
        ]) if p.is_sparse else None

        # Union maps are needed for the sparse grad_al path.
        union_maps = self._make_union_maps(
            p.comp_G_jacobian_sparsity, p.comp_H_jacobian_sparsity
        )

        if p.is_sparse:
            def jacobian(x):
                std_flat = self._build_std_jac_flat(x)
                v_G = np.asarray(p.comp_G_jacobian(x), dtype=float)
                v_H = np.asarray(p.comp_H_jacobian(x), dtype=float)
                if v_G.ndim == 2:
                    v_G = v_G.ravel()
                if v_H.ndim == 2:
                    v_H = v_H.ravel()
                return np.concatenate([std_flat, v_G, v_H])
        else:
            def jacobian(x):
                _, jac_rows = self._build_standard_constraints(x)
                G, H, JG, JH = self._build_comp_jacobians(x)
                jac_rows.extend([JG, JH])
                return np.vstack(jac_rows)

        # ------------------------------------------------------------------ #
        # PHR augmented Lagrangian penalty in the objective
        #
        # phi(G*H, mu, rho) = (1/2rho) * [max(0, mu + rho*G*H)² - mu²]
        #
        # Gradient contribution:
        #   d(phi)/dx = max(0, mu + rho*G*H) * (H * JG + G * JH)
        #
        # Dense path:   (lam * H) @ JG + (lam * G) @ JH
        #   Two BLAS DGEMV calls — O(n_comp * n) flops, O(n) intermediates.
        #   (The old broadcasting chain created 4 × (n_comp, n) temporaries.)
        #
        # Sparse path:  _eval_weighted_union → scatter into length-n vector.
        #   Only touches union(JG, JH) nonzeros — O(nnz_union) work.
        #
        # Both objective and gradient closures read from mu_ref / rho_ref
        # (one-element lists) so they pick up the updated values each
        # outer iteration without rebinding.
        # ------------------------------------------------------------------ #
        mu_ref  = [np.zeros(n_c)]
        rho_ref = [self.rho_0]

        hess_fn, hess_sparsity = None, None
        if self._has_jax_hessian():
            hess_fn, hess_sparsity = self._build_jax_hessian(mu_ref, rho_ref)

        # Pre-allocate gradient scatter buffer for the sparse path.
        if union_maps is not None:
            (r_u_al, c_u_al), map1_al, map2_al = union_maps
            _grad_union_buf = np.empty(len(r_u_al))
        else:
            r_u_al = c_u_al = map1_al = map2_al = _grad_union_buf = None

        def obj_al(x):
            mu_k, rho_k = mu_ref[0], rho_ref[0]
            G = np.asarray(p.comp_G(x))
            H = np.asarray(p.comp_H(x))
            lam = np.maximum(0.0, mu_k + rho_k * G * H)
            penalty = (0.5 / rho_k) * np.sum(lam ** 2 - mu_k ** 2)
            return float(p.objective(x)) + penalty

        def grad_al(x):
            mu_k, rho_k = mu_ref[0], rho_ref[0]
            G = np.asarray(p.comp_G(x))
            H = np.asarray(p.comp_H(x))
            lam = np.maximum(0.0, mu_k + rho_k * G * H)   # (n_comp,)

            if union_maps is not None:
                # Sparse path: scatter (lam*H)*JG + (lam*G)*JH into grad vector.
                vG = np.asarray(p.comp_G_jacobian(x), dtype=float)
                vH = np.asarray(p.comp_H_jacobian(x), dtype=float)
                v_G = vG.ravel() if vG.ndim == 2 else vG
                v_H = vH.ravel() if vH.ndim == 2 else vH
                self._eval_weighted_union(
                    v_G, v_H, lam * H, lam * G,
                    r_u_al, map1_al, map2_al, _grad_union_buf,
                )
                d_al = np.zeros(p.n)
                _scatter_add(d_al, c_u_al, _grad_union_buf)
            else:
                # Dense path: two BLAS DGEMV calls (no large intermediates).
                _, _, JG, JH = self._build_comp_jacobians(x)
                d_al = (lam * H) @ JG + (lam * G) @ JH
            return np.asarray(p.gradient(x)) + d_al

        # ------------------------------------------------------------------ #
        # Outer loop                                                           #
        # ------------------------------------------------------------------ #
        history: list[IterationInfo] = []
        x = p.x0.copy()
        last_info: dict = {}
        warm_dual: dict = {}
        prev_comp_residual = np.inf

        for _ in range(self.max_iter):
            nlp = self._build_nlp(cl, cu, constraints, jacobian, jac_structure,
                                  obj_fn=obj_al, grad_fn=grad_al,
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
            comp_residual = float(np.max(np.abs(G * H)))
            comp_residual_mean = float(np.mean(np.abs(G * H)))

            history.append(IterationInfo(
                epsilon=rho_ref[0],          # store ρ in the epsilon slot
                x=x.copy(),
                obj=float(last_info["obj_val"]),
                status=last_info["status"],
                message=self._decode_msg(last_info["status_msg"]),
                comp_residual=comp_residual,
                comp_residual_mean=comp_residual_mean,
                n_ipopt_iter=nlp.n_ipopt_iter,
            ))
            if self.callback is not None:
                self.callback(len(history) - 1, history[-1])

            # Multiplier update: μ ← max(0, μ + ρ * G*H)
            mu_ref[0] = np.maximum(0.0, mu_ref[0] + rho_ref[0] * G * H)

            # Penalty update: grow ρ if not making enough progress
            if comp_residual > self.eta * prev_comp_residual:
                rho_ref[0] = min(rho_ref[0] * self.tau, self.rho_max)

            prev_comp_residual = comp_residual
            if comp_residual < self.comp_tol:
                break

        G = np.asarray(p.comp_G(x))
        H = np.asarray(p.comp_H(x))

        result = MPCCResult(
            x=x,
            obj=float(p.objective(x)),        # report f(x), not the penalized value
            status=last_info["status"],
            message=self._decode_msg(last_info["status_msg"]),
            G=G,
            H=H,
            comp_residual=float(np.max(np.abs(G * H))),
            comp_residual_mean=float(np.mean(np.abs(G * H))),
            success=last_info["status"] in (0, 1, 3),
            strategy=self.name,
            history=history,
            mult_g=last_info.get("mult_g"),
        )
        result.stationarity = classify_stationarity(result, self.problem)
        return result
