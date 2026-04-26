"""Augmented Lagrangian strategy for MPCC."""
from __future__ import annotations

import numpy as np

from .._kernels import scatter_add as _scatter_add
from .._stationarity import classify_stationarity, compute_kkt_residual
from ..result import IterationInfo, MPCCResult
from ._base import CLEANUP_DEFAULTS, BaseStrategy

_INF = 2e19

_DEFAULTS = dict(
    rho_0=10.0,
    rho_max=1e6,
    tau=10.0,
    eta=0.25,
    max_iter=20,
    comp_tol=1e-8,
    dual_warmstart=True,
    stagnation_iters=5,
    **CLEANUP_DEFAULTS,
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
    stagnation_iters : int
        Maximum number of consecutive outer iterations allowed once ``ρ``
        has reached ``rho_max`` without the complementarity residual
        improving (i.e. ``comp_residual > eta * prev_comp_residual``).
        Prevents wasting iterations when the penalty cap prevents further
        progress (default 5).

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
    _VALID_OPTIONS: frozenset = frozenset(_DEFAULTS)

    def _build_jax_hessian(self, mu_ref: list, rho_ref: list):  # pragma: no cover
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
        super().__init__(problem, ipopt_options,
                         backend=kwargs.pop("backend", "ipopt"),
                         solver_options=kwargs.pop("solver_options", None),
                         callback=kwargs.pop("callback", None))
        opts = {**_DEFAULTS, **kwargs}
        self._validate_augmented_lagrangian_options(
            rho_0=opts["rho_0"],
            rho_max=opts["rho_max"],
            tau=opts["tau"],
            eta=opts["eta"],
            max_iter=opts["max_iter"],
            comp_tol=opts["comp_tol"],
            stagnation_iters=opts["stagnation_iters"],
        )
        self.rho_0: float = opts["rho_0"]
        self.rho_max: float = opts["rho_max"]
        self.tau: float = opts["tau"]
        self.eta: float = opts["eta"]
        self.max_iter: int = opts["max_iter"]
        self.comp_tol: float = opts["comp_tol"]
        self.dual_warmstart: bool = bool(opts["dual_warmstart"])
        self.stagnation_iters: int = int(opts["stagnation_iters"])
        self._init_cleanup(opts)

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

        # Union maps are needed for the sparse grad_al path.
        union_maps = self._make_union_maps(
            p.comp_G_jacobian_sparsity, p.comp_H_jacobian_sparsity
        )

        if p.is_sparse:
            def jacobian(x):
                std_flat = self._build_std_jac_flat(x, cache)
                v_G, v_H = self._eval_comp_jac_raw(x, cache)
                if v_G.ndim == 2:
                    v_G = v_G.ravel()
                if v_H.ndim == 2:
                    v_H = v_H.ravel()
                return np.concatenate([std_flat, v_G, v_H])
        else:
            def jacobian(x):
                _, jac_rows = self._build_standard_constraints(x, cache)
                G, H, JG, JH = self._build_comp_jacobians(x, cache)
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
        if self._has_jax_hessian():  # pragma: no cover
            hess_fn, hess_sparsity = self._build_jax_hessian(mu_ref, rho_ref)

        # Pre-allocate gradient scatter buffer for the sparse path.
        if union_maps is not None:
            (r_u_al, c_u_al), map1_al, map2_al = union_maps
            _grad_union_buf   = np.empty(len(r_u_al))
            _grad_scatter_buf = np.zeros(p.n)
        else:
            r_u_al = c_u_al = map1_al = map2_al = _grad_union_buf = None
            _grad_scatter_buf = None

        def obj_al(x):
            mu_k, rho_k = mu_ref[0], rho_ref[0]
            G, H = self._eval_comp_values(x, cache)
            lam = np.maximum(0.0, mu_k + rho_k * G * H)
            penalty = (0.5 / rho_k) * np.sum(lam ** 2 - mu_k ** 2)
            return float(p.objective(x)) + penalty

        def grad_al(x):
            mu_k, rho_k = mu_ref[0], rho_ref[0]
            G, H = self._eval_comp_values(x, cache)
            lam = np.maximum(0.0, mu_k + rho_k * G * H)   # (n_comp,)

            if union_maps is not None:
                # Sparse path: scatter (lam*H)*JG + (lam*G)*JH into grad vector.
                vG, vH = self._eval_comp_jac_raw(x, cache)
                v_G = vG.ravel() if vG.ndim == 2 else vG
                v_H = vH.ravel() if vH.ndim == 2 else vH
                self._eval_weighted_union(
                    v_G, v_H, lam * H, lam * G,
                    r_u_al, map1_al, map2_al, _grad_union_buf,
                )
                _grad_scatter_buf[:] = 0.0
                _scatter_add(_grad_scatter_buf, c_u_al, _grad_union_buf)
                d_al = _grad_scatter_buf
            else:
                # Dense path: two BLAS DGEMV calls (no large intermediates).
                _, _, JG, JH = self._build_comp_jacobians(x, cache)
                d_al = (lam * H) @ JG + (lam * G) @ JH
            return np.asarray(p.gradient(x)) + d_al

        # ------------------------------------------------------------------ #
        # Outer loop                                                           #
        # ------------------------------------------------------------------ #
        # Build the NLP once — obj_al/grad_al read mu_ref/rho_ref dynamically,
        # and cl/cu are static (complementarity lives in the objective, not
        # constraints).
        nlp = self._build_nlp(cl, cu, constraints, jacobian, jac_structure,
                              obj_fn=obj_al, grad_fn=grad_al,
                              hess_fn=hess_fn, hess_sparsity=hess_sparsity)

        history: list[IterationInfo] = []
        x = p.x0.copy()
        last_info: dict = {}
        warm_dual: dict = {}
        prev_comp_residual = np.inf
        stagnation_count = 0
        total_time: float = 0.0

        # Seed the adaptive inner tolerance from the complementarity residual
        # at x0.  If x0 is already complementary, fall back to 1.0 so the
        # first inner solve is still solved loosely.
        _G0, _H0 = self._eval_comp_values(x, cache)
        _init_comp = max(float(np.max(np.abs(_G0 * _H0))), 1.0)

        for k in range(self.max_iter):
            nlp.n_ipopt_iter = 0
            if k == 1 and self.dual_warmstart:
                nlp.add_option("warm_start_init_point", "yes")
            # Adaptive inner tolerance: mirror the Scholtes pattern — no need
            # to solve tighter than the current complementarity gap warrants.
            # On the first iteration prev_comp_residual is inf, so fall back
            # to the x0 residual; from iteration 1 onwards use the previous
            # outer residual.  The penalty update logic is unaffected because
            # it still reads the unmodified prev_comp_residual (inf on k=0).
            #
            # The upper cap of 1e-6 ensures the final iterate (which may be
            # the very first solve when AL converges in one step) is solved
            # accurately enough for the KKT residual to be meaningful.
            _comp_for_tol = (
                _init_comp if np.isinf(prev_comp_residual) else prev_comp_residual
            )
            _tol_user = self.ipopt_options.get("tol", 1e-8)
            nlp.add_option("tol", max(_tol_user, min(_comp_for_tol * 1e-2, 1e-6)))
            x, last_info, iter_time = self._timed_solve(nlp, x, warm_dual)
            total_time += iter_time
            if self.dual_warmstart:
                warm_dual = {
                    "lagrange": last_info["mult_g"],
                    "zl":       last_info["mult_x_L"],
                    "zu":       last_info["mult_x_U"],
                }

            G, H = self._eval_comp_values(x, cache)
            comp_residual = float(np.max(np.abs(G * H)))
            comp_residual_mean = float(np.mean(np.abs(G * H)))

            _off = n_g + n_h
            _lam_G = last_info["mult_g"][_off       : _off + n_c]
            _lam_H = last_info["mult_g"][_off + n_c : _off + 2 * n_c]
            _mu_eff = np.maximum(0.0, mu_ref[0] + rho_ref[0] * G * H)
            _kkt = self._compute_kkt_iter(
                x, last_info["mult_g"],
                mpcc_mult_G=_lam_G + _mu_eff * H,
                mpcc_mult_H=_lam_H + _mu_eff * G,
                mult_x_L=last_info.get("mult_x_L"),
                mult_x_U=last_info.get("mult_x_U"),
            )
            history.append(IterationInfo(
                epsilon=rho_ref[0],          # store ρ in the epsilon slot
                x=x.copy(),
                obj=float(last_info["obj_val"]),
                status=last_info["status"],
                message=self._decode_msg(last_info["status_msg"]),
                comp_residual=comp_residual,
                comp_residual_mean=comp_residual_mean,
                n_ipopt_iter=nlp.n_ipopt_iter,
                iter_time=iter_time,
                kkt_residual=_kkt,
            ))
            if self.callback is not None:
                self.callback(len(history) - 1, history[-1])

            # Multiplier update: μ ← max(0, μ + ρ * G*H)
            mu_ref[0] = _mu_eff

            # Penalty update: grow ρ if not making enough progress.
            _no_progress = comp_residual > self.eta * prev_comp_residual
            if _no_progress:
                rho_ref[0] = min(rho_ref[0] * self.tau, self.rho_max)

            # Stagnation detection: once ρ is capped at rho_max and the
            # complementarity residual is not improving, further outer
            # iterations cannot help — terminate early.
            if rho_ref[0] >= self.rho_max and _no_progress:
                stagnation_count += 1
                if stagnation_count >= self.stagnation_iters:
                    break
            else:
                stagnation_count = 0

            prev_comp_residual = comp_residual
            if comp_residual < self.comp_tol:
                break

        G, H = self._eval_comp_values(x, cache)

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
            solve_time=total_time,
            history=history,
            mult_g=last_info.get("mult_g"),
        )
        result.stationarity = classify_stationarity(result, self.problem)
        # AL layout: [g, h, G, H] (no G*H constraint — it lives in the
        # objective via the PHR penalty).  The penalty gradient contributes
        # μ_G = λ_G + μ_AL ⊙ H  and  μ_H = λ_H + μ_AL ⊙ G.
        # mu_ref[0] holds the updated penalty multiplier from the last outer
        # iteration which equals (ρ*G*H + μ_prev) → μ_prev·H contribution.
        _off = n_g + n_h
        lam_G = last_info["mult_g"][_off       : _off + n_c]
        lam_H = last_info["mult_g"][_off + n_c : _off + 2 * n_c]
        mpcc_mult_G = lam_G + mu_ref[0] * result.H
        mpcc_mult_H = lam_H + mu_ref[0] * result.G
        result.kkt_residual = compute_kkt_residual(
            result, self.problem,
            mpcc_mult_G=mpcc_mult_G,
            mpcc_mult_H=mpcc_mult_H,
            mult_x_L=last_info.get("mult_x_L"),
            mult_x_U=last_info.get("mult_x_U"),
        )
        result = self._maybe_run_cleanup(result, last_info, x, mpcc_mult_G, mpcc_mult_H)
        return result
