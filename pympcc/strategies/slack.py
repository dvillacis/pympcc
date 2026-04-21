"""Slack (lifting) strategy for MPCC."""
from __future__ import annotations

import numpy as np

from ._base import BaseStrategy
from ..result import IterationInfo, MPCCResult

_INF = 2e19

_DEFAULTS = dict(
    epsilon_0=1.0, reduction=0.1, max_iter=20,
    epsilon_min=1e-8, dual_warmstart=True,
)

_STAT_TOL = 1e-6   # geometric tolerance for biactive-set detection


class SlackStrategy(BaseStrategy):
    """
    Slack (lifting) strategy for MPCC.

    Introduces explicit slack variables ``s_G`` and ``s_H`` that are pinned
    to ``G(x)`` and ``H(x)`` via equality constraints.  The complementarity
    condition is then imposed on the slacks only::

        min  f(x)
        s.t. g(x) ≤ 0,       h(x) = 0
             G(x) - s_G = 0,  H(x) - s_H = 0
             s_G ≥ 0,          s_H ≥ 0
             s_G · s_H ≤ ε

    Jacobian block structure::

            x (n)          s_G (n_comp)    s_H (n_comp)
        [  Jg           |       0         |       0      ]   ← g
        [  Jh           |       0         |       0      ]   ← h
        [  JG           |      −I         |       0      ]   ← G − s_G
        [  JH           |       0         |      −I      ]   ← H − s_H
        [   0           |  diag(s_H)      |  diag(s_G)   ]   ← s_G · s_H

    The x-block of the complementarity rows is **structurally zero**.
    Total nnz for those rows is ``2 * n_comp``, independent of ``n``.
    This is the key benefit for large-n problems (imaging, contact mechanics,
    traffic networks) where ``n_comp ≪ n``.

    The strategy always uses a sparse NLP adapter, regardless of whether
    ``*_jacobian_sparsity`` fields are set on the problem.

    Parameters (passed as **strategy_options** to :class:`MPCCSolver`)
    ------------------------------------------------------------------
    epsilon_0 : float
        Initial relaxation parameter (default 1.0).
    reduction : float
        Multiplicative decrease per outer iteration (default 0.1).
    max_iter : int
        Maximum outer iterations (default 20).
    epsilon_min : float
        Outer loop terminates when ``ε < epsilon_min`` (default 1e-8).
    dual_warmstart : bool
        Warm-start IPOPT dual variables between outer iterations
        (default ``True``).

    Notes
    -----
    **Stationarity**: The multipliers for the pinning equality constraints
    ``G − s_G = 0`` and ``H − s_H = 0`` do not carry the same sign
    guarantees as the G ≥ 0 / H ≥ 0 lower-bound multipliers used by the
    non-lifted strategies.  Therefore:

    - No biactive pairs (all ``G_i > tol`` or ``H_i > tol``):
      vacuously ``"S-stationary"``.
    - Biactive pairs present: ``"unknown"`` (multiplier signs are
      ambiguous in the lifted formulation).
    """

    name = "slack"

    def __init__(self, problem, ipopt_options: dict, **kwargs) -> None:
        super().__init__(problem, ipopt_options, callback=kwargs.pop("callback", None))
        opts = {**_DEFAULTS, **kwargs}
        self.epsilon_0: float = opts["epsilon_0"]
        self.reduction: float = opts["reduction"]
        self.max_iter: int = opts["max_iter"]
        self.epsilon_min: float = opts["epsilon_min"]
        self.dual_warmstart: bool = bool(opts["dual_warmstart"])

    # ------------------------------------------------------------------ #
    # Lifted-space helpers                                                 #
    # ------------------------------------------------------------------ #

    def _build_lifted_jac_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rows, cols)`` COO pattern for the full lifted NLP Jacobian."""
        p = self.problem
        n, n_c = p.n, p.n_comp
        col_sG = n
        col_sH = n + n_c
        row_pinG = p.n_ineq + p.n_eq
        row_pinH = row_pinG + n_c
        row_comp = row_pinH + n_c

        all_rows: list[np.ndarray] = []
        all_cols: list[np.ndarray] = []

        # g block (x-columns only; no slack columns)
        if p.n_ineq > 0:
            if p.ineq_jacobian_sparsity is not None:
                r = np.asarray(p.ineq_jacobian_sparsity[0])
                c = np.asarray(p.ineq_jacobian_sparsity[1])
            else:
                r = np.repeat(np.arange(p.n_ineq), n)
                c = np.tile(np.arange(n), p.n_ineq)
            all_rows.append(r)
            all_cols.append(c)

        # h block (x-columns only)
        if p.n_eq > 0:
            if p.eq_jacobian_sparsity is not None:
                r = np.asarray(p.eq_jacobian_sparsity[0]) + p.n_ineq
                c = np.asarray(p.eq_jacobian_sparsity[1])
            else:
                r = np.repeat(np.arange(p.n_ineq, p.n_ineq + p.n_eq), n)
                c = np.tile(np.arange(n), p.n_eq)
            all_rows.append(r)
            all_cols.append(c)

        # G-pinning rows: JG x-block + −I on s_G diagonal
        if p.comp_G_jacobian_sparsity is not None:
            r = np.asarray(p.comp_G_jacobian_sparsity[0]) + row_pinG
            c = np.asarray(p.comp_G_jacobian_sparsity[1])
        else:
            r = np.repeat(np.arange(n_c), n) + row_pinG
            c = np.tile(np.arange(n), n_c)
        all_rows.append(r)
        all_cols.append(c)
        all_rows.append(np.arange(n_c) + row_pinG)
        all_cols.append(np.arange(n_c) + col_sG)

        # H-pinning rows: JH x-block + −I on s_H diagonal
        if p.comp_H_jacobian_sparsity is not None:
            r = np.asarray(p.comp_H_jacobian_sparsity[0]) + row_pinH
            c = np.asarray(p.comp_H_jacobian_sparsity[1])
        else:
            r = np.repeat(np.arange(n_c), n) + row_pinH
            c = np.tile(np.arange(n), n_c)
        all_rows.append(r)
        all_cols.append(c)
        all_rows.append(np.arange(n_c) + row_pinH)
        all_cols.append(np.arange(n_c) + col_sH)

        # s_G · s_H rows: diag(s_H) at s_G cols, diag(s_G) at s_H cols
        comp_r = np.arange(n_c) + row_comp
        all_rows.append(comp_r)
        all_cols.append(np.arange(n_c) + col_sG)
        all_rows.append(comp_r.copy())
        all_cols.append(np.arange(n_c) + col_sH)

        return np.concatenate(all_rows), np.concatenate(all_cols)

    def _build_jax_hessian_slack(self):
        """Build exact Lagrangian Hessian in lifted z=[x,s_G,s_H] space via JAX."""
        import jax
        import jax.numpy as jnp
        import numpy as np_
        p = self.problem
        n, n_c = p.n, p.n_comp
        n_g, n_h = p.n_ineq, p.n_eq
        m = n_g + n_h + 3 * n_c  # [g, h, G-s_G, H-s_H, s_G*s_H]
        col_sG = n
        col_sH = n + n_c

        def lagrangian(z, lam, obj_factor):
            x   = z[:n]
            s_G = z[col_sG:col_sH]
            s_H = z[col_sH:]
            val = obj_factor * p.objective(x)
            if n_g:
                val = val + jnp.dot(lam[:n_g], p.ineq_constraints(x))
            if n_h:
                val = val + jnp.dot(lam[n_g:n_g + n_h], p.eq_constraints(x))
            G = p.comp_G(x)
            H = p.comp_H(x)
            off = n_g + n_h
            val = val + jnp.dot(lam[off:off + n_c], G - s_G)
            val = val + jnp.dot(lam[off + n_c:off + 2 * n_c], H - s_H)
            val = val + jnp.dot(lam[off + 2 * n_c:], s_G * s_H)
            return val

        x0 = p.x0
        G0 = np_.asarray(p.comp_G(x0))
        H0 = np_.asarray(p.comp_H(x0))
        z0 = np_.concatenate([x0, np_.maximum(G0, 0.0), np_.maximum(H0, 0.0)])
        z0_j   = jnp.asarray(z0, dtype=float)
        lam0_j = jnp.ones(m, dtype=float)
        H_dense = np_.asarray(
            jax.hessian(lambda z: lagrangian(z, lam0_j, 1.0))(z0_j),
            dtype=float,
        )
        mask = np_.tril(np_.abs(H_dense)) > p.jax_sparsity_tol
        rows, cols = np_.where(mask)
        rows = rows.astype(np_.intp)
        cols = cols.astype(np_.intp)
        r_j, c_j = jnp.array(rows), jnp.array(cols)

        @jax.jit
        def _hess_sparse(z, lam, obj_factor):
            H = jax.hessian(lambda zk: lagrangian(zk, lam, obj_factor))(z)
            return H[r_j, c_j]

        def hess_fn(z, lam, obj_factor):
            return np_.asarray(
                _hess_sparse(
                    jnp.asarray(z, dtype=float),
                    jnp.asarray(lam, dtype=float),
                    float(obj_factor),
                ),
                dtype=float,
            )

        n_z = n + 2 * n_c
        return hess_fn, (rows, cols), n_z

    def _build_lifted_nlp(
        self,
        cl: np.ndarray,
        cu: np.ndarray,
        con_fn,
        jac_fn,
        jac_structure: tuple[np.ndarray, np.ndarray],
        obj_fn,
        grad_fn,
        n_z: int,
        xl_z: np.ndarray,
        xu_z: np.ndarray,
        hess_fn=None,
        hess_sparsity=None,
    ):
        """Build a :class:`_SparseNLP` for the lifted ``(x, s_G, s_H)`` space."""
        from .._nlp import _SparseNLP, _HessianMixin

        if hess_fn is not None:
            cls = type("_NLPWithHess", (_HessianMixin, _SparseNLP), {})
        else:
            cls = _SparseNLP

        nlp = cls(
            n=n_z, m=len(cl), xl=xl_z, xu=xu_z, cl=cl, cu=cu,
            obj_fn=obj_fn, grad_fn=grad_fn, con_fn=con_fn, jac_fn=jac_fn,
            jac_rows=jac_structure[0], jac_cols=jac_structure[1],
            hess_fn=hess_fn, hess_sparsity=hess_sparsity,
        )
        for key, val in self.ipopt_options.items():
            nlp.add_option(key, val)
        return nlp

    # ------------------------------------------------------------------ #
    # Solver                                                               #
    # ------------------------------------------------------------------ #

    def solve(self) -> MPCCResult:
        p = self.problem
        n, n_c = p.n, p.n_comp
        n_z = n + 2 * n_c
        col_sG = n
        col_sH = n + n_c

        # Variable bounds in lifted space
        xl_z = np.concatenate([p.xl, np.zeros(n_c), np.zeros(n_c)])
        xu_z = np.concatenate([p.xu, np.full(n_c, _INF), np.full(n_c, _INF)])

        # Build sparsity structure once before the loop
        jac_structure = self._build_lifted_jac_structure()

        def make_cl(eps: float) -> np.ndarray:
            return np.concatenate([
                np.full(p.n_ineq, -_INF),
                np.zeros(p.n_eq),
                np.zeros(n_c),        # G − s_G = 0  (lower)
                np.zeros(n_c),        # H − s_H = 0  (lower)
                np.full(n_c, -_INF),  # s_G · s_H   (no lower bound)
            ])

        def make_cu(eps: float) -> np.ndarray:
            return np.concatenate([
                np.zeros(p.n_ineq),
                np.zeros(p.n_eq),
                np.zeros(n_c),        # G − s_G = 0  (upper)
                np.zeros(n_c),        # H − s_H = 0  (upper)
                np.full(n_c, eps),    # s_G · s_H ≤ ε
            ])

        def constraints(z: np.ndarray) -> np.ndarray:
            x   = z[:n]
            s_G = z[col_sG:col_sH]
            s_H = z[col_sH:]
            std_parts = self._eval_standard_con_values(x)
            G = np.asarray(p.comp_G(x))
            H = np.asarray(p.comp_H(x))
            return np.concatenate([*std_parts, G - s_G, H - s_H, s_G * s_H])

        # Pre-allocate the full flat Jacobian output (fixed size = total nnz).
        # Fill sections in-place on every callback — zero concatenation overhead.
        _jac_buf = np.empty(len(jac_structure[0]))
        _jac_buf[...] = 0.0   # -1.0 sections written once; complement updated each call

        # Pre-compute byte offsets for each named section in _jac_buf.
        # Sections in order: g, h, JG-x, -I_G, JH-x, -I_H, diag(s_H), diag(s_G)
        def _jac_seg_len(sparsity, n_rows, n_cols):
            return len(sparsity[0]) if sparsity is not None else n_rows * n_cols

        _seg_g   = _jac_seg_len(p.ineq_jacobian_sparsity,    p.n_ineq,  n)
        _seg_h   = _jac_seg_len(p.eq_jacobian_sparsity,      p.n_eq,    n)
        _seg_JG  = _jac_seg_len(p.comp_G_jacobian_sparsity,  n_c,       n)
        _seg_JH  = _jac_seg_len(p.comp_H_jacobian_sparsity,  n_c,       n)
        _off_g   = 0
        _off_h   = _off_g  + _seg_g
        _off_JG  = _off_h  + _seg_h
        _off_nIG = _off_JG + _seg_JG   # -I on s_G diagonal
        _off_JH  = _off_nIG + n_c
        _off_nIH = _off_JH + _seg_JH   # -I on s_H diagonal
        _off_sH  = _off_nIH + n_c      # diag(s_H) for s_G block
        _off_sG  = _off_sH  + n_c      # diag(s_G) for s_H block

        # The -I sections are constant; write them once.
        _jac_buf[_off_nIG:_off_nIG + n_c] = -1.0
        _jac_buf[_off_nIH:_off_nIH + n_c] = -1.0

        def jacobian(z: np.ndarray) -> np.ndarray:
            x   = z[:n]
            s_G = z[col_sG:col_sH]
            s_H = z[col_sH:]
            # g block
            if p.n_ineq > 0:
                J = np.asarray(p.ineq_jacobian(x), dtype=float)
                v = J if J.ndim == 1 else J.ravel()
                _jac_buf[_off_g:_off_g + _seg_g] = v
            # h block
            if p.n_eq > 0:
                J = np.asarray(p.eq_jacobian(x), dtype=float)
                v = J if J.ndim == 1 else J.ravel()
                _jac_buf[_off_h:_off_h + _seg_h] = v
            # G-pinning x-block
            vG = np.asarray(p.comp_G_jacobian(x), dtype=float)
            _jac_buf[_off_JG:_off_JG + _seg_JG] = (
                vG if vG.ndim == 1 else vG.ravel())
            # H-pinning x-block
            vH = np.asarray(p.comp_H_jacobian(x), dtype=float)
            _jac_buf[_off_JH:_off_JH + _seg_JH] = (
                vH if vH.ndim == 1 else vH.ravel())
            # s_G · s_H: diag(s_H) for s_G cols, diag(s_G) for s_H cols
            _jac_buf[_off_sH:_off_sH + n_c] = s_H
            _jac_buf[_off_sG:_off_sG + n_c] = s_G
            return _jac_buf

        def obj_lifted(z: np.ndarray) -> float:
            return float(p.objective(z[:n]))

        def grad_lifted(z: np.ndarray) -> np.ndarray:
            g = np.asarray(p.gradient(z[:n]), dtype=float)
            return np.concatenate([g, np.zeros(2 * n_c)])

        hess_fn, hess_sparsity = None, None
        if self._has_jax_hessian():
            hess_fn, hess_sparsity, _ = self._build_jax_hessian_slack()

        # Initialise lifted variable vector
        x = p.x0.copy()
        G0 = np.asarray(p.comp_G(x))
        H0 = np.asarray(p.comp_H(x))
        z = np.concatenate([x, np.maximum(G0, 0.0), np.maximum(H0, 0.0)])

        history: list[IterationInfo] = []
        eps = self.epsilon_0
        last_info: dict = {}
        warm_dual: dict = {}

        for _ in range(self.max_iter):
            nlp = self._build_lifted_nlp(
                make_cl(eps), make_cu(eps),
                constraints, jacobian, jac_structure,
                obj_lifted, grad_lifted, n_z, xl_z, xu_z,
                hess_fn=hess_fn, hess_sparsity=hess_sparsity,
            )
            if self.dual_warmstart and warm_dual:
                nlp.add_option("warm_start_init_point", "yes")
                z, last_info = nlp.solve(z, **warm_dual)
            else:
                z, last_info = nlp.solve(z)
            if self.dual_warmstart:
                warm_dual = {
                    "lagrange": last_info["mult_g"],
                    "zl":       last_info["mult_x_L"],
                    "zu":       last_info["mult_x_U"],
                }

            x   = z[:n]
            s_G = z[col_sG:col_sH]
            s_H = z[col_sH:]
            comp_residual = float(np.max(np.abs(s_G * s_H)))
            comp_residual_mean = float(np.mean(np.abs(s_G * s_H)))
            history.append(IterationInfo(
                epsilon=eps,
                x=x.copy(),
                obj=float(p.objective(x)),
                status=last_info["status"],
                message=self._decode_msg(last_info["status_msg"]),
                comp_residual=comp_residual,
                comp_residual_mean=comp_residual_mean,
                n_ipopt_iter=nlp.n_ipopt_iter,
            ))
            if self.callback is not None:
                self.callback(len(history) - 1, history[-1])

            eps *= self.reduction
            if eps < self.epsilon_min:
                break

        x = z[:n]
        G = np.asarray(p.comp_G(x))
        H = np.asarray(p.comp_H(x))

        # Stationarity: multiplier signs are ambiguous in the lifted formulation.
        biactive = np.where((G <= _STAT_TOL) & (H <= _STAT_TOL))[0]
        stationarity = "unknown" if len(biactive) > 0 else "S-stationary"

        result = MPCCResult(
            x=x,
            obj=float(p.objective(x)),
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
        result.stationarity = stationarity
        return result
