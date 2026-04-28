from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

__all__ = ["MPCCProblem"]


@dataclass
class MPCCProblem:
    """
    Defines an MPCC in the form::

        min  f(x)
        s.t. g(x) <= 0              (n_ineq inequality constraints)
             h(x)  = 0              (n_eq  equality   constraints)
             G(x) >= 0  }
             H(x) >= 0  }           (n_comp complementarity pairs)
             G(x)^T H(x) = 0  }

    Parameters
    ----------
    n : int
        Number of decision variables.
    n_comp : int
        Number of complementarity pairs.
    x0 : array-like, shape (n,)
        Initial guess.
    objective : callable
        ``f(x) -> float``
    gradient : callable or ``"fd"``
        ``grad_f(x) -> ndarray, shape (n,)``.
        Pass ``"fd"`` to use a finite-difference approximation.
    comp_G : callable
        ``G(x) -> ndarray, shape (n_comp,)``  with ``G(x) >= 0``
    comp_G_jacobian : callable or ``"fd"``
        ``jac_G(x) -> ndarray, shape (n_comp, n)`` (dense), or a 1-D array
        of ``nnz`` values when ``comp_G_jacobian_sparsity`` is set.
        Pass ``"fd"`` to use a finite-difference approximation.
    comp_H : callable
        ``H(x) -> ndarray, shape (n_comp,)``  with ``H(x) >= 0``
    comp_H_jacobian : callable or ``"fd"``
        ``jac_H(x) -> ndarray, shape (n_comp, n)`` (dense), or 1-D ``nnz``
        values when ``comp_H_jacobian_sparsity`` is set.
        Pass ``"fd"`` to use a finite-difference approximation.
    xl : array-like, shape (n,), optional
        Lower bounds on x (default: ``-inf``).
    xu : array-like, shape (n,), optional
        Upper bounds on x (default: ``+inf``).
    n_ineq : int
        Number of inequality constraints ``g(x) <= 0``.
    ineq_constraints : callable, optional
        ``g(x) -> ndarray, shape (n_ineq,)``
    ineq_jacobian : callable or ``"fd"``, optional
        ``jac_g(x) -> ndarray, shape (n_ineq, n)``.
        Pass ``"fd"`` to use a finite-difference approximation.
        Requires ``n_ineq > 0`` and ``ineq_constraints``.
    n_eq : int
        Number of equality constraints ``h(x) = 0``.
    eq_constraints : callable, optional
        ``h(x) -> ndarray, shape (n_eq,)``
    eq_jacobian : callable or ``"fd"``, optional
        ``jac_h(x) -> ndarray, shape (n_eq, n)``.
        Pass ``"fd"`` to use a finite-difference approximation.
        Requires ``n_eq > 0`` and ``eq_constraints``.
    fd_h : float, optional
        Step size used for all finite-difference approximations
        (default: ``sqrt(machine_epsilon)`` ≈ 1.49e-8).
    fd_mode : {"forward", "central"}, optional
        Finite-difference scheme (default: ``"forward"``).
        ``"forward"`` costs n+1 evaluations per Jacobian column (O(h) error).
        ``"central"`` costs 2n evaluations per column (O(h²) error).

    Notes
    -----
    Jacobians may be dense ``(n_con, n)`` arrays or, when the corresponding
    ``*_jacobian_sparsity`` field is set, 1-D arrays of ``nnz`` nonzero values
    in COO order.  Passing ``"fd"`` always produces dense Jacobians; combine
    with sparsity fields only if the problem is genuinely sparse.

    A ``UserWarning`` is emitted at construction whenever any ``"fd"``
    sentinel is active.  Finite differences are suitable for prototyping;
    use exact Jacobians in production for speed and accuracy.
    """

    # ------------------------------------------------------------------ #
    # Required                                                             #
    # ------------------------------------------------------------------ #
    n: int
    n_comp: int
    x0: np.ndarray
    objective: Callable[[np.ndarray], float]

    # ------------------------------------------------------------------ #
    # Complementarity callables.                                           #
    # Either comp_G / comp_H must be provided, or all pairs may be        #
    # declared via ``comp_var_pairs`` (see below), in which case both     #
    # may be left as ``None``.  When a mix is used,                       #
    # ``n_comp = n_base + len(comp_var_pairs)`` where ``n_base`` is       #
    # inferred from the pair count of the base comp_G/comp_H.             #
    # ------------------------------------------------------------------ #
    comp_G: Optional[Callable[[np.ndarray], np.ndarray]] = None
    comp_H: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # ------------------------------------------------------------------ #
    # Derivative callables — required, but may be omitted when the       #
    # ``derivatives`` keyword (see below) auto-fills them with ``"jax"`` #
    # or ``"fd"`` sentinels.                                             #
    # ------------------------------------------------------------------ #
    gradient: Optional[Union[Callable[[np.ndarray], np.ndarray], str]] = None
    comp_G_jacobian: Optional[Union[Callable[[np.ndarray], np.ndarray], str]] = None
    comp_H_jacobian: Optional[Union[Callable[[np.ndarray], np.ndarray], str]] = None

    # ------------------------------------------------------------------ #
    # Default derivative source.  When ``"jax"`` (resp. ``"fd"``) every  #
    # unset derivative field — gradient, every Jacobian — is             #
    # auto-filled with the corresponding sentinel before resolution.     #
    # User-supplied callables and explicit sentinels override.           #
    # ------------------------------------------------------------------ #
    derivatives: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Variable bounds (default: unbounded)                                 #
    # ------------------------------------------------------------------ #
    xl: Optional[np.ndarray] = None
    xu: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Standard inequality constraints  g(x) <= 0                          #
    # ------------------------------------------------------------------ #
    n_ineq: int = 0
    ineq_constraints: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ineq_jacobian: Optional[Union[Callable[[np.ndarray], np.ndarray], str]] = None

    # ------------------------------------------------------------------ #
    # Standard equality constraints  h(x) = 0                             #
    # ------------------------------------------------------------------ #
    n_eq: int = 0
    eq_constraints: Optional[Callable[[np.ndarray], np.ndarray]] = None
    eq_jacobian: Optional[Union[Callable[[np.ndarray], np.ndarray], str]] = None

    # ------------------------------------------------------------------ #
    # Finite-difference options                                            #
    # ------------------------------------------------------------------ #
    fd_h: float = float(np.sqrt(np.finfo(float).eps))
    fd_mode: str = "forward"

    # ------------------------------------------------------------------ #
    # JAX autodiff options                                                 #
    # ------------------------------------------------------------------ #
    use_jax_hessian: bool = False
    jax_sparsity_tol: float = 1e-12

    # ------------------------------------------------------------------ #
    # Manual Lagrangian Hessian (optional)                                 #
    # ------------------------------------------------------------------ #
    # When provided, used in place of JAX autodiff or L-BFGS.
    #
    # --- lagrangian_hessian / lagrangian_hessian_sparsity ---
    # Compatible strategies: direct, scholtes, lin_fukushima.
    # NOT compatible with augmented_lagrangian (PHR penalty adds objective
    # Hessian terms not captured here) or smoothing (phi_eps Hessian).
    #
    # Signature:
    #   lagrangian_hessian(x, lagrange, obj_factor) -> ndarray, shape (nnz,)
    #
    # The `lagrange` vector follows the shared ordering for these strategies:
    #   [lam_g (n_ineq), lam_h (n_eq), lam_G (n_comp), lam_H (n_comp),
    #    lam_GH (n_comp, optional)]
    # where lam_GPH in lin_fukushima (G+H block) is beyond lam_GH and has
    # zero Hessian contribution (G+H is linear, so it is silently ignored).
    #
    # --- lagrangian_hessian_slack / lagrangian_hessian_slack_sparsity ---
    # For the slack strategy only.  The slack strategy lifts the variable
    # space to z = [x (n), s_G (n_comp), s_H (n_comp)], so the Hessian
    # is (n+2*n_comp) × (n+2*n_comp).  The callable receives the full
    # lifted vector z.
    #
    # Signature:
    #   lagrangian_hessian_slack(z, lagrange, obj_factor) -> ndarray, shape (nnz,)
    #
    # The `lagrange` ordering for slack is:
    #   [lam_g, lam_h, lam_{G-sG}, lam_{H-sH}, lam_{sG·sH}]
    #
    # lagrangian_hessian_sparsity / lagrangian_hessian_slack_sparsity:
    #   COO (rows, cols), 0-based, lower triangle (row >= col).
    lagrangian_hessian: Optional[Callable] = None
    lagrangian_hessian_sparsity: Optional[tuple] = None
    lagrangian_hessian_slack: Optional[Callable] = None
    lagrangian_hessian_slack_sparsity: Optional[tuple] = None

    # ------------------------------------------------------------------ #
    # Optional sparse Jacobian structures (COO format, 0-based indices)   #
    # ------------------------------------------------------------------ #
    # When provided, the corresponding *_jacobian callable must return a  #
    # 1-D float array of nnz values instead of a dense 2-D matrix.       #
    comp_G_jacobian_sparsity: Optional[tuple] = None
    comp_H_jacobian_sparsity: Optional[tuple] = None
    ineq_jacobian_sparsity:   Optional[tuple] = None
    eq_jacobian_sparsity:     Optional[tuple] = None

    # ------------------------------------------------------------------ #
    # Variable-paired complementarity (MCP form).                         #
    # Each entry is (var_idx, h_fn) or (var_idx, h_fn, h_jac_fn).        #
    # var_idx  — index j in [0, n); declares x[j] >= 0 as the G side.   #
    # h_fn     — callable, h_fn(x) -> float; the H side.                 #
    # h_jac_fn — callable, h_jac_fn(x) -> ndarray shape (n,); or None   #
    #            to use forward finite differences for that row.          #
    # xl[var_idx] is silently clamped to max(xl[var_idx], 0.0).          #
    # ------------------------------------------------------------------ #
    comp_var_pairs: Optional[list] = None

    # ------------------------------------------------------------------ #
    # Optional per-pair diagonal rescaling for the complementarity block #
    # ------------------------------------------------------------------ #
    # When set, every strategy operates on (s_G * G, s_H * H) instead of #
    # raw (G, H).  Multipliers returned by the solver are in scaled      #
    # space; multiply by the corresponding scale to recover original     #
    # KKT duals.  See pympcc.unscale_multipliers().                      #
    comp_G_scale: Optional[np.ndarray] = None
    comp_H_scale: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #

    @property
    def is_sparse(self) -> bool:
        """True if any Jacobian block has an explicit sparsity structure."""
        return any(s is not None for s in [
            self.comp_G_jacobian_sparsity, self.comp_H_jacobian_sparsity,
            self.ineq_jacobian_sparsity, self.eq_jacobian_sparsity,
        ])

    @property
    def has_comp_scale(self) -> bool:
        """True if either complementarity scale vector is set."""
        return self.comp_G_scale is not None or self.comp_H_scale is not None

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        if self.xl is None:
            self.xl = np.full(self.n, -np.inf)
        if self.xu is None:
            self.xu = np.full(self.n, np.inf)
        self.xl = np.asarray(self.xl, dtype=float)
        self.xu = np.asarray(self.xu, dtype=float)
        self._apply_derivatives_default()
        # jax/fd resolution skips comp_G/H Jacobians when comp_G is None
        # (they will be built by _normalize_var_pairs from the var-pair declarations).
        self._resolve_jax_fields()
        self._resolve_fd_fields()
        self._normalize_var_pairs()   # merges var-pairs into comp_G/H and builds Jacobians
        self._check_derivatives_resolved()
        self._validate()

    def _apply_derivatives_default(self) -> None:
        """Auto-fill ``None`` derivative fields with the ``derivatives`` sentinel.

        ``derivatives="jax"`` and ``derivatives="fd"`` provide a single
        keyword opt-in into autodiff or finite differences for every
        derivative field the user hasn't supplied.  Fields that already
        hold a callable or an explicit sentinel are left untouched.
        """
        if self.derivatives is None:
            return
        if self.derivatives not in ("jax", "fd"):
            raise ValueError(
                f"derivatives must be None, 'jax', or 'fd'; got {self.derivatives!r}"
            )
        sentinel = self.derivatives

        if self.gradient is None:
            self.gradient = sentinel
        if self.comp_G_jacobian is None:
            self.comp_G_jacobian = sentinel
        if self.comp_H_jacobian is None:
            self.comp_H_jacobian = sentinel
        if self.n_ineq > 0 and self.ineq_constraints is not None and self.ineq_jacobian is None:
            self.ineq_jacobian = sentinel
        if self.n_eq > 0 and self.eq_constraints is not None and self.eq_jacobian is None:
            self.eq_jacobian = sentinel

    def _normalize_var_pairs(self) -> None:
        """Merge ``comp_var_pairs`` declarations into ``comp_G`` / ``comp_H`` and their Jacobians.

        After this method runs:
        * ``comp_G`` and ``comp_H`` are non-None callables.
        * ``comp_G_jacobian`` is an exact callable (identity rows for var-pair block).
        * ``comp_H_jacobian`` is a callable (user-provided or fd per var-pair row).
        """
        if not self.comp_var_pairs:
            if self.comp_G is None or self.comp_H is None:
                raise ValueError(
                    "comp_G and comp_H are required when comp_var_pairs is not used"
                )
            return

        from ._fd import fd_jacobian as _fd_jac

        pairs = list(self.comp_var_pairs)
        k = len(pairs)
        n = self.n
        h_fd = self.fd_h
        h_mode = self.fd_mode

        # Parse tuples: (var_idx, h_fn) or (var_idx, h_fn, h_jac_fn)
        var_idxs: list[int] = []
        h_fns: list = []
        h_jac_fns: list = []  # None → build fd inline
        for entry in pairs:
            if len(entry) == 2:
                vi, hf = entry
                hj = None
            elif len(entry) == 3:
                vi, hf, hj = entry
            else:
                raise ValueError(
                    "comp_var_pairs entries must be (var_idx, h_fn) or (var_idx, h_fn, h_jac_fn)"
                )
            if not (0 <= int(vi) < n):
                raise ValueError(
                    f"comp_var_pairs: var_idx {vi} is out of range [0, {n})"
                )
            var_idxs.append(int(vi))
            h_fns.append(hf)
            h_jac_fns.append(hj)

        var_idx_arr = np.array(var_idxs, dtype=np.intp)

        # Enforce lower bound >= 0 for each paired variable
        for vi in var_idxs:
            self.xl[vi] = max(self.xl[vi], 0.0)

        # Build per-row H Jacobians (resolve fd now for any missing)
        resolved_h_jac: list = []
        for hf, hj in zip(h_fns, h_jac_fns):
            if hj is not None:
                resolved_h_jac.append(hj)
            else:
                def _scalar_fd(x, _f=hf, _h=h_fd, _m=h_mode):
                    return _fd_jac(_f, 1, n, _h, _m)(x)[0]
                resolved_h_jac.append(_scalar_fd)

        if self.comp_G is None:
            # All-var-pairs mode: every comp pair is declared via comp_var_pairs
            if k != self.n_comp:
                raise ValueError(
                    f"comp_var_pairs has {k} entries but n_comp={self.n_comp}; "
                    "when comp_G is None, len(comp_var_pairs) must equal n_comp"
                )

            def _G(x, _idx=var_idx_arr):
                return np.asarray(x, dtype=float)[_idx]

            def _H(x, _hfs=h_fns, _k=k):
                return np.array([float(np.asarray(hf(x)).ravel()[0]) for hf in _hfs])

            def _G_jac(x, _idx=var_idx_arr, _k=k, _n=n):
                J = np.zeros((_k, _n))
                J[np.arange(_k), _idx] = 1.0
                return J

            def _H_jac(x, _jfs=resolved_h_jac):
                return np.stack([jf(x) for jf in _jfs], axis=0)

            self.comp_G = _G
            self.comp_H = _H
            self.comp_G_jacobian = _G_jac
            self.comp_H_jacobian = _H_jac

        else:
            # Mixed mode: base comp_G/comp_H + k var-pair rows appended at the end
            n_base = self.n_comp - k
            if n_base <= 0:
                raise ValueError(
                    f"comp_var_pairs has {k} entries but n_comp={self.n_comp}; "
                    "mixed mode requires n_comp > len(comp_var_pairs) (at least one base pair)"
                )

            base_G = self.comp_G
            base_H = self.comp_H
            base_G_jac = self.comp_G_jacobian
            base_H_jac = self.comp_H_jacobian

            if not callable(base_G_jac):
                raise ValueError(
                    "Mixed comp_var_pairs requires comp_G_jacobian to be a resolved callable. "
                    "Pass derivatives='fd', derivatives='jax', or supply comp_G_jacobian explicitly."
                )
            if not callable(base_H_jac):
                raise ValueError(
                    "Mixed comp_var_pairs requires comp_H_jacobian to be a resolved callable. "
                    "Pass derivatives='fd', derivatives='jax', or supply comp_H_jacobian explicitly."
                )

            def _G(x, _bG=base_G, _idx=var_idx_arr):
                return np.concatenate([
                    np.asarray(_bG(x), dtype=float),
                    np.asarray(x, dtype=float)[_idx],
                ])

            def _H(x, _bH=base_H, _hfs=h_fns):
                return np.concatenate([
                    np.asarray(_bH(x), dtype=float),
                    np.array([float(np.asarray(hf(x)).ravel()[0]) for hf in _hfs]),
                ])

            def _G_jac(x, _bJac=base_G_jac, _idx=var_idx_arr, _k=k, _n=n):
                base_rows = np.asarray(_bJac(x), dtype=float)
                id_rows = np.zeros((_k, _n))
                id_rows[np.arange(_k), _idx] = 1.0
                return np.vstack([base_rows, id_rows])

            def _H_jac(x, _bJac=base_H_jac, _jfs=resolved_h_jac):
                base_rows = np.asarray(_bJac(x), dtype=float)
                extra_rows = np.stack([jf(x) for jf in _jfs], axis=0)
                return np.vstack([base_rows, extra_rows])

            self.comp_G = _G
            self.comp_H = _H
            self.comp_G_jacobian = _G_jac
            self.comp_H_jacobian = _H_jac

    def _check_derivatives_resolved(self) -> None:
        """Raise a clear error when a required derivative is still missing.

        Runs after ``_resolve_jax_fields`` and ``_resolve_fd_fields`` so
        any sentinel that survived resolution (e.g. JAX missing) has
        already been reported.
        """
        missing: list[str] = []
        if self.gradient is None or isinstance(self.gradient, str):
            missing.append("gradient")
        if self.comp_G_jacobian is None or isinstance(self.comp_G_jacobian, str):
            missing.append("comp_G_jacobian")
        if self.comp_H_jacobian is None or isinstance(self.comp_H_jacobian, str):
            missing.append("comp_H_jacobian")
        if self.n_ineq > 0 and (
            self.ineq_jacobian is None or isinstance(self.ineq_jacobian, str)
        ):
            missing.append("ineq_jacobian")
        if self.n_eq > 0 and (
            self.eq_jacobian is None or isinstance(self.eq_jacobian, str)
        ):
            missing.append("eq_jacobian")
        if missing:
            raise ValueError(
                f"Missing derivative callable(s): {missing}. "
                "Pass each as a callable, set the field to 'jax' or 'fd', "
                "or pass derivatives='jax' / derivatives='fd' to fill all "
                "unset derivative fields at once."
            )

    def _resolve_jax_fields(self) -> None:
        """Replace any ``"jax"`` sentinel with a JAX-autodiff callable."""
        from ._jax import HAS_JAX

        # Collect which fields hold the "jax" sentinel.
        _sentinel_fields = [
            self.gradient,
            self.comp_G_jacobian, self.comp_H_jacobian,
            self.ineq_jacobian, self.eq_jacobian,
        ]
        jax_fields = [v for v in _sentinel_fields if v == "jax"]

        if not jax_fields:
            return   # nothing to do — fast path for most users

        if not HAS_JAX:
            raise ImportError(
                "JAX is required for the 'jax' sentinel but is not installed. "
                "Install it with:  pip install 'pympcc[jax]'"
            )

        from ._jax import jax_gradient, jax_jacobian

        tol = self.jax_sparsity_tol
        jax_used: list[str] = []

        if self.gradient == "jax":
            self.gradient = jax_gradient(self.objective, self.n, self.x0, tol)
            jax_used.append("gradient")

        for attr, fn_attr, sparsity_attr, n_out in [
            ("comp_G_jacobian", "comp_G", "comp_G_jacobian_sparsity", self.n_comp),
            ("comp_H_jacobian", "comp_H", "comp_H_jacobian_sparsity", self.n_comp),
            ("ineq_jacobian", "ineq_constraints", "ineq_jacobian_sparsity", self.n_ineq),
            ("eq_jacobian", "eq_constraints", "eq_jacobian_sparsity", self.n_eq),
        ]:
            if getattr(self, attr) != "jax":
                continue
            fn = getattr(self, fn_attr)
            # Defer comp_G/H Jacobian resolution to _normalize_var_pairs when comp_G is None
            if fn is None and attr in ("comp_G_jacobian", "comp_H_jacobian"):
                continue
            if fn is None or n_out == 0:
                raise ValueError(
                    f"{attr}='jax' requires the corresponding function "
                    f"({fn_attr}) to be set and n_out > 0"
                )
            jac_fn, sparsity = jax_jacobian(fn, n_out, self.n, self.x0, tol)
            setattr(self, attr, jac_fn)
            setattr(self, sparsity_attr, sparsity)
            jax_used.append(attr)

        if jax_used:
            warnings.warn(
                f"JAX autodiff active for: {', '.join(jax_used)}. "
                "Ensure all primal callables are JAX-differentiable.",
                UserWarning,
                stacklevel=3,
            )

    def _resolve_fd_fields(self) -> None:
        """Replace any ``"fd"`` sentinel with a finite-difference callable."""
        from ._fd import fd_gradient, fd_jacobian

        if self.fd_mode not in ("forward", "central"):
            raise ValueError(
                f"fd_mode must be 'forward' or 'central', got {self.fd_mode!r}"
            )
        h, mode = self.fd_h, self.fd_mode
        fd_used: list[str] = []

        if self.gradient == "fd":
            self.gradient = fd_gradient(self.objective, self.n, h, mode)
            fd_used.append("gradient")
        if self.comp_G_jacobian == "fd":
            if self.comp_G is not None:
                self.comp_G_jacobian = fd_jacobian(
                    self.comp_G, self.n_comp, self.n, h, mode
                )
                fd_used.append("comp_G_jacobian")
            # else: comp_G is None (var-pairs only) — _normalize_var_pairs builds exact G Jac
        if self.comp_H_jacobian == "fd":
            if self.comp_H is not None:
                self.comp_H_jacobian = fd_jacobian(
                    self.comp_H, self.n_comp, self.n, h, mode
                )
                fd_used.append("comp_H_jacobian")
            # else: comp_H is None (var-pairs only) — _normalize_var_pairs builds H Jac (fd per row)
        if self.ineq_jacobian == "fd":
            if self.n_ineq == 0 or self.ineq_constraints is None:
                raise ValueError(
                    "ineq_jacobian='fd' requires n_ineq > 0 and ineq_constraints"
                )
            self.ineq_jacobian = fd_jacobian(
                self.ineq_constraints, self.n_ineq, self.n, h, mode
            )
            fd_used.append("ineq_jacobian")
        if self.eq_jacobian == "fd":
            if self.n_eq == 0 or self.eq_constraints is None:
                raise ValueError(
                    "eq_jacobian='fd' requires n_eq > 0 and eq_constraints"
                )
            self.eq_jacobian = fd_jacobian(
                self.eq_constraints, self.n_eq, self.n, h, mode
            )
            fd_used.append("eq_jacobian")

        if fd_used:
            warnings.warn(
                f"Finite-difference Jacobian active for: {', '.join(fd_used)}. "
                "Suitable for prototyping; use exact Jacobians in production.",
                UserWarning,
                stacklevel=3,
            )

    def _validate(self) -> None:
        x0 = self.x0
        assert self.xl is not None and self.xu is not None  # set by __post_init__
        if x0.shape != (self.n,):
            raise ValueError(f"x0 must have shape ({self.n},), got {x0.shape}")
        if self.xl.shape != (self.n,):
            raise ValueError(f"xl must have shape ({self.n},)")
        if self.xu.shape != (self.n,):
            raise ValueError(f"xu must have shape ({self.n},)")
        if not np.all(self.xl <= self.xu):
            raise ValueError("xl must be <= xu element-wise")
        # Warn when x0 violates finite bounds — IPOPT projects x0 internally.
        finite_lb = np.isfinite(self.xl)
        finite_ub = np.isfinite(self.xu)
        viol = np.where(
            (finite_lb & (self.x0 < self.xl)) | (finite_ub & (self.x0 > self.xu))
        )[0]
        if viol.size:
            details = ", ".join(
                f"x0[{i}]={self.x0[i]:.4g} not in [{self.xl[i]:.4g}, {self.xu[i]:.4g}]"
                for i in viol[:5]
            )
            if viol.size > 5:
                details += f" ... ({viol.size} total)"
            warnings.warn(
                f"x0 violates bounds at {viol.size} index(es): {details}",
                UserWarning,
                stacklevel=3,
            )
        if self.n_comp < 1:
            raise ValueError("n_comp must be >= 1")

        if not np.isfinite(self.objective(x0)):
            raise ValueError("objective(x0) returned non-finite value (NaN or Inf)")
        if not np.all(np.isfinite(self.gradient(x0))):  # type: ignore[operator]
            raise ValueError("gradient(x0) returned non-finite values (NaN or Inf)")

        self._check_shape("comp_G", self.comp_G, x0, (self.n_comp,))
        if self.comp_G_jacobian_sparsity is None:
            self._check_shape("comp_G_jacobian", self.comp_G_jacobian, x0,  # type: ignore[arg-type]
                              (self.n_comp, self.n))
        self._check_shape("comp_H", self.comp_H, x0, (self.n_comp,))
        if self.comp_H_jacobian_sparsity is None:
            self._check_shape("comp_H_jacobian", self.comp_H_jacobian, x0,  # type: ignore[arg-type]
                              (self.n_comp, self.n))

        if self.n_ineq > 0:
            if self.ineq_constraints is None or self.ineq_jacobian is None:
                raise ValueError(
                    "ineq_constraints and ineq_jacobian are required when n_ineq > 0"
                )
            self._check_shape("ineq_constraints", self.ineq_constraints, x0,
                               (self.n_ineq,))
            if self.ineq_jacobian_sparsity is None:
                self._check_shape("ineq_jacobian", self.ineq_jacobian, x0,  # type: ignore[arg-type]
                                   (self.n_ineq, self.n))

        if self.n_eq > 0:
            if self.eq_constraints is None or self.eq_jacobian is None:
                raise ValueError(
                    "eq_constraints and eq_jacobian are required when n_eq > 0"
                )
            self._check_shape("eq_constraints", self.eq_constraints, x0,
                               (self.n_eq,))
            if self.eq_jacobian_sparsity is None:
                self._check_shape("eq_jacobian", self.eq_jacobian, x0,  # type: ignore[arg-type]
                                   (self.n_eq, self.n))

        # Sparse structure validation
        for _name, _fn, _sparsity, _n_rows in [
            ("comp_G_jacobian", self.comp_G_jacobian,
             self.comp_G_jacobian_sparsity, self.n_comp),
            ("comp_H_jacobian", self.comp_H_jacobian,
             self.comp_H_jacobian_sparsity, self.n_comp),
            ("ineq_jacobian", self.ineq_jacobian,
             self.ineq_jacobian_sparsity, self.n_ineq),
            ("eq_jacobian", self.eq_jacobian,
             self.eq_jacobian_sparsity, self.n_eq),
        ]:
            if _sparsity is None or _fn is None:
                continue
            _rows = np.asarray(_sparsity[0])
            _cols = np.asarray(_sparsity[1])
            if _rows.ndim != 1 or _cols.ndim != 1 or len(_rows) != len(_cols):
                raise ValueError(
                    f"{_name}_sparsity: rows and cols must be 1-D arrays of equal length"
                )
            if len(_rows) == 0:
                raise ValueError(f"{_name}_sparsity must be non-empty")
            if np.any(_rows < 0) or np.any(_rows >= _n_rows):
                raise ValueError(
                    f"{_name}_sparsity: row indices out of range [0, {_n_rows})"
                )
            if np.any(_cols < 0) or np.any(_cols >= self.n):
                raise ValueError(
                    f"{_name}_sparsity: col indices out of range [0, {self.n})"
                )
            _keys = _rows.astype(np.intp, copy=False) * self.n + _cols.astype(np.intp, copy=False)
            if np.unique(_keys).size != _keys.size:
                raise ValueError(
                    f"{_name}_sparsity: duplicate (row, col) entries are not allowed"
                )
            _vals = np.asarray(_fn(x0))  # type: ignore[operator]
            if _vals.shape != (len(_rows),):
                raise ValueError(
                    f"{_name}(x0) with sparsity must return shape ({len(_rows)},),"
                    f" got {_vals.shape}"
                )
            if not np.all(np.isfinite(_vals)):
                raise ValueError(
                    f"{_name}(x0) returned non-finite sparse values (NaN or Inf)"
                )

        for _name in ("comp_G_scale", "comp_H_scale"):
            _scale = getattr(self, _name)
            if _scale is None:
                continue
            _arr = np.asarray(_scale, dtype=float)
            if _arr.shape != (self.n_comp,):
                raise ValueError(
                    f"{_name} must have shape ({self.n_comp},), got {_arr.shape}"
                )
            if not np.all(np.isfinite(_arr)):
                raise ValueError(f"{_name} contains non-finite values")
            if not np.all(_arr > 0.0):
                raise ValueError(f"{_name} must be strictly positive")
            setattr(self, _name, _arr)

    @staticmethod
    def _check_shape(name: str, fn: Callable, x0: np.ndarray,
                     expected: tuple) -> None:
        result = np.asarray(fn(x0))
        if result.shape != expected:
            raise ValueError(
                f"{name}(x0) must have shape {expected}, got {result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError(
                f"{name}(x0) returned non-finite values (NaN or Inf)"
            )
