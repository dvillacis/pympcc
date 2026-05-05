from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from ._constants import SPARSITY_TOL as _SPARSITY_TOL
from ._problem_derivatives import (
    apply_derivatives_default as _apply_derivatives_default_fn,
)
from ._problem_derivatives import (
    check_derivatives_resolved as _check_derivatives_resolved_fn,
)
from ._problem_derivatives import (
    resolve_fd_fields as _resolve_fd_fields_fn,
)
from ._problem_derivatives import (
    resolve_jax_fields as _resolve_jax_fields_fn,
)
from ._problem_reduction import (
    normalize_box_pairs as _normalize_box_pairs_fn,
)
from ._problem_reduction import (
    normalize_var_pairs as _normalize_var_pairs_fn,
)
from ._typing import FDMode

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
    fd_mode: FDMode = "forward"

    # ------------------------------------------------------------------ #
    # JAX autodiff options                                                 #
    # ------------------------------------------------------------------ #
    use_jax_hessian: bool = False
    jax_sparsity_tol: float = _SPARSITY_TOL

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
    # Bulk variable-paired complementarity (MCP form, vectorized).       #
    # Use this for large MCPs (k >= 1e3) to avoid Python per-row dispatch.#
    # 4-tuple: (var_idxs, h_bulk_fn, h_bulk_jac_fn, h_bulk_jac_sparsity) #
    #   var_idxs           — ndarray (k,) of int; declares                #
    #                        x[var_idxs[i]] >= 0  ⊥  h_bulk_fn(x)[i] >= 0 #
    #   h_bulk_fn          — x -> ndarray (k,); single vectorized call.   #
    #   h_bulk_jac_fn      — x -> ndarray (nnz,); flat sparse values.     #
    #   h_bulk_jac_sparsity — (rows, cols) of the H block, k rows total.  #
    # Mutually exclusive with comp_var_pairs. xl[var_idxs] clamped to 0.  #
    # ------------------------------------------------------------------ #
    comp_var_pairs_bulk: Optional[tuple] = None

    # ------------------------------------------------------------------ #
    # Box-MCP / doubly-bounded complementarity (PATH / AMPL convention). #
    # Each entry declares                                                  #
    #     xl[var_idx] <= x[var_idx] <= xu[var_idx]   ⊥   F_fn(x)          #
    # and is dispatched by bound finiteness:                              #
    #   * lower-only (xl finite, xu = +inf) → comp pair                   #
    #         (x[var_idx] - xl) >= 0  ⊥  F_fn(x) >= 0                     #
    #   * upper-only (xl = -inf, xu finite) → comp pair                   #
    #         (xu - x[var_idx]) >= 0  ⊥  -F_fn(x) >= 0                    #
    #   * free (both infinite) → equality F_fn(x) = 0 appended to eq      #
    #   * doubly-bounded (both finite) → NotImplementedError (§4.9 Phase 2)#
    # Two grammars:                                                        #
    #   (var_idx, F_fn)                — F Jacobian via forward fd         #
    #   (var_idx, F_fn, F_jac_fn)      — user-supplied dense Jacobian row  #
    # n_comp / n_eq are auto-bumped to include the synthesized rows;       #
    # the user supplies n_comp / n_eq for the *base* problem only.         #
    # Mutually exclusive with comp_var_pairs and comp_var_pairs_bulk.      #
    # ------------------------------------------------------------------ #
    comp_box_pairs: Optional[list] = None

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
        _apply_derivatives_default_fn(self)
        # jax/fd resolution skips comp_G/H Jacobians when comp_G is None
        # (they will be built by normalize_var_pairs from the var-pair declarations).
        _resolve_jax_fields_fn(self)
        _resolve_fd_fields_fn(self)
        if self.comp_box_pairs and (self.comp_var_pairs or self.comp_var_pairs_bulk):
            raise ValueError(
                "comp_box_pairs cannot be combined with comp_var_pairs or "
                "comp_var_pairs_bulk in this release"
            )
        _normalize_var_pairs_fn(self)   # merges var-pairs into comp_G/H and builds Jacobians
        _normalize_box_pairs_fn(self)   # appends box-pair rows to comp_G/H or eq block
        _check_derivatives_resolved_fn(self)
        self._validate()

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
        if not np.all(np.isfinite(self.gradient(x0))):  # type: ignore[misc, operator]
            raise ValueError("gradient(x0) returned non-finite values (NaN or Inf)")

        self._check_shape("comp_G", self.comp_G, x0, (self.n_comp,))  # type: ignore[arg-type]
        if self.comp_G_jacobian_sparsity is None:
            self._check_shape("comp_G_jacobian", self.comp_G_jacobian, x0,  # type: ignore[arg-type]
                              (self.n_comp, self.n))
        self._check_shape("comp_H", self.comp_H, x0, (self.n_comp,))  # type: ignore[arg-type]
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
