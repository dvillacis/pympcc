from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np


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
    gradient: Union[Callable[[np.ndarray], np.ndarray], str]
    comp_G: Callable[[np.ndarray], np.ndarray]
    comp_G_jacobian: Union[Callable[[np.ndarray], np.ndarray], str]
    comp_H: Callable[[np.ndarray], np.ndarray]
    comp_H_jacobian: Union[Callable[[np.ndarray], np.ndarray], str]

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
    # Optional sparse Jacobian structures (COO format, 0-based indices)   #
    # ------------------------------------------------------------------ #
    # When provided, the corresponding *_jacobian callable must return a  #
    # 1-D float array of nnz values instead of a dense 2-D matrix.       #
    comp_G_jacobian_sparsity: Optional[tuple] = None
    comp_H_jacobian_sparsity: Optional[tuple] = None
    ineq_jacobian_sparsity:   Optional[tuple] = None
    eq_jacobian_sparsity:     Optional[tuple] = None

    # ------------------------------------------------------------------ #

    @property
    def is_sparse(self) -> bool:
        """True if any Jacobian block has an explicit sparsity structure."""
        return any(s is not None for s in [
            self.comp_G_jacobian_sparsity, self.comp_H_jacobian_sparsity,
            self.ineq_jacobian_sparsity, self.eq_jacobian_sparsity,
        ])

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        if self.xl is None:
            self.xl = np.full(self.n, -np.inf)
        if self.xu is None:
            self.xu = np.full(self.n, np.inf)
        self.xl = np.asarray(self.xl, dtype=float)
        self.xu = np.asarray(self.xu, dtype=float)
        self._resolve_jax_fields()   # must run before fd so "jax" sentinels are cleared
        self._resolve_fd_fields()
        self._validate()

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
            self.comp_G_jacobian = fd_jacobian(
                self.comp_G, self.n_comp, self.n, h, mode
            )
            fd_used.append("comp_G_jacobian")
        if self.comp_H_jacobian == "fd":
            self.comp_H_jacobian = fd_jacobian(
                self.comp_H, self.n_comp, self.n, h, mode
            )
            fd_used.append("comp_H_jacobian")
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
        if x0.shape != (self.n,):
            raise ValueError(f"x0 must have shape ({self.n},), got {x0.shape}")
        if self.xl.shape != (self.n,):
            raise ValueError(f"xl must have shape ({self.n},)")
        if self.xu.shape != (self.n,):
            raise ValueError(f"xu must have shape ({self.n},)")
        if not np.all(self.xl <= self.xu):
            raise ValueError("xl must be <= xu element-wise")
        if self.n_comp < 1:
            raise ValueError("n_comp must be >= 1")

        self._check_shape("comp_G", self.comp_G, x0, (self.n_comp,))
        if self.comp_G_jacobian_sparsity is None:
            self._check_shape("comp_G_jacobian", self.comp_G_jacobian, x0,
                              (self.n_comp, self.n))
        self._check_shape("comp_H", self.comp_H, x0, (self.n_comp,))
        if self.comp_H_jacobian_sparsity is None:
            self._check_shape("comp_H_jacobian", self.comp_H_jacobian, x0,
                              (self.n_comp, self.n))

        if self.n_ineq > 0:
            if self.ineq_constraints is None or self.ineq_jacobian is None:
                raise ValueError(
                    "ineq_constraints and ineq_jacobian are required when n_ineq > 0"
                )
            self._check_shape("ineq_constraints", self.ineq_constraints, x0,
                               (self.n_ineq,))
            if self.ineq_jacobian_sparsity is None:
                self._check_shape("ineq_jacobian", self.ineq_jacobian, x0,
                                   (self.n_ineq, self.n))

        if self.n_eq > 0:
            if self.eq_constraints is None or self.eq_jacobian is None:
                raise ValueError(
                    "eq_constraints and eq_jacobian are required when n_eq > 0"
                )
            self._check_shape("eq_constraints", self.eq_constraints, x0,
                               (self.n_eq,))
            if self.eq_jacobian_sparsity is None:
                self._check_shape("eq_jacobian", self.eq_jacobian, x0,
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
            _vals = np.asarray(_fn(x0))
            if _vals.shape != (len(_rows),):
                raise ValueError(
                    f"{_name}(x0) with sparsity must return shape ({len(_rows)},),"
                    f" got {_vals.shape}"
                )

    @staticmethod
    def _check_shape(name: str, fn: Callable, x0: np.ndarray,
                     expected: tuple) -> None:
        result = np.asarray(fn(x0))
        if result.shape != expected:
            raise ValueError(
                f"{name}(x0) must have shape {expected}, got {result.shape}"
            )
