"""
StructuredMPCC — MPCC model with explicitly separated linear and nonlinear
constraint layers.

Most real-world MPCC problems contain a mix of constraint types:

* **Linear equalities** arise from KKT stationarity conditions, mass-balance
  equations, or network flow constraints.  Their Jacobian is a constant matrix,
  so there is no need to pass a callable.
* **Nonlinear equalities** capture equilibrium conditions, dynamics, or other
  relationships whose Jacobian varies with x.
* The same split applies to inequality constraints.

:class:`StructuredMPCC` lets users specify each layer separately and then
assembles them into a single :class:`~pympcc.problem.MPCCProblem` via
:meth:`to_mpcc_problem`, which is accepted directly by
:func:`~pympcc.solver.solve` and :class:`~pympcc.solver.MPCCSolver`.

Problem form::

    min  f(x)
    s.t. A_eq  @ x  = b_eq          (n_lin_eq  linear equalities)
         h_nl(x)    = 0             (n_nl_eq   nonlinear equalities)
         A_ineq @ x ≤ b_ineq        (n_lin_ineq linear inequalities)
         g_nl(x)   ≤ 0              (n_nl_ineq  nonlinear inequalities)
         G(x) >= 0, H(x) >= 0, G(x)^T H(x) = 0   (complementarity)
         xl <= x <= xu
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from .problem import MPCCProblem


@dataclass
class StructuredMPCC:
    """
    MPCC with explicitly separated linear and nonlinear constraint layers.

    Parameters
    ----------
    n : int
        Number of decision variables.
    n_comp : int
        Number of complementarity pairs  (>= 1).
    x0 : array-like, shape (n,)
        Initial guess.
    objective : callable
        ``f(x) -> float``
    gradient : callable
        ``∇f(x) -> ndarray, shape (n,)``
    comp_G : callable
        ``G(x) -> ndarray, shape (n_comp,)``  with ``G(x) >= 0``
    comp_G_jacobian : callable
        ``∇G(x) -> ndarray, shape (n_comp, n)``
    comp_H : callable
        ``H(x) -> ndarray, shape (n_comp,)``  with ``H(x) >= 0``
    comp_H_jacobian : callable
        ``∇H(x) -> ndarray, shape (n_comp, n)``
    xl : array-like, shape (n,), optional
        Lower bounds on x (default: ``-inf``).
    xu : array-like, shape (n,), optional
        Upper bounds on x (default: ``+inf``).

    Linear equalities  ``A_eq @ x = b_eq``
    ----------------------------------------
    A_eq : ndarray, shape (n_lin_eq, n), optional
    b_eq : ndarray, shape (n_lin_eq,),  optional
        Must be supplied together.

    Nonlinear equalities  ``h_nl(x) = 0``
    ----------------------------------------
    n_nl_eq : int
        Number of nonlinear equality constraints (default 0).
    eq_nl : callable, optional
        ``h_nl(x) -> ndarray, shape (n_nl_eq,)``
    jac_eq_nl : callable, optional
        ``∇h_nl(x) -> ndarray, shape (n_nl_eq, n)``

    Linear inequalities  ``A_ineq @ x ≤ b_ineq``
    -----------------------------------------------
    A_ineq : ndarray, shape (n_lin_ineq, n), optional
    b_ineq : ndarray, shape (n_lin_ineq,),  optional
        Must be supplied together.

    Nonlinear inequalities  ``g_nl(x) ≤ 0``
    ------------------------------------------
    n_nl_ineq : int
        Number of nonlinear inequality constraints (default 0).
    ineq_nl : callable, optional
        ``g_nl(x) -> ndarray, shape (n_nl_ineq,)``
    jac_ineq_nl : callable, optional
        ``∇g_nl(x) -> ndarray, shape (n_nl_ineq, n)``

    Notes
    -----
    All Jacobians (linear and nonlinear) must be dense 2-D arrays.
    Dimensions and callables shapes are validated at construction time by
    evaluating every callable at ``x0``.

    Examples
    --------
    >>> import numpy as np
    >>> import pympcc
    >>>
    >>> model = pympcc.StructuredMPCC(
    ...     n=3, n_comp=1,
    ...     x0=np.ones(3),
    ...     xl=np.zeros(3),
    ...     objective=lambda x: x[0]**2 + x[1]**2 + x[2]**2,
    ...     gradient=lambda x: 2*x,
    ...     # linear equality:  x[0] + x[1] = 1
    ...     A_eq=np.array([[1.0, 1.0, 0.0]]),
    ...     b_eq=np.array([1.0]),
    ...     # nonlinear equality:  x[0]^2 + x[2] = 1
    ...     n_nl_eq=1,
    ...     eq_nl=lambda x: np.array([x[0]**2 + x[2] - 1.0]),
    ...     jac_eq_nl=lambda x: np.array([[2*x[0], 0.0, 1.0]]),
    ...     # complementarity:  x[0] >= 0, x[1] >= 0, x[0]*x[1] = 0
    ...     comp_G=lambda x: np.array([x[0]]),
    ...     comp_G_jacobian=lambda x: np.array([[1.0, 0.0, 0.0]]),
    ...     comp_H=lambda x: np.array([x[1]]),
    ...     comp_H_jacobian=lambda x: np.array([[0.0, 1.0, 0.0]]),
    ... )
    >>> result = pympcc.solve(model)
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
    # Variable bounds                                                       #
    # ------------------------------------------------------------------ #
    xl: Optional[np.ndarray] = None
    xu: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Linear equalities  A_eq @ x = b_eq                                   #
    # ------------------------------------------------------------------ #
    A_eq: Optional[np.ndarray] = None
    b_eq: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Nonlinear equalities  h_nl(x) = 0                                    #
    # ------------------------------------------------------------------ #
    n_nl_eq: int = 0
    eq_nl: Optional[Callable[[np.ndarray], np.ndarray]] = None
    jac_eq_nl: Optional[Union[Callable[[np.ndarray], np.ndarray], str]] = None

    # ------------------------------------------------------------------ #
    # Linear inequalities  A_ineq @ x ≤ b_ineq                            #
    # ------------------------------------------------------------------ #
    A_ineq: Optional[np.ndarray] = None
    b_ineq: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Nonlinear inequalities  g_nl(x) ≤ 0                                  #
    # ------------------------------------------------------------------ #
    n_nl_ineq: int = 0
    ineq_nl: Optional[Callable[[np.ndarray], np.ndarray]] = None
    jac_ineq_nl: Optional[Union[Callable[[np.ndarray], np.ndarray], str]] = None

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

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        if self.xl is None:
            self.xl = np.full(self.n, -np.inf)
        if self.xu is None:
            self.xu = np.full(self.n, np.inf)
        self.xl = np.asarray(self.xl, dtype=float)
        self.xu = np.asarray(self.xu, dtype=float)
        if self.A_eq is not None:
            self.A_eq = np.asarray(self.A_eq, dtype=float)
        if self.b_eq is not None:
            self.b_eq = np.asarray(self.b_eq, dtype=float)
        if self.A_ineq is not None:
            self.A_ineq = np.asarray(self.A_ineq, dtype=float)
        if self.b_ineq is not None:
            self.b_ineq = np.asarray(self.b_ineq, dtype=float)
        self._resolve_jax_fields()
        self._resolve_fd_fields()
        self._validate()

    def _resolve_jax_fields(self) -> None:
        """Replace any ``"jax"`` sentinel with a JAX-autodiff callable."""
        from ._jax import HAS_JAX

        sentinel_values = [
            self.gradient,
            self.comp_G_jacobian, self.comp_H_jacobian,
            self.jac_eq_nl, self.jac_ineq_nl,
        ]
        jax_fields = [v for v in sentinel_values if v == "jax"]
        if not jax_fields:
            return

        if not HAS_JAX:
            raise ImportError(
                "JAX is required for the 'jax' sentinel but is not installed. "
                "Install it with:  pip install 'pympcc[jax]'"
            )

        from ._jax import jax_gradient, jax_jacobian_dense

        tol = self.jax_sparsity_tol
        jax_used: list[str] = []

        if self.gradient == "jax":
            self.gradient = jax_gradient(self.objective, self.n, self.x0, tol)
            jax_used.append("gradient")

        for attr, fn_attr, n_out in [
            ("comp_G_jacobian", "comp_G", self.n_comp),
            ("comp_H_jacobian", "comp_H", self.n_comp),
        ]:
            if getattr(self, attr) == "jax":
                fn = getattr(self, fn_attr)
                setattr(self, attr,
                        jax_jacobian_dense(fn, n_out, self.n, self.x0, tol))
                jax_used.append(attr)

        if self.jac_eq_nl == "jax":
            if self.n_nl_eq == 0 or self.eq_nl is None:
                raise ValueError(
                    "jac_eq_nl='jax' requires n_nl_eq > 0 and eq_nl"
                )
            self.jac_eq_nl = jax_jacobian_dense(
                self.eq_nl, self.n_nl_eq, self.n, self.x0, tol
            )
            jax_used.append("jac_eq_nl")

        if self.jac_ineq_nl == "jax":
            if self.n_nl_ineq == 0 or self.ineq_nl is None:
                raise ValueError(
                    "jac_ineq_nl='jax' requires n_nl_ineq > 0 and ineq_nl"
                )
            self.jac_ineq_nl = jax_jacobian_dense(
                self.ineq_nl, self.n_nl_ineq, self.n, self.x0, tol
            )
            jax_used.append("jac_ineq_nl")

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
        if self.jac_eq_nl == "fd":
            if self.n_nl_eq == 0 or self.eq_nl is None:
                raise ValueError(
                    "jac_eq_nl='fd' requires n_nl_eq > 0 and eq_nl"
                )
            self.jac_eq_nl = fd_jacobian(
                self.eq_nl, self.n_nl_eq, self.n, h, mode
            )
            fd_used.append("jac_eq_nl")
        if self.jac_ineq_nl == "fd":
            if self.n_nl_ineq == 0 or self.ineq_nl is None:
                raise ValueError(
                    "jac_ineq_nl='fd' requires n_nl_ineq > 0 and ineq_nl"
                )
            self.jac_ineq_nl = fd_jacobian(
                self.ineq_nl, self.n_nl_ineq, self.n, h, mode
            )
            fd_used.append("jac_ineq_nl")

        if fd_used:
            warnings.warn(
                f"Finite-difference Jacobian active for: {', '.join(fd_used)}. "
                "Suitable for prototyping; use exact Jacobians in production.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------ #
    # Derived dimensions                                                    #
    # ------------------------------------------------------------------ #

    @property
    def n_lin_eq(self) -> int:
        """Number of linear equality constraints (rows of ``A_eq``)."""
        return 0 if self.A_eq is None else int(self.A_eq.shape[0])

    @property
    def n_lin_ineq(self) -> int:
        """Number of linear inequality constraints (rows of ``A_ineq``)."""
        return 0 if self.A_ineq is None else int(self.A_ineq.shape[0])

    @property
    def n_eq(self) -> int:
        """Total equality constraints: linear + nonlinear."""
        return self.n_lin_eq + self.n_nl_eq

    @property
    def n_ineq(self) -> int:
        """Total inequality constraints: linear + nonlinear."""
        return self.n_lin_ineq + self.n_nl_ineq

    # ------------------------------------------------------------------ #
    # Validation                                                            #
    # ------------------------------------------------------------------ #

    def _validate(self) -> None:
        x0, n = self.x0, self.n

        if x0.shape != (n,):
            raise ValueError(f"x0 must have shape ({n},), got {x0.shape}")
        if self.xl.shape != (n,):
            raise ValueError(f"xl must have shape ({n},)")
        if self.xu.shape != (n,):
            raise ValueError(f"xu must have shape ({n},)")
        if not np.all(self.xl <= self.xu):
            raise ValueError("xl must be <= xu element-wise")
        if self.n_comp < 1:
            raise ValueError("n_comp must be >= 1")

        # Complementarity
        MPCCProblem._check_shape("comp_G", self.comp_G, x0, (self.n_comp,))
        MPCCProblem._check_shape("comp_G_jacobian", self.comp_G_jacobian, x0,
                                 (self.n_comp, n))
        MPCCProblem._check_shape("comp_H", self.comp_H, x0, (self.n_comp,))
        MPCCProblem._check_shape("comp_H_jacobian", self.comp_H_jacobian, x0,
                                 (self.n_comp, n))

        # Linear equalities
        if (self.A_eq is None) != (self.b_eq is None):
            raise ValueError("A_eq and b_eq must both be provided or both be None")
        if self.A_eq is not None:
            if self.A_eq.ndim != 2 or self.A_eq.shape[1] != n:
                raise ValueError(
                    f"A_eq must have shape (m, {n}), got {self.A_eq.shape}"
                )
            if self.b_eq.shape != (self.n_lin_eq,):
                raise ValueError(
                    f"b_eq must have shape ({self.n_lin_eq},), got {self.b_eq.shape}"
                )

        # Nonlinear equalities
        if self.n_nl_eq > 0:
            if self.eq_nl is None or self.jac_eq_nl is None:
                raise ValueError(
                    "eq_nl and jac_eq_nl are required when n_nl_eq > 0"
                )
            MPCCProblem._check_shape("eq_nl", self.eq_nl, x0, (self.n_nl_eq,))
            MPCCProblem._check_shape("jac_eq_nl", self.jac_eq_nl, x0,
                                     (self.n_nl_eq, n))

        # Linear inequalities
        if (self.A_ineq is None) != (self.b_ineq is None):
            raise ValueError(
                "A_ineq and b_ineq must both be provided or both be None"
            )
        if self.A_ineq is not None:
            if self.A_ineq.ndim != 2 or self.A_ineq.shape[1] != n:
                raise ValueError(
                    f"A_ineq must have shape (m, {n}), got {self.A_ineq.shape}"
                )
            if self.b_ineq.shape != (self.n_lin_ineq,):
                raise ValueError(
                    f"b_ineq must have shape ({self.n_lin_ineq},), "
                    f"got {self.b_ineq.shape}"
                )

        # Nonlinear inequalities
        if self.n_nl_ineq > 0:
            if self.ineq_nl is None or self.jac_ineq_nl is None:
                raise ValueError(
                    "ineq_nl and jac_ineq_nl are required when n_nl_ineq > 0"
                )
            MPCCProblem._check_shape("ineq_nl", self.ineq_nl, x0,
                                     (self.n_nl_ineq,))
            MPCCProblem._check_shape("jac_ineq_nl", self.jac_ineq_nl, x0,
                                     (self.n_nl_ineq, n))

    # ------------------------------------------------------------------ #
    # Conversion                                                            #
    # ------------------------------------------------------------------ #

    def to_mpcc_problem(self) -> MPCCProblem:
        """
        Assemble a :class:`~pympcc.problem.MPCCProblem` from the structured
        constraint layers.

        The linear and nonlinear equality (inequality) blocks are stacked in
        order — linear first, nonlinear second — into a single callable pair
        suitable for :class:`~pympcc.problem.MPCCProblem`.

        Returns
        -------
        MPCCProblem
        """
        eq_fn, jac_eq_fn = self._build_eq_callables()
        ineq_fn, jac_ineq_fn = self._build_ineq_callables()

        return MPCCProblem(
            n=self.n,
            n_comp=self.n_comp,
            x0=self.x0.copy(),
            xl=self.xl.copy(),
            xu=self.xu.copy(),
            objective=self.objective,
            gradient=self.gradient,
            comp_G=self.comp_G,
            comp_G_jacobian=self.comp_G_jacobian,
            comp_H=self.comp_H,
            comp_H_jacobian=self.comp_H_jacobian,
            n_eq=self.n_eq,
            eq_constraints=eq_fn,
            eq_jacobian=jac_eq_fn,
            n_ineq=self.n_ineq,
            ineq_constraints=ineq_fn,
            ineq_jacobian=jac_ineq_fn,
            use_jax_hessian=self.use_jax_hessian,
            jax_sparsity_tol=self.jax_sparsity_tol,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_eq_callables(self):
        """Build combined equality constraint callable and Jacobian callable."""
        n_lin = self.n_lin_eq
        n_nl  = self.n_nl_eq

        if self.n_eq == 0:
            return None, None

        A = self.A_eq          # captured by closure (may be None if n_lin=0)
        b = self.b_eq
        h_nl    = self.eq_nl
        jac_h_nl = self.jac_eq_nl

        def eq_constraints(x: np.ndarray) -> np.ndarray:
            parts = []
            if n_lin > 0:
                parts.append(A @ x - b)
            if n_nl > 0:
                parts.append(np.asarray(h_nl(x)))
            return np.concatenate(parts)

        def eq_jacobian(x: np.ndarray) -> np.ndarray:
            rows = []
            if n_lin > 0:
                rows.append(A)                          # constant Jacobian
            if n_nl > 0:
                rows.append(np.asarray(jac_h_nl(x)))   # varying Jacobian
            return np.vstack(rows)

        return eq_constraints, eq_jacobian

    def _build_ineq_callables(self):
        """Build combined inequality constraint callable and Jacobian callable."""
        n_lin = self.n_lin_ineq
        n_nl  = self.n_nl_ineq

        if self.n_ineq == 0:
            return None, None

        A = self.A_ineq
        b = self.b_ineq
        g_nl    = self.ineq_nl
        jac_g_nl = self.jac_ineq_nl

        def ineq_constraints(x: np.ndarray) -> np.ndarray:
            parts = []
            if n_lin > 0:
                parts.append(A @ x - b)
            if n_nl > 0:
                parts.append(np.asarray(g_nl(x)))
            return np.concatenate(parts)

        def ineq_jacobian(x: np.ndarray) -> np.ndarray:
            rows = []
            if n_lin > 0:
                rows.append(A)
            if n_nl > 0:
                rows.append(np.asarray(jac_g_nl(x)))
            return np.vstack(rows)

        return ineq_constraints, ineq_jacobian
