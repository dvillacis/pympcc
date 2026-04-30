# Quickstart

A 15-line MPCC: minimise $(x_0-2)^2 + (x_1-1)^2$ subject to $x_0 \ge 0 \perp x_1 \ge 0$.

```python
import numpy as np
import pympcc

problem = pympcc.MPCCProblem(
    n=2, n_comp=1,
    x0=np.array([0.5, 0.5]),
    xl=np.zeros(2),
    objective=lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
    gradient=lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
    comp_G=lambda x: np.array([x[0]]),
    comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
    comp_H=lambda x: np.array([x[1]]),
    comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
)

result = pympcc.solve(problem, strategy="scholtes")
print(result.x, result.obj, result.success)
```

The optimum is $x^* = (2, 0)$, $f^* = 1$.

## Where to next

- [Problem setup](user_guide/problem_setup.md) — `MPCCProblem`, `StructuredMPCC`, variable-paired complementarity, derivative shorthand.
- [Strategies](user_guide/strategies.md) — when to pick each of the six reformulations.
- [Diagnostics](user_guide/diagnostics.md) — CQ classification, B-stationarity, SOSC, TNLP-certified multipliers.
- [Differentiable solve](user_guide/autodiff.md) — `jax.grad` through a converged MPCC.
