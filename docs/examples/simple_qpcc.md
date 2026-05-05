---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# A first MPCC, walked through

The quickstart problem unpacked. We solve

$$
\min_{x \in \RR^2}\ (x_0 - 2)^2 + (x_1 - 1)^2 \quad\text{s.t.}\quad x_0 \ge 0\ \perp\ x_1 \ge 0.
$$

The complementarity says "at most one of $x_0$, $x_1$ is positive at the optimum." Since the unconstrained minimum is at $(2, 1)$ — both positive — the complementarity will force one of the two to zero. Pulling $x_1 \to 0$ costs $(1-0)^2 = 1$; pulling $x_0 \to 0$ costs $(0-2)^2 = 4$. The optimum is therefore $x^\* = (2, 0)$ with $f^\* = 1$.

```{code-cell} ipython3
import warnings
import numpy as np
import pympcc

problem = pympcc.MPCCProblem(
    n=2, n_comp=1,
    x0=np.array([0.5, 0.5]),
    xl=np.zeros(2),
    objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
    gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
    comp_G=lambda x: np.array([x[0]]),
    comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
    comp_H=lambda x: np.array([x[1]]),
    comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    result = pympcc.solve(problem, strategy="scholtes")

print(f"x* = {result.x}")
print(f"f* = {result.obj:.6f}    (reference 1)")
print(f"success = {result.success}")
```

## Reading the result

Every solve returns an `MPCCResult` dataclass. The fields you'll touch most often:

```{code-cell} ipython3
print(f"x                = {result.x}")
print(f"obj              = {result.obj:.6e}")
print(f"comp_residual    = {result.comp_residual:.2e}    # max(G_i · H_i)")
print(f"kkt_residual     = {result.kkt_residual:.2e}    # KKT optimality residual")
print(f"stationarity     = {result.stationarity}")
print(f"per_pair_status  = {result.per_pair_status}")
print(f"n_outer          = {len(result.history)}")
```

- `comp_residual` confirms the complementarity is satisfied: $G_i \cdot H_i \approx 0$.
- `kkt_residual` is the MPCC-KKT optimality residual; small means we landed at a stationary point.
- `per_pair_status` classifies each pair: `"G_active"` (here $x_1 = 0$ so $H = 0$), `"H_active"`, `"biactive"` (both zero), or `"inactive"`.
- `stationarity` is the strongest CQ-respecting label the solver can certify; for this problem we get `S-stationary`.

## Per-pair table

For larger problems with many comp pairs, `result.to_dataframe()` returns a per-pair view (requires pandas):

```{code-cell} ipython3
try:
    df = result.to_dataframe()
    print(df.to_string(index=False))
except ImportError:
    print("pandas not installed — install pympcc[all] for to_dataframe support")
```

## Where to next

- [Tour of strategies](strategies_tour.md) — the same problem solved by all six canonical reformulations.
- [KKT residual](kkt_residual.md) — interpreting the optimality residual.
- [Stationarity hierarchy](stationarity_hierarchy.md) — S / M / C / W in one notebook.
