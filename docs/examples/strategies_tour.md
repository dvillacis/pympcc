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

# Tour of strategies

The same MPCC, solved by all six canonical reformulations. The package ships **six** strategies (plus seven NCP-function variants documented in [NCP variants](ncp_variants_tour.md)):

| Strategy | One-line description |
|---|---|
| `direct` | Single IPOPT solve with `G·H ≤ 0`. Fast but LICQ generically fails. |
| `scholtes` | Sequential `G·H ≤ ε` with `ε → 0`. The textbook MPCC default. |
| `smoothing` | Fischer-Burmeister smoothing $\varphi_\varepsilon(G, H) = 0$. |
| `lin_fukushima` | `G·H ≤ ε` and `G + H ≥ ε`. Guarantees MPCC-MFCQ. |
| `augmented_lagrangian` | PHR penalty on `G·H` in the objective. |
| `slack` | Lifts $s_G = G(x)$, $s_H = H(x)$. Best for large `n`. |

For when to pick each, see the [strategy selection guide](../user_guide/strategy_selection.md).

```{code-cell} ipython3
import warnings
import numpy as np
import pympcc

def make_problem():
    return pympcc.MPCCProblem(
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

strategies = ["direct", "scholtes", "smoothing", "lin_fukushima",
              "augmented_lagrangian", "slack"]

print(f"  {'strategy':<22}  {'f*':>8}  {'comp_res':>10}  {'kkt_res':>10}  status")
print("  " + "-" * 22 + "  " + "-" * 8 + "  " + "-" * 10 + "  " + "-" * 10 + "  " + "-" * 14)
for s in strategies:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = pympcc.solve(make_problem(), strategy=s)
    print(
        f"  {s:<22}  {r.obj:>8.4f}  "
        f"{r.comp_residual:>10.2e}  {r.kkt_residual:>10.2e}  "
        f"{r.stationarity}"
    )
```

All six converge to $f^\* = 1$ at $x^\* = (2, 0)$. They differ in:

- **Iteration count** — `direct` is one IPOPT solve; `scholtes` / `smoothing` / `lin_fukushima` run an outer ε-loop; `augmented_lagrangian` runs an outer ρ-escalation loop.
- **CQ guarantees** — `direct` typically violates LICQ at the optimum (still converges in practice but multipliers may be poorly defined); `scholtes` and `lin_fukushima` are MFCQ-respecting in the limit.
- **Per-iteration history** — `result.history` records every outer iterate for the iterative strategies.

```{code-cell} ipython3
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r = pympcc.solve(make_problem(), strategy="scholtes")

print(f"Scholtes ran {len(r.history)} outer iterations")
for k, info in enumerate(r.history):
    print(f"  iter {k}: ε = {info.epsilon:.2e}    "
          f"obj = {info.obj:.6f}    "
          f"comp_res = {info.comp_residual:.2e}")
```

The ε schedule shrinks geometrically toward zero; each outer solve warm-starts from the previous primal-dual point.
