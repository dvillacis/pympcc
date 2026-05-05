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

# Slack-lifting strategy

The `slack` strategy is a lifting reformulation: it introduces auxiliary variables $s_G = G(x)$ and $s_H = H(x)$ so the complementarity rows become $s_G \cdot s_H = 0$ — completely independent of $x$. The problem grows from $n$ to $n + 2 n_\text{comp}$ variables, but the Jacobian of every comp row has only **two** nonzeros (one in each slack block), regardless of how dense $\nabla G$ and $\nabla H$ are.

## When to reach for it

| Problem property | `slack` is a good fit when… |
|---|---|
| `n` is large (≳ 10⁴) | yes — sparse comp Jacobians fixed at `2·n_comp` nnz |
| `G` and `H` are **dense** functions of `x` | yes — moves the density out of the comp block |
| `n_comp` is small | maybe — the lifting overhead may not pay off |
| You need analytical Hessians | supports `lagrangian_hessian_slack` |

For sparse $G$, $H$ that already have lean Jacobians, `scholtes` typically wins on raw iterations. See the [strategy selection guide](../user_guide/strategy_selection.md).

## A small lifted solve

```{code-cell} ipython3
import warnings
import numpy as np
import pympcc

# 30-variable problem with 15 dense comp pairs:
# G_i(x) = sum_j x_j - target_G[i]      (every G_i depends on every x)
# H_i(x) = x[2i] + 0.5 * x[2i + 1]
n = 30
n_comp = 15
target_G = np.linspace(1.0, 2.0, n_comp)

def make_problem():
    return pympcc.MPCCProblem(
        n=n, n_comp=n_comp,
        x0=np.full(n, 0.5),
        xl=np.zeros(n),
        objective=lambda x: float(np.sum(x ** 2)),
        gradient=lambda x: 2.0 * x,
        comp_G=lambda x: np.full(n_comp, np.sum(x)) - target_G,
        comp_G_jacobian=lambda x: np.ones((n_comp, n)),    # dense!
        comp_H=lambda x: x[0::2][:n_comp] + 0.5 * x[1::2][:n_comp],
        comp_H_jacobian=lambda x: np.array(
            [[1.0 if j == 2 * i else (0.5 if j == 2 * i + 1 else 0.0)
              for j in range(n)] for i in range(n_comp)]
        ),
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r_scholtes = pympcc.solve(make_problem(), strategy="scholtes")
    r_slack    = pympcc.solve(make_problem(), strategy="slack")

print(f"  {'strategy':<10}  {'f*':>10}  {'comp_res':>10}  {'iters':>6}")
print("  " + "-" * 10 + "  " + "-" * 10 + "  " + "-" * 10 + "  " + "-" * 6)
print(f"  {'scholtes':<10}  {r_scholtes.obj:>10.4f}  "
      f"{r_scholtes.comp_residual:>10.2e}  {len(r_scholtes.history):>6}")
print(f"  {'slack':<10}  {r_slack.obj:>10.4f}  "
      f"{r_slack.comp_residual:>10.2e}  {len(r_slack.history):>6}")
```

Both reach the same optimum. The wall-clock timing is barely distinguishable at $n = 30$, but on production-sized problems the slack reformulation reduces per-callback Jacobian assembly from $\mathcal O(n_\text{comp} \cdot n)$ entries to $\mathcal O(n_\text{comp})$.

## Reading the lifted variables

The result is reported in the **original** variable space — `slack` strips the slack variables from `result.x` automatically, so user code never sees the lifting. If you need the slack values, they're recoverable via $s_G = G(x^\*)$:

```{code-cell} ipython3
print(f"x*    has length {r_slack.x.size}        (original n = {n})")
print(f"G(x*) = {r_slack.G[:3]} ...")
print(f"H(x*) = {r_slack.H[:3]} ...")
print(f"comp_residual = max |G_i · H_i| = {r_slack.comp_residual:.2e}")
```

## When `slack` underperforms

- **Small dense problems** ($n \lesssim 100$, $n_\text{comp} \lesssim 50$) — the lifting overhead exceeds the per-iteration savings.
- **Already-sparse `G`, `H`** — you've paid the cost of writing sparse Jacobians, so `scholtes` already gets the cheap callbacks.
- **Strict CQ requirements** — the lifting destroys MPCC-MFCQ; if you need the certified-multiplier story, prefer `lin_fukushima` or [TNLP refinement](tnlp_refinement.md).
