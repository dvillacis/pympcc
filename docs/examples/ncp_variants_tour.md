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

# NCP-function variants

In addition to the six canonical strategies, `pympcc` ships seven **smooth NCP-function** reformulations. An NCP function $\varphi(a, b)$ has the property $\varphi(a, b) = 0 \iff a \ge 0,\ b \ge 0,\ ab = 0$. Replacing the complementarity row with $\varphi_\varepsilon(G, H) = 0$ for a smoothed $\varphi_\varepsilon$ gives an equality constraint that vanishes the textbook degeneracy.

| Strategy | Formula | When to try it |
|---|---|---|
| `smoothing` (Fischer-Burmeister) | $\sqrt{G^2 + H^2 + \varepsilon^2} - G - H = 0$ | Default smooth alternative to scholtes. |
| `smooth_min` | $\frac12(G + H - \sqrt{(G - H)^2 + 4\varepsilon^2}) = 0$ | When FB stalls — symmetric & simpler. |
| `chen_chen_kanzow` | $\lambda \cdot \varphi_\text{FB} + (1-\lambda) \cdot G \cdot H = 0$ | Convex combination; tune `lam` ∈ (0, 1]. |
| `kanzow_schwartz` | $G + H - \sqrt{G^2 + H^2 + 2\lambda G H + \varepsilon^2} = 0$ | One-parameter FB family. |
| `chen_mangasarian` | $-\frac{1}{\varepsilon} \log(e^{-\varepsilon G} + e^{-\varepsilon H}) = 0$ | Smooth log-sum-exp; high-dynamic-range. |
| `billups` | $(G + H)^2 - G^2 - H^2 - \varepsilon^2 = 0$ | Quadratic-time smoothing; Hessian-friendly. |
| `veelken_ulbrich_pow` | Power-smoothed min-map | Robust on ill-scaled biactive sets. |
| `veelken_ulbrich_sin` | Trigonometric-smoothed min-map | Same, alternative shape. |

For the formulas in full, see the [strategies user guide](../user_guide/strategies.md).

## All variants on the quickstart problem

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

variants = [
    "smoothing",            # Fischer-Burmeister (canonical)
    "smooth_min",
    "chen_chen_kanzow",
    "kanzow_schwartz",
    "chen_mangasarian",
    "billups",
    "veelken_ulbrich_pow",
    "veelken_ulbrich_sin",
]

print(f"  {'variant':<22}  {'f*':>8}  {'comp_res':>10}  {'kkt_res':>10}  iters")
print("  " + "-" * 22 + "  " + "-" * 8 + "  " + "-" * 10 + "  " + "-" * 10 + "  -----")
for v in variants:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = pympcc.solve(make_problem(), strategy=v)
    print(f"  {v:<22}  {r.obj:>8.4f}  "
          f"{r.comp_residual:>10.2e}  {r.kkt_residual:>10.2e}  "
          f"{len(r.history):>5d}")
```

All eight smoothings reach the same $f^\* = 1$. They differ in:

- **Iteration count** — different ε-shapes have different conditioning at intermediate ε.
- **Robustness on biactive sets** — `chen_mangasarian` and `veelken_ulbrich_*` are designed to be more forgiving when $I_{00}$ is large.
- **Tunable parameters** — `chen_chen_kanzow` exposes `lam`, `kanzow_schwartz` exposes `lam`, the rest only ε-related options.

## Tuning a parameterised variant

```{code-cell} ipython3
print(f"  {'lam':<5}  {'iters':>6}  {'f*':>10}  {'kkt_res':>10}")
print("  " + "-" * 5 + "  " + "-" * 6 + "  " + "-" * 10 + "  " + "-" * 10)
for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = pympcc.solve(make_problem(), strategy="chen_chen_kanzow", lam=lam)
    print(f"  {lam:<5.2f}  {len(r.history):>6}  "
          f"{r.obj:>10.4f}  {r.kkt_residual:>10.2e}")
```

`lam = 1.0` is pure Fischer-Burmeister; `lam → 0` shifts toward the inner-product formulation (`G·H = 0`). The default 0.5 is a reasonable middle ground.

## Picking a variant

If `smoothing` (FB) works for your problem, stop there — it's the most-studied variant. Reach for the others when:

- **FB diverges or oscillates near biactive sets** → try `smooth_min` or `chen_mangasarian`.
- **You're already biactive-heavy and need robust convergence** → `veelken_ulbrich_pow` is the most aggressive smoother.
- **You want explicit interpolation between smoothing and inner-product penalisation** → `chen_chen_kanzow` with a small `lam` recovers a near-direct strategy.
