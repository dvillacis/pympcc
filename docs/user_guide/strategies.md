# Strategies

Thirteen NLP-based reformulation strategies are available — six canonical plus seven NCP-function variants. Each transforms the MPCC into one or more standard NLPs solved with IPOPT. See the [selection guide](strategy_selection.md) for help picking the right one.

## Canonical strategies

| Strategy | How it works | Loop | Best for |
|---|---|---|---|
| `"direct"` | Replaces $G \cdot H = 0$ with $G \cdot H \le 0$; single solve | No | Quick feasibility checks |
| `"scholtes"` | Relaxes to $G \cdot H \le \varepsilon$, drives $\varepsilon \to 0$ (Scholtes 2001) | Yes | **Default — most robust** |
| `"smoothing"` | Fischer-Burmeister equation $\varphi_\varepsilon(G,H) = 0$, drives $\varepsilon \to 0$ | Yes | Tight complementarity residuals |
| `"lin_fukushima"` | $G \cdot H \le \varepsilon$ and $G+H \ge \varepsilon$; guarantees MPCC-MFCQ (Lin & Fukushima 2003) | Yes | Degenerate problems |
| `"augmented_lagrangian"` | PHR penalty in the objective; complementarity never enters the NLP constraints | Yes | Problems where MFCQ fails |
| `"slack"` | Lifts $G$, $H$ to slack variables; complementarity rows have zero $x$-entries | Yes | Large-$n$ problems ($n \gg n_\text{comp}$) |

### Practical advice

- `"scholtes"` is the safest default.
- `"smoothing"` often achieves tighter complementarity residuals on smooth problems.
- `"lin_fukushima"` is more robust on degenerate problems where Scholtes stalls.
- `"slack"` is the best choice when $n$ is large (hundreds to thousands) and $n_\text{comp}$ is small — the Jacobian of the complementarity block is $O(n_\text{comp})$ rather than $O(n_\text{comp} \cdot n)$.

## NCP-function variants

Seven smoothed-NCP-function strategies share the same ε-continuation harness as `"smoothing"` but swap in different $\varphi_\varepsilon$ functions.  Each is a stand-alone strategy name (e.g. `pympcc.solve(problem, strategy="chen_chen_kanzow", lam=0.7)`) and is also reachable through the unified `strategy="ncp"` entry point with the matching `ncp_function=...` key.

| Strategy | $\varphi_\varepsilon(G, H)$ | Param | When to reach for it |
|---|---|---|---|
| `"smooth_min"` | $\tfrac{1}{2}(G + H - \sqrt{(G-H)^2 + 4\varepsilon^2})$ | — | Symmetric smoothing of $\min(G, H)$. Stable when $G$ and $H$ have similar scales. |
| `"chen_chen_kanzow"` | $\lambda\,\varphi_{\mathrm{FB},\varepsilon} + (1-\lambda)\,G H$ | `lam ∈ (0,1]` | Convex combination of Fischer-Burmeister and the inner-product penalty. ``lam → 1`` = pure smoothing; lower ``lam`` adds direct comp-pressure. |
| `"kanzow_schwartz"` | $G + H - \sqrt{G^2 + H^2 + 2\lambda G H + \varepsilon^2}$ | `lam ∈ [0,1)` | Modified FB; `lam → 1` approaches smoothed $\lvert G + H \rvert$. |
| `"chen_mangasarian"` | $G + H - \sqrt{G^2 + H^2 - 2\alpha G H + \varepsilon^2}$ | `alpha ∈ [0,1]` | FB↔min interpolation. `alpha=0` = FB, `alpha=1` = $2\min(G,H)$. Matches NLPEC's `CMxf` / `CMfx`. |
| `"billups"` | $\varphi_{\mathrm{FB},\varepsilon} - \gamma G_{+\varepsilon} H_{+\varepsilon}$ | `gamma ≥ 0` | FB plus a positive-part penalty that pulls iterates harder onto the complementary cone.  Helps when FB stalls just inside the cone. |
| `"veelken_ulbrich_pow"` | $\tfrac{1}{2}(G + H - \sigma_\varepsilon^{\mathrm{pow}}(G - H))$ | — | Smooth-min with a $C^2$ piecewise-polynomial smoothing of $\lvert\cdot\rvert$.  Good for Newton-type solvers that benefit from continuous second derivatives. |
| `"veelken_ulbrich_sin"` | $\tfrac{1}{2}(G + H - (2t/\pi)\arctan(\pi t / 2\varepsilon))$, $t = G - H$ | — | Smooth-min with a $C^\infty$ arctan smoothing.  Vanishes exactly at $G = H = 0$, which can help second-order solvers. |

### Practical advice for NCP variants

- All seven share the ε-continuation knobs (`epsilon_0`, `reduction`, `max_iter`, `epsilon_min`, `comp_tol`, `dual_warmstart`) with `"smoothing"`.
- The Veelken-Ulbrich pair is the natural choice when the user-supplied derivatives are smooth enough that the solver can exploit second-order information (`use_jax_hessian=True` on the problem).
- `"chen_mangasarian"` with `alpha=0.5` is a good FB-replacement on problems where FB hits restoration; the asymmetric radicand often resolves the local stall.
- `"billups"` with `gamma=0.05`-`0.1` adds a gentle pull onto the complementary cone — useful when the solution shows non-zero comp residual that FB can't drive below ~$10^{-6}$.

## Unified NCP-function entry point

```python
result = pympcc.solve(
    problem,
    strategy="ncp",
    ncp_function="chen_chen_kanzow",   # any registry key from the table above
    ncp_params={"lam": 0.7},           # variant-specific overrides
    epsilon_0=1.0, reduction=0.1,
)
```

Equivalent to `pympcc.solve(problem, strategy="chen_chen_kanzow", lam=0.7)`. The `strategy="ncp"` entry point is preferred when you want to sweep over multiple NCP functions without changing strategy names.

## `solve` options

```python
result = pympcc.solve(
    problem,
    strategy="scholtes",
    ipopt_options=None,        # dict of IPOPT options, e.g. {"max_iter": 500}

    # Iterative strategy options:
    epsilon_0=1.0,
    reduction=0.1,
    max_iter=20,
    epsilon_min=1e-8,
    dual_warmstart=True,

    # augmented_lagrangian-specific:
    rho_0=10.0, rho_max=1e6, tau=10.0, eta=0.25,
    comp_tol=1e-8,

    # Diagnostics:
    diagnostics=False,
    b_stat_max_biactive=10,

    # TNLP certified multiplier refinement:
    tnlp_refine=False,
    tnlp_max_iter=500,

    # Presolve:
    presolve=False,

    # Wall-clock limit:
    time_limit=None,
)
```

## `MPCCResult`

| Field | Type | Description |
|---|---|---|
| `x` | `(n,)` | Solution vector |
| `obj` | `float` | Objective value `f(x)` |
| `G`, `H` | `(n_comp,)` | Complementarity values |
| `comp_residual` | `float` | $\max_i \lvert G_i \cdot H_i \rvert$ |
| `success` | `bool` | IPOPT converged (status 0, 1, or 3) |
| `status` | `int` | Raw IPOPT exit code |
| `strategy` | `str` | Strategy name used |
| `stationarity` | `str` | `"S-stationary"`, `"W-stationary"`, or `"unknown"` |
| `kkt_residual` | `float \| None` | Stationarity residual |
| `history` | `list[IterationInfo]` | Per-outer-iteration diagnostics |
| `per_pair_status` | `list[str]` | `"G_active"` / `"H_active"` / `"biactive"` / `"inactive"` |
| `cq` | `str \| None` | `"MPCC-LICQ"` / `"MPCC-MFCQ"` / `"none"` (requires `diagnostics=True`) |
| `b_stationary` | `str \| None` | B-stationarity verdict (requires `diagnostics=True`) |
| `sosc` | `bool \| None` | SOSC verdict (requires `diagnostics=True`) |
| `mult_comp_G_mpcc`, `mult_comp_H_mpcc` | `(n_comp,) \| None` | Certified MPCC multipliers (requires `tnlp_refine=True`) |
| `tnlp_refined` | `TNLPResult \| None` | Full TNLP sub-result |
| `jac_condition`, `hessian_condition_estimate` | `float \| None` | Condition diagnostics |
| `time_limit_hit` | `bool` | Wall-clock budget exhausted |

## Iteration history

```python
result = pympcc.solve(problem, strategy="scholtes", max_iter=10)
for it in result.history:
    print(f"ε={it.epsilon:.2e}  obj={it.obj:.6f}  comp={it.comp_residual:.2e}")
```

## JSON / DataFrame export

```python
result.to_json()         # full result, JSON-encoded
result.to_dataframe()    # per-pair table (requires pandas)
```
