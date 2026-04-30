# Strategies

Six NLP-based reformulation strategies are available. Each transforms the MPCC into one or more standard NLPs solved with IPOPT.

| Strategy | How it works | Loop | Best for |
|---|---|---|---|
| `"direct"` | Replaces $G \cdot H = 0$ with $G \cdot H \le 0$; single solve | No | Quick feasibility checks |
| `"scholtes"` | Relaxes to $G \cdot H \le \varepsilon$, drives $\varepsilon \to 0$ (Scholtes 2001) | Yes | **Default — most robust** |
| `"smoothing"` | Fischer-Burmeister equation $\varphi_\varepsilon(G,H) = 0$, drives $\varepsilon \to 0$ | Yes | Tight complementarity residuals |
| `"lin_fukushima"` | $G \cdot H \le \varepsilon$ and $G+H \ge \varepsilon$; guarantees MPCC-MFCQ (Lin & Fukushima 2003) | Yes | Degenerate problems |
| `"augmented_lagrangian"` | PHR penalty in the objective; complementarity never enters the NLP constraints | Yes | Problems where MFCQ fails |
| `"slack"` | Lifts $G$, $H$ to slack variables; complementarity rows have zero $x$-entries | Yes | Large-$n$ problems ($n \gg n_\text{comp}$) |

## Practical advice

- `"scholtes"` is the safest default.
- `"smoothing"` often achieves tighter complementarity residuals on smooth problems.
- `"lin_fukushima"` is more robust on degenerate problems where Scholtes stalls.
- `"slack"` is the best choice when $n$ is large (hundreds to thousands) and $n_\text{comp}$ is small — the Jacobian of the complementarity block is $O(n_\text{comp})$ rather than $O(n_\text{comp} \cdot n)$.

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
