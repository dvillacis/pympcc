# Troubleshooting

A symptom-to-action map for the most common failure modes.  See
[strategy selection](strategy_selection.md) when the issue is
"converged to the wrong stationary point" rather than "didn't
converge".

## IPOPT divergence / restoration failure

### Symptom: `result.success is False` and `result.status == 2`

IPOPT's restoration phase failed.  This is the catch-all "couldn't
make progress" code; pympcc surfaces a per-iteration counter that
distinguishes the two underlying causes:

```python
for it in result.history:
    print(f"iter {it.epsilon:.2e}  restoration={it.restoration_iter_count}"
          f"  entered={it.entered_restoration}")
```

| `restoration_iter_count` per inner solve | Likely cause | Action |
|---|---|---|
| `0` consistently, fails at first inner | infeasible at $x_0$ | Provide a strictly feasible starting point |
| Climbs to ≥ 5 across consecutive inner solves | ε is too tight for local geometry | Switch to `strategy="lin_fukushima"` (MFCQ-friendly bracket) or pass `safeguards="all"` to enable rollback |
| Spikes once, recovers | transient — IPOPT found its way out | No action; the safeguard auto-rolled |

### Symptom: `result.success is True` but `comp_residual > 1e-4`

IPOPT terminated cleanly but the relaxation never drove
$G \cdot H$ all the way to zero.  Two likely causes:

* **ε floor too loose.** Try `epsilon_min=1e-10` (default `1e-8`).
* **MFCQ degenerate at the solution.** Switch to
  `strategy="augmented_lagrangian"`, which never makes the comp
  product a constraint in the first place.

If the result is still loose, run with `tnlp_refine=True` to attempt
a single MPCC-clean polish pass after continuation:

```python
result = pympcc.solve(problem, strategy="scholtes", tnlp_refine=True)
print(result.tnlp_refined.comp_residual)   # typically several orders tighter
```

### Symptom: multipliers explode across outer iterations

IPOPT is dual-unstable, often a sign that MPCC-MFCQ fails.  Inspect:

```python
for it in result.history:
    if it.kkt_residual is not None:
        print(f"ε={it.epsilon:.2e}  ‖λ‖∞≈{np.abs(it.kkt_residual):.2e}")
```

If `‖λ‖∞` grows by more than ~$10^3$ between consecutive accepted
iterates, enable the rollback safeguard:

```python
result = pympcc.solve(
    problem, strategy="scholtes",
    safeguards="all",                 # rollback + adaptive ε + KKT-term + plateau
    rollback_lambda_jump=1e3,         # default
)
```

The rollback safeguard restores the previous iterate when the dual
explodes and backs off ε; the adaptive-ε safeguard makes the next
reduction more conservative on the way back down.

## Biactive pairs at the solution

### Symptom: `result.per_pair_status` contains `"biactive"` entries

Both $G_i$ and $H_i$ are within the biactive tolerance of zero
simultaneously.  This is *not* an error — it's a geometric property
of the local solution.  But it does mean MPCC-LICQ fails at $x^*$,
which knocks out implicit-differentiation results
(`solve_jax` falls back to the regularised Tikhonov adjoint and
emits a `UserWarning`).

Options:

1.  **Live with it.** Most strategies handle biactive pairs fine.
    `result.cq` will report `"MPCC-MFCQ"` or `"none"` rather than
    `"MPCC-LICQ"`; that's expected.
2.  **Force a non-biactive resolution.** Pass `cleanup="auto"` (or
    `safeguards="all"` which implies `cleanup="auto"`) so the
    polish phase pins each pair to one side using the magnitude
    rule:

    ```python
    result = pympcc.solve(problem, strategy="scholtes", cleanup="auto")
    print(result.cleanup_active_set)   # (I_G_active, I_H_active)
    print(result.per_pair_status)      # no "biactive" entries
    ```

3.  **Override the active-set partition.** When you know which side
    should be pinned (from physics, or from a continuation predecessor):

    ```python
    result = pympcc.solve(
        problem, strategy="scholtes",
        cleanup=True,
        cleanup_active_set=(I_G_idx, I_H_idx),   # disjoint, covers 0..n_comp-1
    )
    ```

### Symptom: `result.b_stationary` is `"unknown"`

B-stationarity certification ran out of biactive-set enumerations
(the default cap is `b_stat_max_biactive=10`, which corresponds to
$2^{10} = 1024$ tightened-NLP solves).  Either:

* Lower the biactive count by polishing first (`cleanup="auto"`
  often drops the count to 0); or
* Raise the cap explicitly:

  ```python
  result = pympcc.solve(problem, diagnostics=True, b_stat_max_biactive=14)
  ```

  Each unit raises the worst-case enumeration cost by 2x.

## When to use `tnlp_refine=True`

`tnlp_refine` is opt-in because it costs an extra IPOPT solve per
call.  Reach for it when:

* You need **certified MPCC multipliers** (the standard ε-relaxed
  multipliers conflate $\lambda_G$, $\lambda_H$, and $\lambda_\phi$).
* You're feeding multipliers into `pympcc.sensitivity(...)` or
  `solve_jax(...)`.  The TNLP refinement gives implicit
  differentiation a clean active-set Jacobian to work with.
* `diagnostics=True` + `tnlp_refine=True` is the canonical
  publication-grade configuration.

Skip it for:

* Single-shot solves where you only care about the primal $x^*$.
* Strategies that already terminate with clean multipliers
  (`"slack"`, `"augmented_lagrangian"` — the comp constraint is
  in a different layout than the standard ε-relaxed Scholtes
  formulation).

## What the diagnostic fields mean

`pympcc.solve(problem, diagnostics=True)` populates several
read-only fields on `MPCCResult`.  Quick reference:

| Field | Type | Meaning |
|---|---|---|
| `cq` | `str` | Strongest CQ that holds: `"MPCC-LICQ"`, `"MPCC-MFCQ"`, `"none"`. |
| `b_stationary` | `str` | `"yes"` / `"no"` / `"unknown"`. Verifies B-stationarity by enumerating tightened NLPs over biactive sub-sets. |
| `sosc` | `bool` | Second-order sufficient conditions check on the reduced Hessian. `True` = strict local minimum. |
| `kkt_residual` | `float` | Max-norm MPCC-KKT residual at $x^*$. Goal: $\le 10^{-6}$. |
| `merit_cross_check` | `dict` | Three independent merit functions; agreement is a smoke test for numerical health. Discrepancy of more than 10x between any pair is a red flag. |
| `jac_condition` | `float` | $\kappa_2$ of the active-constraint Jacobian. $> 10^{12}$ = badly conditioned (LICQ near-failure). |
| `hessian_condition_estimate` | `float` | $\kappa_2$ of the reduced Hessian. $> 10^{12}$ = SOSC near-failure. |
| `per_pair_status` | `list[str]` | One label per comp pair: `"G_active"`, `"H_active"`, `"biactive"`, `"inactive"`. |

If any of `cq`, `b_stationary`, `sosc` is missing (`None`),
`diagnostics` was either off or the corresponding check ran into a
failure mode it surfaces via `UserWarning` (e.g. SOSC needs the
Hessian; if you didn't supply `lagrangian_hessian` and JAX isn't
installed, SOSC is `None`).

## When in doubt

Run with `verbose=True` to see the per-iteration table.  The four
most diagnostic columns are:

```
  ipopt_it   — IPOPT iters this inner solve.  > 1000 = struggling.
  status     — 0 (Solve_Succeeded), 1 (Acceptable), 2 (INFEASIBLE),
               -4 (MAX_TIME).  Anything else = inner solve trouble.
  comp_max   — max|G_i·H_i|.  Should monotonically decrease.
  kkt_res    — MPCC-KKT residual.  Should match comp_max within 1-2 OOM.
```

A `kkt_res` that's *much* larger than `comp_max` means complementarity
holds but stationarity doesn't — typically a multiplier-recovery issue
that `tnlp_refine=True` will fix.
