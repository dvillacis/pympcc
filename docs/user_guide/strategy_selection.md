# Strategy selection

Thirteen reformulation strategies ship with pympcc.  This page is a
decision tree for picking a starting point.  The tree intentionally
errs on the side of the most-robust default; only deviate when you
have a concrete reason.

## TL;DR

```
Start with strategy="scholtes".
If it converges → done.
If comp_residual stagnates above 1e-6 → strategy="smoothing".
If IPOPT enters restoration repeatedly → strategy="lin_fukushima".
If MFCQ fails (multipliers blow up) → strategy="augmented_lagrangian".
If n is large and n_comp is small → strategy="slack".
If you need certified MPCC multipliers → tnlp_refine=True (any strategy).
```

## Decision tree

The tree below is keyed on observable properties of your problem.
Run a quick `pympcc.solve(problem, strategy="scholtes", diagnostics=True)`
to surface most of these signals before committing to a switch.

```
1.  Is this a one-shot feasibility check (don't care about a clean
    optimum, just want a feasible point)?
        → strategy="direct"   (single-NLP, no continuation loop)

2.  Does the problem have well-conditioned bounds on every comp pair
    AND smooth user callables?

    YES → strategy="scholtes"   (default; most-tested)
        Fails to drive comp_residual below 1e-6?
            → strategy="smoothing"          (FB equation; often tighter)
        IPOPT hits restoration > 5 iters per inner solve?
            → strategy="lin_fukushima"      (MFCQ-friendly bracket)
        Multipliers ‖λ‖∞ explode between iterations?
            → strategy="augmented_lagrangian"   (penalty in objective)

    NO (degenerate, or comp pair scales differ by > 100×) →
        Apply autoscale=True at solver level FIRST.
        Then branch by failure mode as above.

3.  Is n large and n_comp small (n / n_comp > ~10)?
        → strategy="slack"
        The complementarity block is lifted to slack variables
        s_G = G(x), s_H = H(x); the comp constraint rows then have
        zero entries in the x-block, which lets cyipopt avoid
        re-evaluating expensive primal Jacobians inside the comp loop.

4.  Do you need exact second-order info (Hessian-aware solver, Newton
    convergence in the inner loop)?
        → Pick from the smooth-NCP family with C^∞ smoothing:
          strategy="veelken_ulbrich_sin"   (arctan-based, smooth at 0)
        or, for a C^2 piecewise-polynomial alternative:
          strategy="veelken_ulbrich_pow"

5.  Are you embedding the solve in a larger differentiable
    pipeline (autograd / jax)?
        → use solve_jax(parametric, theta, strategy="scholtes", ...)
        The strategy choice is independent of differentiability;
        the wrapper is a custom_vjp that works with any strategy.
```

## When the FB-family is stalling

If `"smoothing"` (Fischer-Burmeister) stalls with comp_residual
plateauing above ~$10^{-6}$, try one of these NCP-family variants
before giving up on the smoothing approach entirely:

| Symptom | Try |
|---|---|
| FB hits restoration on a single pair | `"chen_mangasarian"` with `alpha=0.5` (asymmetric radicand resolves the stall) |
| FB converges to the cone but not onto it | `"billups"` with `gamma=0.05` – `0.1` (positive-part penalty pulls onto the complementary cone) |
| FB and inner-product penalty disagree | `"chen_chen_kanzow"` with `lam=0.7` (convex combination, smooth at 0) |
| Want a smooth-min that's truly $C^\infty$ | `"veelken_ulbrich_sin"` |

All of these share the standard ε-continuation knobs (`epsilon_0`,
`reduction`, `max_iter`, `epsilon_min`, `comp_tol`).  Sweep over them
with the unified `strategy="ncp"` entry point:

```python
for ncp_fn in ("chen_chen_kanzow", "chen_mangasarian", "billups"):
    result = pympcc.solve(
        problem, strategy="ncp",
        ncp_function=ncp_fn,
        ncp_params={"lam": 0.7} if ncp_fn == "chen_chen_kanzow" else None,
    )
    print(f"{ncp_fn:>20s}: comp_res={result.comp_residual:.2e}")
```

## Warm-start use cases

When you're solving a sequence of related MPCCs (parametric sweeps,
rolling-horizon MPC, multi-start with shared structure), pick the
strategy on the first solve and reuse it via
:class:`pympcc.MPCCSolver`:

```python
solver = pympcc.MPCCSolver(problem, strategy="scholtes")
for theta_k in theta_path:
    problem.x0 = warm_x0_for(theta_k)
    result = solver.resolve(problem)   # reuses internal warm-start state
```

`MPCCSolver.resolve` keeps multipliers, sparsity patterns, and the
inner cyipopt object alive across successive solves — typically
3-10x faster than constructing a fresh solver each time.  Strategy
choice doesn't change inside the loop; if you need to A/B different
strategies, use distinct `MPCCSolver` instances.

## Diagnostics first

When in doubt, gather more information before changing strategy:

```python
result = pympcc.solve(problem, strategy="scholtes", diagnostics=True)
print(result.cq)              # MPCC-LICQ / MPCC-MFCQ / none
print(result.b_stationary)    # B-stationarity verdict
print(result.sosc)            # second-order sufficient conditions
print(result.kkt_residual)    # MPCC-KKT residual
print(result.per_pair_status) # per-pair active-set classification
```

A `cq == "none"` result means MPCC-MFCQ failed at the solution; that's
the canonical indication that `"augmented_lagrangian"` is worth a
try.  A high biactive-pair count (many `"biactive"` entries in
`per_pair_status`) is the canonical indication that `"lin_fukushima"`
or `"chen_mangasarian"` may handle the geometry better than
`"scholtes"`.

See [troubleshooting](troubleshooting.md) for a symptom-to-action
table covering the most common failure modes.
