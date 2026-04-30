# Differentiable solve (`solve_jax`)

`pympcc.solve_jax` registers an MPCC solve as `jax.custom_vjp` so JAX can differentiate **through** the converged solution. Forward calls IPOPT; backward solves the KKT saddle once and contracts via `jax.vjp` on user-supplied parametric callables.

## `ParametricMPCC`

A parametric problem description whose callables take `(x, theta)` with JAX-traceable bodies.

```python
import jax.numpy as jnp
import pympcc

pmpcc = pympcc.ParametricMPCC(
    n=4, n_comp=2,
    objective=lambda x, theta: 0.5 * jnp.sum((x - theta) ** 2),
    comp_G=lambda x, theta: x[:2],
    comp_H=lambda x, theta: x[2:],
    # Optional:
    # eq_constraints=lambda x, theta: ...,
    # ineq_constraints=lambda x, theta: ...,
    # n_eq=..., n_ineq=...,
    # xl=..., xu=...,
)

# Materialise to a plain MPCCProblem (closes over θ):
problem = pmpcc.materialise(theta_np, x0=x0_np)
```

## `solve_jax`

```python
import jax
import jax.numpy as jnp

def loss(theta):
    x_star = pympcc.solve_jax(pmpcc, theta, x0=jnp.asarray(theta))
    return 0.5 * jnp.sum(x_star ** 2)

grad_theta = jax.grad(loss)(jnp.array([2.0, 0.0, 0.0, 3.0]))
```

The forward pass uses `tnlp_refine=True` automatically so the adjoint solve has certified MPCC multipliers.

## Phase-1 limitations

- ✅ `jax.grad`, `jax.value_and_grad`, scalar-output cotangents.
- ❌ `jax.jacrev` — internally `vmap`s the backward, which our NumPy/IPOPT bwd cannot trace. Assemble Jacobians per-row via `jax.grad`:

  ```python
  def x_i(theta, i):
      return pympcc.solve_jax(pmpcc, theta, x0=x0)[i]

  rows = [jax.grad(lambda th, i=i: x_i(th, i))(theta) for i in range(n)]
  J = jnp.stack(rows, axis=0)
  ```

- ❌ `jax.jit` — the forward is a NumPy/IPOPT call, not a traceable primitive.
- Phase-2 (custom_jvp for forward-mode, θ-dependent bounds, jit) is planned.

## Skip behaviour

Returns a zero θ-cotangent with a `UserWarning` when:

- the forward solve fails to converge, or
- the optimum is biactive (IFT prerequisites invalid).

Same skip semantics as the lower-level [`pympcc.sensitivity`](sensitivity.md) primitive.
