# EPEC: equilibrium problems with equilibrium constraints

`pympcc.bilevel.from_epec` compiles a multi-leader-common-follower
**Equilibrium Problem with Equilibrium Constraints** (EPEC) into a
single :class:`pympcc.MPCCProblem` solvable by every backend.

## Problem form

An EPEC has $N$ leaders, each choosing an own block $x_i \in
\mathbb{R}^{n_x^{(i)}}$ and sharing the same lower-level (follower)
decision $y \in \mathbb{R}^{n_y}$.  Leader $i$ solves the MPCC

$$
\begin{aligned}
\min_{x_i}\quad & F_i(x, y) \\
\text{s.t.}\quad & y \in \arg\min_y \{ f(x, y) : g(x, y) \le 0,\ h(x, y) = 0 \}
\end{aligned}
$$

where $x = (x_1, \dots, x_N)$ is the concatenated leader decision
vector.  The follower problem is **shared across all leaders**.

`from_epec` ships the diagonalised KKT-stack reformulation: each
leader's lower-level argmin is replaced by its KKT system, and the
$N$ resulting MPCCs are stacked into a single MPCC over
$(x, y, \lambda, \mu)$, where $\lambda, \mu$ are the lower-level
multipliers.  v1 supports common-follower EPECs only; private leader
constraints (`Leader.g_priv`, `Leader.h_priv`) are deferred to v2.

## Quickstart

```python
import jax.numpy as jnp
import numpy as np
import pympcc

# A two-leader Cournot game with a shared market-clearing follower:
#   leader i picks production x_i ≥ 0 to maximise its revenue
#   y[0] = market price, satisfying  Σ x_i + y[0] = demand
def F_leader_i(i):
    def F(x_full, y):
        # Leader i's profit = price · own production
        return -y[0] * x_full[i]
    return F

def f_lower(x_full, y):
    # Pick the price that clears the market; quadratic for smoothness
    return 0.5 * (y[0] - (10.0 - x_full.sum())) ** 2

def g_lower(x_full, y):
    # Lower-level constraint: y[0] ≥ 0 (price non-negative)
    return -y

leaders = [
    pympcc.Leader(n_x=1, F=F_leader_i(0), x0=np.array([2.0])),
    pympcc.Leader(n_x=1, F=F_leader_i(1), x0=np.array([2.0])),
]
common_lower = pympcc.LowerLevel(
    n_y=1, f=f_lower,
    n_g=1, g=g_lower,
    y0=np.array([5.0]),
)

problem = pympcc.from_epec(leaders=leaders, common_lower=common_lower)
result = pympcc.solve(problem, strategy="scholtes")
print(result.x)
```

The returned `result.x` carries the lifted decision vector
$z = (x_1, \dots, x_N, y, \lambda, \mu, \text{leader-duals})$.  The
leader-block sizes are known from the input dataclasses, so a
caller-side slice is straightforward:

```python
import numpy as np

n_x_per = [ld.n_x for ld in leaders]
x_offsets = np.cumsum([0] + n_x_per).tolist()
n_x_total = x_offsets[-1]
n_y, n_g, n_h = common_lower.n_y, common_lower.n_g, common_lower.n_h

x_full = result.x[:n_x_total]
y      = result.x[n_x_total : n_x_total + n_y]
lam    = result.x[n_x_total + n_y : n_x_total + n_y + n_g]   # lower-level inequality multipliers
mu     = result.x[n_x_total + n_y + n_g : n_x_total + n_y + n_g + n_h]   # eq multipliers

x_blocks = [x_full[x_offsets[i]:x_offsets[i + 1]] for i in range(len(leaders))]
```

The trailing `result.x[n_x_total + n_y + n_g + n_h:]` slice carries
the per-leader auxiliary multipliers used by the diagonalised KKT
stack; users typically don't read those directly.

## When to reach for `from_epec`

* **Cournot / Bertrand / Stackelberg games** with a common
  market-clearing follower.
* **Network games** where each player optimises over a shared
  congestion model.
* **Mechanism design** problems where bidders share a common-prior
  inference layer.

When the lower-level is **leader-specific** (each leader has its own
follower), use `from_lower_level` once per leader and stack the
results manually — `from_epec` is specifically the
common-follower diagonalisation and the v2 extension to private
follower constraints is on the roadmap.

## Derivative requirements

`from_epec` v1 supports `derivatives="jax"` only.  The emitted
equality rows differentiate the lower-level Lagrangian gradient
w.r.t. $(x, y, \lambda, \mu)$ — a second-order autodiff backend is
required, and the FD fallback is too noise-prone for this
formulation to converge reliably.  Passing `derivatives="fd"` raises
`NotImplementedError`; FD support is planned for v2.

Practically: install JAX (`pip install pympcc[jax]`) and write your
`F_i`, `f`, `g`, `h` callables in pure `jax.numpy` ops so the tracer
can compile them.

## Limitations (v1)

* Common follower only.  Private leader constraints (separate $g_i$,
  $h_i$ per leader) are deferred to v2.
* Equilibrium uniqueness is **not** verified — `from_epec` returns
  the diagonalised MPCC; whether it has a unique equilibrium depends
  on the payoff matrices, which the emitter doesn't analyse.
* `solve_jax(...)` differentiability through an EPEC emit is
  un-tested in v1; use the standard `solve(...)` path.

## See also

* [Bilevel programs (single leader)](bilevel.md) — `from_lower_level`
  for the simpler one-leader case.
* [Strategy selection](strategy_selection.md) — most EPECs benefit
  from `tnlp_refine=True` because the diagonalised KKT stack tends
  to land on degenerate biactive sets.
* §5.5 in [`ROADMAP.md`](https://github.com/dvillacis/pympcc/blob/main/ROADMAP.md)
  for the full design notes and v2 plan.
