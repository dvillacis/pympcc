# pympcc
[![PyPI](https://img.shields.io/pypi/v/pympcc)](https://pypi.org/project/pympcc/)
[![CI](https://github.com/dvillacis/pympcc/actions/workflows/tests.yml/badge.svg)](https://github.com/dvillacis/pympcc/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/pympcc/badge/?version=latest)](https://pympcc.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python solver for **Mathematical Programs with Complementarity Constraints (MPCC)** using [IPOPT](https://github.com/coin-or/Ipopt) via [cyipopt](https://github.com/mechmotum/cyipopt), with an optional SciPy backend.

> Thirteen reformulation strategies (six canonical + seven NCP variants), certified MPCC multipliers, B-stationarity / SOSC diagnostics, parametric sensitivity, and a `jax.custom_vjp`-registered solve for end-to-end differentiability.  See [strategy selection](https://pympcc.readthedocs.io/en/latest/user_guide/strategy_selection.html) for help picking one.

## Problem form

```
min  f(x)
s.t. g(x) ≤ 0              (inequality constraints)
     h(x) = 0              (equality constraints)
     G(x) ≥ 0  }
     H(x) ≥ 0  }           (complementarity)
     G(x)ᵀ H(x) = 0  }
```

## Install

**Requirements:** Python ≥ 3.11. The IPOPT backend also requires a working IPOPT installation.

```bash
brew install ipopt                           # macOS
sudo apt-get install coinor-libipopt-dev     # Linux

pip install pympcc                           # default — pulls cyipopt
pip install "pympcc[scipy]"                  # also include the SciPy backend
pip install "pympcc[jax]"                    # solve_jax + JAX-AD derivatives
pip install "pympcc[dev]"                    # for development
```

See [the documentation](https://pympcc.readthedocs.io/en/latest/installation.html) for the optional custom IPOPT linear-solver bridge and source builds.

## Quickstart

```python
import numpy as np
import pympcc

# min (x0-2)² + (x1-1)²   s.t.  x0 ≥ 0 ⊥ x1 ≥ 0
problem = pympcc.MPCCProblem(
    n=2, n_comp=1,
    x0=np.array([0.5, 0.5]),
    xl=np.zeros(2),
    objective=lambda x: (x[0] - 2)**2 + (x[1] - 1)**2,
    gradient=lambda x: np.array([2*(x[0]-2), 2*(x[1]-1)]),
    comp_G=lambda x: np.array([x[0]]),
    comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
    comp_H=lambda x: np.array([x[1]]),
    comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
)

result = pympcc.solve(problem, strategy="scholtes")
print(result.x, result.obj, result.success)   # [2. 0.] 1.0 True
```

## Strategies

| Strategy | How it works | Best for |
|---|---|---|
| `"direct"` | `G·H ≤ 0`; single solve | Quick feasibility checks |
| `"scholtes"` | `G·H ≤ ε`, drives `ε → 0` | **Default — most robust** |
| `"smoothing"` | Fischer-Burmeister `φ_ε(G,H) = 0`, drives `ε → 0` | Tight complementarity residuals |
| `"lin_fukushima"` | `G·H ≤ ε` and `G+H ≥ ε`; preserves MPCC-MFCQ | Degenerate problems |
| `"augmented_lagrangian"` | PHR penalty in objective | Problems where MFCQ fails |
| `"slack"` | Lifts `G`, `H` to slacks; sparse comp rows | Large-`n` problems (`n ≫ n_comp`) |

## Highlights

- **Certified stationarity** — `tnlp_refine=True` re-solves a tightened NLP with the active set fixed and extracts MPCC-clean multipliers; classifies S- / W-stationarity. ([docs](https://pympcc.readthedocs.io/en/latest/user_guide/diagnostics.html))
- **Diagnostics** — MPCC-LICQ / MPCC-MFCQ check, B-stationarity by branch enumeration, MPCC-SOSC, multi-merit cross-check, condition numbers, degeneracy report. ([docs](https://pympcc.readthedocs.io/en/latest/user_guide/diagnostics.html))
- **Parametric sensitivity** — `pympcc.sensitivity(result, problem, dgrad_L_dp=..., dc_dp=...)` returns `dx*/dp` and `dλ*/dp` from a single KKT linear solve. ([docs](https://pympcc.readthedocs.io/en/latest/user_guide/sensitivity.html))
- **Differentiable** — `pympcc.solve_jax(parametric, theta, ...)` is `jax.custom_vjp`-registered: differentiate transparently through a converged MPCC. ([docs](https://pympcc.readthedocs.io/en/latest/user_guide/autodiff.html))
- **Bilevel** — `pympcc.bilevel.from_lower_level(...)` compiles `min_x F(x,y) s.t. y ∈ argmin_y f(x,y)` to an MPCC by emitting lower-level KKT. ([docs](https://pympcc.readthedocs.io/en/latest/user_guide/bilevel.html))
- **Sparse-friendly** — COO-format Jacobians end-to-end; the `slack` strategy keeps the comp block at `2·n_comp` nonzeros regardless of `n`.
- **AMPL `.nl` reader** — `pympcc.frontend.ampl.from_nl(path)` for MacMPEC-format inputs.
- **Multistart and presolve** — `pympcc.multistart(...)`, `presolve=True`.

## Documentation

Full user guide, API reference, and theory primer at **[pympcc.readthedocs.io](https://pympcc.readthedocs.io)**.

## Testing

```bash
uv run pytest tests/ -v
```

The suite covers all thirteen strategies, all diagnostic modules, the MacMPEC benchmark collection, sensitivity, and `solve_jax`. The `direct` strategy tests are marked `xfail(strict=False)` because LICQ generically fails at MPCC feasible points.

## Acknowledgements

pympcc is a thin orchestration layer on top of two excellent
upstream projects:

- [**IPOPT**](https://github.com/coin-or/Ipopt) — the interior-point
  NLP solver that does the actual numerical heavy lifting (Eclipse
  Public License 2.0, &copy; the COIN-OR Foundation contributors).
- [**cyipopt**](https://github.com/mechmotum/cyipopt) — the Python
  bindings to IPOPT, by the cyipopt maintainers (EPL 2.0).

If you use pympcc in published work, please cite IPOPT alongside it.

## References

- Scholtes, S. (2001). Convergence properties of a regularization scheme for mathematical programs with complementarity constraints. *SIAM J. Optim.* 11(4), 918–936.
- Lin, G.-H., & Fukushima, M. (2003). New relaxation method for mathematical programs with complementarity constraints. *J. Optim. Theory Appl.* 118(1), 81–116.
- Fischer, A. (1992). A special Newton-type optimization method. *Optimization* 24(3–4), 269–284.
- Luo, Z.-Q., Pang, J.-S., & Ralph, D. (1996). *Mathematical Programs with Equilibrium Constraints*. Cambridge University Press.
- Leyffer, S. (2000). *MacMPEC — AMPL collection of MPECs.* Argonne National Laboratory.
