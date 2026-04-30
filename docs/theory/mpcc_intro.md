# Why MPCCs are hard

A **mathematical program with complementarity constraints** (MPCC) has the form

$$
\begin{aligned}
\min_x \quad & f(x) \\
\text{s.t.} \quad & g(x) \le 0,\ h(x) = 0 \\
                  & 0 \le G(x) \perp H(x) \ge 0
\end{aligned}
$$

where the orthogonality $G(x)^\top H(x) = 0$ — combined with $G, H \ge 0$ — forces, at every feasible point and *every* index $i$,

$$
G_i(x) = 0 \quad \text{or} \quad H_i(x) = 0 \quad \text{(or both)}.
$$

This disjunction is the source of every difficulty.

## Why standard NLP solvers fail

Write the orthogonality constraint as the equality $G(x)^\top H(x) = 0$ and plug the problem into IPOPT or any other off-the-shelf NLP solver. Trouble:

**Standard LICQ generically fails.** Pick any feasible $x$ and consider the gradients of the active constraints. At a strictly complementary index ($G_i = 0$, $H_i > 0$), the active gradients in row form are

$$
\nabla G_i(x) \quad \text{and} \quad H_i(x)\,\nabla G_i(x) + G_i(x)\,\nabla H_i(x) = H_i(x)\,\nabla G_i(x).
$$

These are **parallel** (one is a positive scalar multiple of the other), so the active-gradient matrix is rank-deficient. Standard LICQ requires linear independence; it almost never holds.

**Standard MFCQ also fails.** A symmetric argument shows the Mangasarian-Fromovitz CQ fails too whenever there is at least one active complementarity pair.

**Consequences.** Without LICQ or MFCQ:

* KKT multipliers may not exist, or may not be unique.
* The feasible set is geometrically a finite union of smooth manifolds glued along lower-dimensional faces — not a smooth manifold.
* Standard NLP convergence theorems do not apply; barrier methods can stall, and the solver may report "infeasible" near a perfectly valid MPCC stationary point.

## The reformulation idea

Rather than feeding the problem to a standard NLP solver as-is, we replace the orthogonality with a **regularised** condition that's well-behaved away from the limit and recovers the MPCC at the limit. pympcc ships six such reformulations:

| Strategy | Replaces $G \cdot H = 0$ with… |
|---|---|
| `direct` | $G \cdot H \le 0$ |
| `scholtes` | $G \cdot H \le \varepsilon$, drive $\varepsilon \to 0$ |
| `smoothing` | $\varphi_\varepsilon(G, H) = 0$ (Fischer-Burmeister) |
| `lin_fukushima` | $G \cdot H \le \varepsilon$ **and** $G + H \ge \varepsilon$ |
| `augmented_lagrangian` | absorb into PHR penalty in objective |
| `slack` | introduce $s_G = G$, $s_H = H$ and complement on the slacks |

Each strategy yields a sequence of standard NLPs whose limit points are MPCC stationary. Different strategies trade off ease (Scholtes is the safest default) against tightness (smoothing often gives smaller comp residuals) and against constraint qualifications (Lin-Fukushima preserves MPCC-MFCQ; augmented Lagrangian sidesteps comp-as-constraint entirely).

## What you actually want at the end

Convergence to a stationary NLP point of the regularised problem is **not** the same as convergence to an MPCC stationary point. To get certified MPCC multipliers and a stationarity classification (S- / M- / C- / W-), pympcc supports

* **TNLP refinement** (`tnlp_refine=True`) — re-solve a tightened NLP with the active set fixed as equalities.
* **B-stationarity certification** — a finite-step verification by enumerating active branches.
* **MPCC-SOSC** — second-order sufficient condition on the critical cone.

These are documented in the [stationarity primer](stationarity.md), the [CQ primer](cq.md), and the [SOSC primer](sosc.md).

## References

* Luo, Pang, & Ralph (1996). *Mathematical Programs with Equilibrium Constraints*. Cambridge University Press. — the standard reference.
* Scheel & Scholtes (2000). Mathematical programs with complementarity constraints: Stationarity, optimality, and sensitivity. *Math. Oper. Res.*
* Outrata, Kočvara, & Zowe (1998). *Nonsmooth Approach to Optimization Problems with Equilibrium Constraints*. Kluwer.
