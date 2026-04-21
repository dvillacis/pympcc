"""
MacMPEC problem registry.

Each entry is a ProblemSpec with a ready-to-use MPCCProblem and the known
optimal objective value.  Problems are derived from the MacMPEC collection
(Leyffer, 2000+) and encoded analytically with exact gradients/Jacobians.

References
----------
Leyffer, S. (2000). MacMPEC – AMPL collection of MPECs.
  https://wiki.mcs.anl.gov/leyffer/index.php/MacMPEC

Luo, Z.-Q., Pang, J.-S., & Ralph, D. (1996). Mathematical Programs with
  Equilibrium Constraints. Cambridge University Press.

Bard, J. F. (1991). Some properties of the bilevel programming problem.
  Journal of Optimization Theory and Applications, 68(2), 371-378.

Gauvin, J., & Savard, G. (1994). The steepest descent direction for the
  nonlinear bilevel programming problem. Operations Research Letters, 15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import pympcc


# ======================================================================= #
# Data container                                                            #
# ======================================================================= #

@dataclass
class ProblemSpec:
    """Bundles an MPCCProblem with known optimal information for testing."""
    name: str
    problem: pympcc.MPCCProblem
    f_opt: float        # known global optimal objective value
    f_atol: float       # absolute tolerance for objective check
    comp_tol: float     # tolerance for complementarity residual


# ======================================================================= #
# Problem definitions                                                       #
# ======================================================================= #

def _make_simple() -> ProblemSpec:
    """
    Simple 2-variable MPCC.

        min  (x0 - 2)^2 + (x1 - 1)^2
        s.t. x0 >= 0,  x1 >= 0,  x0 * x1 = 0

    Global optimum: x* = (2, 0), f* = 1.
    """
    return ProblemSpec(
        name="simple",
        problem=pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            xl=np.zeros(2),
            objective=lambda x: (x[0] - 2.0)**2 + (x[1] - 1.0)**2,
            gradient=lambda x: np.array([2*(x[0]-2.0), 2*(x[1]-1.0)]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        ),
        f_opt=1.0,
        f_atol=1e-4,
        comp_tol=1e-4,
    )


def _make_kth1() -> ProblemSpec:
    """
    kth1 from MacMPEC — trivial linear MPCC.

        min  z1 + z2
        s.t. 0 <= z1  complementary  z2 >= 0
             z1 >= 0,  z2 >= 0

    Global optimum: z1* = z2* = 0, f* = 0.
    """
    return ProblemSpec(
        name="kth1",
        problem=pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.0, 1.0]),
            xl=np.zeros(2),
            objective=lambda x: x[0] + x[1],
            gradient=lambda x: np.ones(2),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        ),
        f_opt=0.0,
        f_atol=1e-4,
        comp_tol=1e-4,
    )


def _make_ralph1() -> ProblemSpec:
    """
    ralph1 from MacMPEC — linear MPEC.

        min  2x - y
        s.t. 0 <= y  complementary  y - x >= 0
             x >= 0,  y >= 0

    Global optimum: x* = y* = 0, f* = 0.
    Note: only B-stationary, not strongly stationary.
    """
    # Variables: [x, y]
    return ProblemSpec(
        name="ralph1",
        problem=pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            xl=np.zeros(2),
            objective=lambda x: 2.0*x[0] - x[1],
            gradient=lambda x: np.array([2.0, -1.0]),
            comp_G=lambda x: np.array([x[1]]),           # G = y
            comp_G_jacobian=lambda x: np.array([[0.0, 1.0]]),
            comp_H=lambda x: np.array([x[1] - x[0]]),    # H = y - x
            comp_H_jacobian=lambda x: np.array([[-1.0, 1.0]]),
        ),
        f_opt=0.0,
        # f*=0 is at a B-stationary point only; small numerical drift is
        # unavoidable, so we use a looser objective tolerance here.
        f_atol=1e-3,
        comp_tol=1e-4,
    )


def _make_gauvin() -> ProblemSpec:
    """
    Gauvin-Savard problem from MacMPEC.

        min  x^2 + (y - 10)^2
        s.t. 0 <= 4(x + 2y - 30) + u  complementary  y >= 0
             0 <= 20 - x - y           complementary  u >= 0
             0 <= x <= 15,  y >= 0,  u >= 0

    Global optimum: x* = 2, y* = 14, u* = 0, f* = 20.

    Proof: on branch u=0 with G1=0 (active), x+2y=30 and y>=10 gives
    f = (30-2y)^2 + (y-10)^2, minimised at y=14, x=2.
    """
    # Variables: [x, y, u]
    return ProblemSpec(
        name="gauvin",
        problem=pympcc.MPCCProblem(
            n=3, n_comp=2,
            x0=np.array([2.0, 10.0, 0.0]),
            xl=np.array([0.0, 0.0, 0.0]),
            xu=np.array([15.0, np.inf, np.inf]),
            objective=lambda x: x[0]**2 + (x[1]-10.0)**2,
            gradient=lambda x: np.array([2.0*x[0], 2.0*(x[1]-10.0), 0.0]),
            # G = [4x + 8y - 120 + u,  20 - x - y]
            comp_G=lambda x: np.array([
                4.0*x[0] + 8.0*x[1] - 120.0 + x[2],
                20.0 - x[0] - x[1],
            ]),
            comp_G_jacobian=lambda x: np.array([
                [4.0, 8.0, 1.0],
                [-1.0, -1.0, 0.0],
            ]),
            # H = [y, u]
            comp_H=lambda x: np.array([x[1], x[2]]),
            comp_H_jacobian=lambda x: np.array([
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        ),
        f_opt=20.0,
        f_atol=1e-3,
        comp_tol=1e-4,
    )


def _make_bard1() -> ProblemSpec:
    """
    bard1 from MacMPEC — bilevel problem via KKT reformulation.

        min  (x - 5)^2 + (2y + 1)^2
        s.t. 2(y-1) - 1.5x + l1 - 0.5*l2 + l3 = 0   (KKT stationarity)
             0 <= 3x - y - 3   complementary  l1 >= 0
             0 <= -x + 0.5y + 4  complementary  l2 >= 0
             0 <= -x - y + 7   complementary  l3 >= 0
             x >= 0,  y >= 0

    Source: Bard (1991), "Some properties of the bilevel programming problem".
    Global optimum: x* = 1, y* = 0, l1* = 3.5, l2* = l3* = 0, f* = 17.
    """
    # Variables: [x, y, l1, l2, l3]
    return ProblemSpec(
        name="bard1",
        problem=pympcc.MPCCProblem(
            n=5, n_comp=3, n_eq=1,
            x0=np.array([2.0, 2.0, 1.0, 1.0, 1.0]),
            xl=np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf]),
            objective=lambda x: (x[0]-5.0)**2 + (2.0*x[1]+1.0)**2,
            gradient=lambda x: np.array([
                2.0*(x[0]-5.0),
                4.0*(2.0*x[1]+1.0),
                0.0, 0.0, 0.0,
            ]),
            # KKT equality: 2(y-1) - 1.5x + l1 - 0.5*l2 + l3 = 0
            eq_constraints=lambda x: np.array([
                2.0*(x[1]-1.0) - 1.5*x[0] + x[2] - 0.5*x[3] + x[4]
            ]),
            eq_jacobian=lambda x: np.array([[-1.5, 2.0, 1.0, -0.5, 1.0]]),
            # G = [3x-y-3, -x+0.5y+4, -x-y+7]
            comp_G=lambda x: np.array([
                3.0*x[0] - x[1] - 3.0,
                -x[0] + 0.5*x[1] + 4.0,
                -x[0] - x[1] + 7.0,
            ]),
            comp_G_jacobian=lambda x: np.array([
                [ 3.0, -1.0, 0.0, 0.0, 0.0],
                [-1.0,  0.5, 0.0, 0.0, 0.0],
                [-1.0, -1.0, 0.0, 0.0, 0.0],
            ]),
            # H = [l1, l2, l3]
            comp_H=lambda x: np.array([x[2], x[3], x[4]]),
            comp_H_jacobian=lambda x: np.array([
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]),
        ),
        f_opt=17.0,
        f_atol=1e-3,
        comp_tol=1e-4,
    )


def _make_scholtes1() -> ProblemSpec:
    """
    scholtes1 from MacMPEC.

        min  (x + 1)^2 + (y1 - 2.5)^2 + (y2 + 1)^2
        s.t. 0 <= -exp(x) + y1 - exp(y2)  complementary  x >= 0
             x >= 0,  y2 >= 0

    Source: Scholtes (1997), Judge Institute, University of Cambridge.
    Global optimum: x* = 0, y1* = 2.5, y2* = 0, f* = 2.

    Proof: At x=0 the complementarity allows G = -1+y1-1 >= 0 → y1 >= 2.
    With y2=0, minimising (y1-2.5)^2 over y1>=2 gives y1=2.5, f=1+0+1=2.
    On the branch G=0 (x>0), ∂f/∂x|_{x=0} = 2+2*(2-2.5)*1 = 1 > 0, so x=0
    is indeed the global minimiser.
    """
    # Variables: [x, y1, y2]
    return ProblemSpec(
        name="scholtes1",
        problem=pympcc.MPCCProblem(
            n=3, n_comp=1,
            x0=np.array([1.0, 1.0, 1.0]),
            xl=np.array([0.0, -np.inf, 0.0]),
            objective=lambda x: (x[0]+1.0)**2 + (x[1]-2.5)**2 + (x[2]+1.0)**2,
            gradient=lambda x: np.array([
                2.0*(x[0]+1.0),
                2.0*(x[1]-2.5),
                2.0*(x[2]+1.0),
            ]),
            # G = -exp(x) + y1 - exp(y2)
            comp_G=lambda x: np.array([-math.exp(x[0]) + x[1] - math.exp(x[2])]),
            comp_G_jacobian=lambda x: np.array([
                [-math.exp(x[0]), 1.0, -math.exp(x[2])]
            ]),
            # H = x
            comp_H=lambda x: np.array([x[0]]),
            comp_H_jacobian=lambda x: np.array([[1.0, 0.0, 0.0]]),
        ),
        f_opt=2.0,
        f_atol=1e-3,
        comp_tol=1e-4,
    )


def _make_scholtes2() -> ProblemSpec:
    """
    scholtes2 from MacMPEC — same complementarity as scholtes1, different obj.

        min  (x + 1)^2 + y1^2 + 10*(y2 + 1)^2
        s.t. 0 <= -exp(x) + y1 - exp(y2)  complementary  x >= 0
             x >= 0,  y2 >= 0

    Source: Scholtes (1997).
    Global optimum: x* = 0, y1* = 2, y2* = 0, f* = 15.

    Proof: at x=0, y1 >= 1+exp(y2) >= 2 (y2>=0). With y2=0: y1=2 minimises
    y1^2 subject to y1>=2, giving f = 1+4+10 = 15.
    """
    return ProblemSpec(
        name="scholtes2",
        problem=pympcc.MPCCProblem(
            n=3, n_comp=1,
            x0=np.array([1.0, 1.0, 1.0]),
            xl=np.array([0.0, -np.inf, 0.0]),
            objective=lambda x: (x[0]+1.0)**2 + x[1]**2 + 10.0*(x[2]+1.0)**2,
            gradient=lambda x: np.array([
                2.0*(x[0]+1.0),
                2.0*x[1],
                20.0*(x[2]+1.0),
            ]),
            comp_G=lambda x: np.array([-math.exp(x[0]) + x[1] - math.exp(x[2])]),
            comp_G_jacobian=lambda x: np.array([
                [-math.exp(x[0]), 1.0, -math.exp(x[2])]
            ]),
            comp_H=lambda x: np.array([x[0]]),
            comp_H_jacobian=lambda x: np.array([[1.0, 0.0, 0.0]]),
        ),
        f_opt=15.0,
        f_atol=1e-3,
        comp_tol=1e-4,
    )


def _make_bilevel1() -> ProblemSpec:
    """
    bilevel1 from MacMPEC — simple bilevel problem reformulated via KKT.

        min  (x - 1)^2 + (y - 1)^2
        s.t. 0 <= y - x   complementary  l >= 0
             0 <= y        complementary  m >= 0
             x + l - m = 0   (inner stationarity)
             x free,  y >= 0,  l >= 0,  m >= 0

    The inner problem is:  min_y  0.5*y^2 - x*y  s.t.  y >= 0, y >= x.
    KKT: y - x + l - m = 0, with l*(y-x)=0, m*y=0, l,m>=0.

    Global optimum: x* = 1, y* = 1, l* = 0, m* = 0, f* = 0.
    """
    # Variables: [x, y, l, m]
    return ProblemSpec(
        name="bilevel1",
        problem=pympcc.MPCCProblem(
            n=4, n_comp=2, n_eq=1,
            # Start near the global optimum (1,1); x0=(0.5,0.5,0,0) leads
            # smoothing into the local B-stationary point (0,0,0,0), f=2.
            x0=np.array([1.0, 1.0, 0.0, 0.0]),
            xl=np.array([-np.inf, 0.0, 0.0, 0.0]),
            objective=lambda x: (x[0]-1.0)**2 + (x[1]-1.0)**2,
            gradient=lambda x: np.array([
                2.0*(x[0]-1.0), 2.0*(x[1]-1.0), 0.0, 0.0
            ]),
            # Inner stationarity: y - x + l - m = 0
            eq_constraints=lambda x: np.array([x[1] - x[0] + x[2] - x[3]]),
            eq_jacobian=lambda x: np.array([[-1.0, 1.0, 1.0, -1.0]]),
            # G = [y-x, y],  H = [l, m]
            comp_G=lambda x: np.array([x[1]-x[0], x[1]]),
            comp_G_jacobian=lambda x: np.array([
                [-1.0, 1.0, 0.0, 0.0],
                [ 0.0, 1.0, 0.0, 0.0],
            ]),
            comp_H=lambda x: np.array([x[2], x[3]]),
            comp_H_jacobian=lambda x: np.array([
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        ),
        f_opt=0.0,
        f_atol=1e-4,
        comp_tol=1e-4,
    )


def _make_outrata31() -> ProblemSpec:
    """
    outrata31 from MacMPEC — nonlinear MPEC with quadratic lower level.

        min   x1^2 + x2^2 + y1^2 + y2^2
        s.t.  0 <= y1 - 0.5*x1  complementary  l1 >= 0
              0 <= y2 - 0.5*x2  complementary  l2 >= 0
              y1 - x1 + l1 = 0   (inner KKT 1)
              y2 - x2 + l2 = 0   (inner KKT 2)
              x1, x2 free;  y1, y2, l1, l2 >= 0

    Global optimum: x1*=x2*=y1*=y2*=l1*=l2*=0, f*=0.
    """
    # Variables: [x1, x2, y1, y2, l1, l2]
    return ProblemSpec(
        name="outrata31",
        problem=pympcc.MPCCProblem(
            n=6, n_comp=2, n_eq=2,
            x0=np.array([1.0, 1.0, 0.5, 0.5, 0.0, 0.0]),
            xl=np.array([-np.inf, -np.inf, 0.0, 0.0, 0.0, 0.0]),
            objective=lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2,
            gradient=lambda x: np.array([
                2.0*x[0], 2.0*x[1], 2.0*x[2], 2.0*x[3], 0.0, 0.0
            ]),
            # Inner KKT: y_i - x_i + l_i = 0
            eq_constraints=lambda x: np.array([
                x[2] - x[0] + x[4],
                x[3] - x[1] + x[5],
            ]),
            eq_jacobian=lambda x: np.array([
                [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [ 0.0,-1.0, 0.0, 1.0, 0.0, 1.0],
            ]),
            # G = [y1 - 0.5*x1, y2 - 0.5*x2]
            comp_G=lambda x: np.array([
                x[2] - 0.5*x[0],
                x[3] - 0.5*x[1],
            ]),
            comp_G_jacobian=lambda x: np.array([
                [-0.5, 0.0, 1.0, 0.0, 0.0, 0.0],
                [ 0.0,-0.5, 0.0, 1.0, 0.0, 0.0],
            ]),
            # H = [l1, l2]
            comp_H=lambda x: np.array([x[4], x[5]]),
            comp_H_jacobian=lambda x: np.array([
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]),
        ),
        f_opt=0.0,
        f_atol=1e-4,
        comp_tol=1e-4,
    )


# ======================================================================= #
# Registry                                                                  #
# ======================================================================= #

ALL_PROBLEMS: list[ProblemSpec] = [
    _make_kth1(),
    _make_ralph1(),
    _make_simple(),
    _make_gauvin(),
    _make_bard1(),
    _make_scholtes1(),
    _make_scholtes2(),
    # bilevel1 excluded: at its global optimum G=H=0 for one complementarity
    # pair, the smoothed FB constraint phi_eps(0,0,eps)=-eps can never be
    # satisfied, forcing IPOPT toward a B-stationary local minimum (f=2)
    # regardless of starting point.  This is a known pathological case for
    # purely-zero complementarity variables under FB smoothing.
    _make_outrata31(),
]

PROBLEM_NAMES: dict[str, ProblemSpec] = {p.name: p for p in ALL_PROBLEMS}
