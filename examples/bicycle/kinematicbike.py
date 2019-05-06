r"""
Model taken from
Kinematic  and  Dynamic  Vehicle  Models  for  Autonomous  Driving Control  Design. Kong et al.
https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf
"""


import time

import numpy as np
import funcy as fn

from redax.module import Interface, CompositeInterface
from redax.spaces import DynamicCover, EmbeddedGrid, FixedCover
from redax.synthesis import ReachGame, ControlPre, DecompCPre
from redax.visualizer import plot3D, plot3D_QT, pixel2D, scatter2D
from redax.utils.overapprox import maxmincos, maxminsin, shiftbox, bloatbox
from redax.predicates.dd import BDD
from redax.utils.heuristics import order_heuristic
from redax.ops import coarsen


# x
# y
# phi
# v
# beta

lf = 1.105
lr = 1.738

vmax = .5
# vmax = .3
angturn = 1.5
Ts = .1

def bikedynamics(state, control, ts=.1):
    x, y, phi, v, beta = state
    delta, a = control

    xnext = x + ts * (v * np.cos(phi + beta))
    ynext = y + ts * (v * np.sin(phi + beta))
    phinext = phi + ts * (v / lr) * np.sin(beta)
    vnext = v + ts * (a)
    betanext = beta + ts * np.arctan(lf/(lf+lr) * np.tan(delta))

    return (xnext, ynext, phinext, vnext, betanext)


def xwindow(x, v, theta):
    """
    Parameters
    ----------
    x: Tuple(float, float)
    v: float
    theta: Tuple(float, float)

    Returns
    -------
    Tuple(float, float)
        Overapproximation of possible values of x + v cos(theta)

    """
    mincos, maxcos = maxmincos(theta[0], theta[1])
    return x[0] + v * mincos, x[1] + v * maxcos

def ywindow(y, v, theta):
    """
    Parameters
    ----------
    y: Tuple(float, float)
    v: float
    theta: Tuple(float, float)

    Returns
    -------
    Tuple(float, float)
        Overapproximation of possible values of y + v sin(theta)

    """
    minsin, maxsin = maxminsin(theta[0], theta[1])
    return y[0] + v * minsin, y[1] + v * maxsin

def thetawindow(theta, v, omega):
    """
    Parameters
    ----------
    y: Tuple(float, float)
    v: float
    theta: Tuple(float, float)

    Returns
    -------
    Tuple(float, float)
        Overapproximation of possible values of y + v sin(theta)

    """
    return theta[0] + (1/L) * v * np.sin(omega), theta[1] + (1/L) * v*np.sin(omega)

def vwindow(v, a):
    return v[0] + Ts * a, v[1] + Ts * a

def setup():
    mgr = BDD()
    mgr.configure(reordering=False)

    # Spaces
    # Declare continuous state spaces
    xspace      = DynamicCover(0, 4)
    yspace      = DynamicCover(-.5,.5)
    anglespace  = DynamicCover(-np.pi, np.pi, periodic=True)
    vspace      = DynamicCover(0, 1)
    # Declare discrete control spaces
    accspace    = EmbeddedGrid(4, -3, 1)
    angaccspace = EmbeddedGrid(3, -angturn, angturn)

    # Interfaces
    """
    Declare interfaces
    """
    Fx        = Interface(mgr, {'x': pspace, 'theta': anglespace, 'v': vspace},
                                {'xnext': pspace})
    Fy        = Interface(mgr, {'y': pspace, 'theta': anglespace, 'v': vspace},
                                {'ynext': pspace})
    Fv        = Interface(mgr, {'v': vspace, 'a': accspace},
                               {'vnext': vspace})

    Ftheta    = Interface(mgr, {'theta': anglespace, 'v': vspace, 'omega': angaccspace},
                                {'thetanext': anglespace})




    return mgr, Fx, Fy, Fv, Ftheta

