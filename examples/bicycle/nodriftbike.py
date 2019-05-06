r"""
Model taken from
Kinematic  and  Dynamic  Vehicle  Models  for  Autonomous  Driving Control  Design. Kong et al.
https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf
"""


import os

import time
directory = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import funcy as fn

from redax.module import Interface, CompositeInterface
from redax.spaces import DynamicCover, EmbeddedGrid, FixedCover
from redax.synthesis import ReachGame, ControlPre, DecompCPre, ReachAvoidGame
from redax.visualizer import plot3D, plot3D_QT, pixel2D, scatter2D
from redax.utils.overapprox import maxmincos, maxminsin, shiftbox, bloatbox
from redax.predicates.dd import BDD
from redax.utils.heuristics import order_heuristic
from redax.ops import coarsen, ihide
from redax.controllers import MemorylessController

L = .7

vmax = .5
# vmax = .3
angturn = 1.4
Ts = .2

def bikedynamics(state, control, ts=Ts):
    x, y, theta, v = state['x'], state['y'], state['theta'], state['v']
    a, omega = control['a'], control['omega']

    xnext = x + ts * (v * np.cos(theta))
    ynext = y + ts * (v * np.sin(theta))
    thetanext = theta + ts* (v/L) *np.sin(omega)
    vnext = v + ts * (a)
    # betanext = beta + ts * np.arctan(lf/(lf+lr) * np.tan(delta))

    return (xnext, ynext, thetanext, vnext)


def xwindow(x, v, theta):
    """
    Parameters
    ----------
    x: Tuple(float, float)
    v: Tuple(float, float)
    theta: Tuple(float, float)

    Returns
    -------
    Tuple(float, float)
        Overapproximation of possible values of x + v cos(theta)

    """
    mincos, maxcos = maxmincos(theta[0], theta[1])
    bounds = [ x[0] + Ts * v[0] * mincos, x[0] + Ts * v[0] * maxcos,
               x[0] + Ts * v[1] * mincos, x[0] + Ts * v[1] * maxcos,
               x[1] + Ts * v[0] * mincos, x[1] + Ts * v[0] * maxcos,
               x[1] + Ts * v[1] * mincos, x[1] + Ts * v[1] * maxcos,
            ]
    left = min(bounds)
    right = max(bounds)

    return left, right

def ywindow(y, v, theta):
    """
    Parameters
    ----------
    y: Tuple(float, float)
    v: Tuple(float, float)
    theta: Tuple(float, float)

    Returns
    -------
    Tuple(float, float)
        Overapproximation of possible values of y + v sin(theta)

    """
    minsin, maxsin = maxminsin(theta[0], theta[1])
    bounds = [ y[0] + Ts * v[0] * minsin, y[0] + Ts * v[0] * maxsin,
               y[0] + Ts * v[1] * minsin, y[0] + Ts * v[1] * maxsin,
               y[1] + Ts * v[0] * minsin, y[1] + Ts * v[0] * maxsin,
               y[1] + Ts * v[1] * minsin, y[1] + Ts * v[1] * maxsin,
            ]
    left = min(bounds)
    right = max(bounds)
    return left, right

def thetawindow(theta, v, omega):
    """
    Parameters
    ----------
    theta: Tuple(float, float)
    v: Tuple(float, float)
    omega: float

    Returns
    -------
    Tuple(float, float)
        Overapproximation of possible values of y + v sin(theta)

    """
    sinval = np.sin(omega)
    if sinval > 0:
        return theta[0] + Ts*(1/L) * v[0] * sinval, theta[1] + Ts* (1/L) * v[1]*sinval
    else:
        return theta[0] + Ts*(1/L) * v[1] * sinval, theta[1] + Ts* (1/L) * v[0]*sinval

def vwindow(v, a):
    return v[0] + Ts * a, v[1] + Ts * a

def setup():
    mgr = BDD()
    mgr.configure(reordering=False)

    # Spaces
    # Declare continuous state spaces
    xspace      = DynamicCover(0, 4)
    yspace      = DynamicCover(-.5,.5)
    anglespace  = DynamicCover(-2, 2)
    vspace      = DynamicCover(0, .8)
    # Declare discrete control spaces
    accspace    = EmbeddedGrid(4, -2.1, .7)
    angaccspace = EmbeddedGrid(3, -angturn, angturn)

    # Interfaces
    """
    Declare interfaces
    """
    Fx        = Interface(mgr, {'x': xspace, 'theta': anglespace, 'v': vspace},
                                {'xnext': xspace})
    Fy        = Interface(mgr, {'y': yspace, 'theta': anglespace, 'v': vspace},
                                {'ynext': yspace})
    Fv        = Interface(mgr, {'v': vspace, 'a': accspace},
                               {'vnext': vspace})

    Ftheta    = Interface(mgr, {'theta': anglespace, 'v': vspace, 'omega': angaccspace},
                                {'thetanext': anglespace})

    return mgr, Fx, Fy, Fv, Ftheta


def coarse_abstract(f: Interface, concrete, bits=6, verbose=False):
    r"""
    Constructs an abstraction of interface f with two passes with
    overlapping input sets.

    First pass with 2^bits bins along each dimension.
    Second pass with same granuarity but shifted by .5.

    The abstraction is saved with 2^(bits+1) bins along each dimension.
    """
    iter_precision = {'x': bits, 'y':bits, 'theta': bits, 'v':  bits,
                 'xnext': bits, 'ynext': bits, 'thetanext': bits, 'vnext':bits}
    save_precision = {k: v+1 for k,v in iter_precision.items()}
    outvar = fn.first(f.outputs)

    if verbose:
        print("Abstracting: ", f)

    for shift in [0.0, 0.5]:
        iter = f.input_iter(precision=iter_precision)
        for iobox in iter:

            for k, v in iobox.copy().items():
                if isinstance(v, tuple):
                    iobox[k] = bloatbox(shiftbox(v, shift), .001)

            # Assign output
            iobox[outvar] = concrete(**iobox)

            f = f.io_refined(iobox, nbits=save_precision)

    return f

def make_safe(mgr, composite: CompositeInterface):

    xspace = composite['x']
    yspace = composite['y']
    anglespace = composite['theta']
    vspace = composite['v']

    obs1 = xspace.conc2pred(mgr, 'x', [.5,1.5], 7, innerapprox=False)
    obs1 &= yspace.conc2pred(mgr, 'y', [-.15,.15], 7, innerapprox=False)

    obs2 = xspace.conc2pred(mgr, 'x', [2.5,3.5], 7, innerapprox=False)
    obs2 &= yspace.conc2pred(mgr, 'y', [-.4,0], 7, innerapprox=False)

    obs3 = xspace.conc2pred(mgr, 'x', [1.9,2.1], 7, innerapprox=False)
    obs3 &= yspace.conc2pred(mgr, 'y', [-.35,-.3], 7, innerapprox=False)

    obs = obs1 | obs2 | obs3

    safeinter = Interface(mgr,
                          {'x': xspace, 'y': yspace, 'theta': anglespace, 'v': vspace},
                          {},
                          guar=mgr.true,
                          assum=~obs)

    return safeinter

def make_target(mgr, composite: CompositeInterface):

    xspace = composite['x']
    yspace = composite['y']
    anglespace = composite['theta']
    vspace = composite['v']

    target =  xspace.conc2pred(mgr, 'x', [3.6,3.99], 7, innerapprox=False)
    target &= yspace.conc2pred(mgr, 'y', [-.4,-.3], 7, innerapprox=False)

    targetinter = Interface(mgr,
                          {'x': xspace, 'y': yspace, 'theta': anglespace, 'v': vspace},
                          {},
                          guar=mgr.true,
                          assum=target)

    return targetinter

def simulate(controller: MemorylessController, steps:int=20):
    """
    Simulate trajectories
    """
    state = {'x': .2, 'y': .3, 'theta': 0, 'v': 0.1}
    for step in range(steps):
        print(step, state)

        # Sample a valid safe control input from the controller.allows(state) iterator
        u = fn.first(controller.allows(state))
        print("Got control")
        if u is None:
            print("No more valid control inputs. Terminating simulation.")
            break
        state.update(u)

        # Iterate dynamics
        nextstate = bikedynamics(state, u)

        state = {'x': nextstate[0],
                 'y': nextstate[1],
                 'theta': nextstate[2],
                 'v': nextstate[3]}

if __name__ == "__main__":
    pass

    mgr, Fx, Fy, Fv, Ftheta = setup()


    print("Parameters:  vmax = {}, angturn = {}, Ts = {}, L = {}".format(vmax, angturn, Ts, L))

    bits =  6 # Coarse iteration bits. Save precision is bits + 1
    starttime = time.time()
    Fv = coarse_abstract(Fv, vwindow, bits=bits, verbose=True)
    Ftheta = coarse_abstract(Ftheta, thetawindow, bits=bits, verbose=True)
    mgr.reorder(order_heuristic(mgr)); mgr.configure(reordering=False)
    Fx = coarse_abstract(Fx, xwindow, bits=bits, verbose=True)
    Fy = coarse_abstract(Fy, ywindow, bits=bits, verbose=True)
    print("Abs Time: ", time.time() - starttime)

    F = CompositeInterface([Fx, Fy, Fv, Ftheta])
    mgr.reorder(order_heuristic(mgr))
    mgr.configure(reordering=False)


    safe = make_safe(mgr, F)
    # pixel2D(mgr, ('x', F['x']), ('y', F['y']), safe.pred)
    target = make_target(mgr, F)
    # pixel2D(mgr, ('x', F['x']), ('y', F['y']), target.pred)

    # Combine
    # Fxy = Fx * Fy * ihide(safe, ['v', 'theta']) # Combine position dynamics with the safety region constraint
    # F = CompositeInterface([Fxy, Fv, Ftheta])

    # Subroutine to print intermediate results
    def print_xy_proj(Z: Interface):
        print_xy_proj.counter += 1
        pixel2D(mgr, ('x', F['x']), ('y', F['y']), ihide(Z, ['theta', 'v']).pred,
                  fname="{}/figs/xyslowernocomp_{}".format(directory, print_xy_proj.counter)
                 )
        return Z
    print_xy_proj.counter = 0


    cpre = DecompCPre(F,
                     states= (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext'), ('v', 'vnext')),
                     control=('a', 'omega'),
                     pre_process=print_xy_proj
                     )

    starttime = time.time()
    game = ReachAvoidGame(cpre, safe, target)
    basin, steps, controller = game.run(verbose=True)
    # controller.C._assum = mgr.load("{0}/contr_assum.dddmp".format(directory))

    print("Game Solve Time: ", time.time() - starttime)
    simulate(controller, steps=20)

