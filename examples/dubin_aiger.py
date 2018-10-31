r"""
Dubins vehicle example.
"""

import time

import numpy as np

from aiger.expr import BoolExpr
from redax.predicates.aiger import aigerwrapper
from aiger_analysis.abc import simplify

from redax.module import Interface, CompositeModule
from redax.spaces import DynamicCover, EmbeddedGrid, FixedCover
from redax.synthesis import ReachGame, ControlPre, DecompCPre
from redax.visualizer import plot3D, plot3D_QT, pixel2D

from redax.utils.overapprox import maxmincos, maxminsin

"""
Specify dynamics and overapproximations
"""

L= 1.4
vmax = .5
def dynamics(x,y,theta,v,omega):
    return x + v*np.cos(theta), y + v*np.sin(theta), theta + (1/L) * v * np.sin(omega)

# Overapproximations of next states for each component
# v and omega are points and not windows
def xwindow(x, v, theta):
    mincos, maxcos = maxmincos(theta[0], theta[1])
    return x[0] + v * mincos, x[1] + v * maxcos
def ywindow(y, v, theta):
    minsin, maxsin = maxminsin(theta[0], theta[1])
    return y[0] + v * minsin, y[1] + v * maxsin
def thetawindow(theta, v, omega):
    return theta[0] + (1/L) * v * np.sin(omega), theta[1] + (1/L) * v*np.sin(omega)

outorder = {0: 'xnext', 1: 'ynext', 2: 'thetanext'}

"""
Declare modules
"""

mgr = aigerwrapper()

# Declare continuous state spaces
pspace      = DynamicCover(-2,2)
anglespace  = DynamicCover(-np.pi, np.pi, periodic=True)
# Declare discrete control spaces
vspace      = EmbeddedGrid(2, vmax/2, vmax)
angaccspace = EmbeddedGrid(3, -1.5, 1.5)


# Declare modules
dubins_x        = Interface(mgr, {'x': pspace, 'theta': anglespace, 'v': vspace}, 
                                      {'xnext': pspace})
dubins_y        = Interface(mgr, {'y': pspace, 'theta': anglespace, 'v': vspace}, 
                                      {'ynext': pspace})
dubins_theta    = Interface(mgr, {'theta': anglespace, 'v': vspace, 'omega': angaccspace}, 
                                      {'thetanext': anglespace})

dubins = (dubins_x * dubins_y * dubins_theta)
composite = CompositeModule([dubins_x, dubins_y, dubins_theta])

bits = 7
precision = {'x': bits, 'y':bits, 'theta': bits,
             'xnext': bits, 'ynext': bits, 'thetanext': bits}
bittotal = sum(precision.values()) + 3 # +3 for the discrete embedded grid bits

coarseiter = 0
coarse_errors = {'x': 0, 'y': 0, 'theta': 0}
abs_starttime = time.time()


# Sample generator
random_errors = {'x': 0, 'y': 0, 'theta': 0}
np.random.seed(1337)
for numapplied in range(5):

    print("Iteration: {}".format(numapplied))

    # Shrink window widths over time
    scale = 1/np.log10(1.0*numapplied+10)

    # Generate random input windows and points
    f_width = {'x':     .05*pspace.width(),
               'y':     .05*pspace.width(),
               'theta': .05*anglespace.width()}
    f_left = {'x':     pspace.lb + np.random.rand() * (pspace.width() - f_width['x']),
              'y':     pspace.lb + np.random.rand() * (pspace.width() - f_width['y']),
              'theta': anglespace.lb + np.random.rand() * (anglespace.width())}
    f_right = {k: f_width[k] + f_left[k] for k in f_width}
    iobox = {'v':     np.random.randint(1, 3) * vmax/2,
             'omega': np.random.randint(-1, 2) * 1.5}
    iobox.update({k: (f_left[k], f_right[k]) for k in f_width})

    # Generate output windows
    iobox['xnext']     = xwindow(**{k: v for k, v in iobox.items() if k in dubins_x.inputs})
    iobox['ynext']     = ywindow(**{k: v for k, v in iobox.items() if k in dubins_y.inputs})
    iobox['thetanext'] = thetawindow(**{k: v for k, v in iobox.items() if k in dubins_theta.inputs})

    composite = composite.io_refined(iobox, nbits=precision)

    for comp in composite.children:
        comp.pred.aag = BoolExpr(aig=simplify(comp.pred.aag.aig))

    # for comp in composite.children:
    #     fname = [i for i in comp.outputs.keys()][0]
    #     comp.pred.aag.aig.write("{}_pred_{}.aag".format(fname, numapplied))

print("Abstraction Time: ", time.time() - abs_starttime)
# composite.check()

# for comp in composite.children:
#     fname = [i for i in comp.outputs.keys()][0]
#     comp.pred.aag.aig.write("{}.aag".format(fname))
