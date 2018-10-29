r"""
Dubins vehicle example.
"""

import time

import numpy as np
import funcy as fn


from redax.module import Interface, CompositeModule
from redax.spaces import DynamicCover, EmbeddedGrid, FixedCover
from redax.synthesis import ReachGame, ControlPre, DecompCPre
from redax.visualizer import plot3D, plot3D_QT, pixel2D
from redax.utils.overapprox import maxmincos, maxminsin
from redax.predicates.dd import BDD


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

mgr = BDD()
mgr.configure(reordering=False)

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
possible_transitions = dubins.count_io_space(bittotal)

coarseiter = 0
coarse_errors = {'x': 0, 'y': 0, 'theta': 0}
abs_starttime = time.time()
# # Coarse but exhaustive iteration over the input space.
# for iobox in dubins.input_iter(precision={'x': 4, 'y': 4, 'theta': 3}):
#     # Generate output windows
#     iobox['xnext']     = xwindow(**{k: v for k, v in iobox.items() if k in dubins_x.inputs})
#     iobox['ynext']     = ywindow(**{k: v for k, v in iobox.items() if k in dubins_y.inputs})
#     iobox['thetanext'] = thetawindow(**{k: v for k, v in iobox.items() if k in dubins_theta.inputs})

#     composite = composite.io_refined(iobox, nbits=precision)


# Sample generator
random_errors = {'x': 0, 'y': 0, 'theta': 0}
np.random.seed(1337)
for numapplied in range(10000):

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


print("Abstraction Time: ", time.time() - abs_starttime)
composite.check()

from redax.utils.heuristics import order_heuristic
mgr.reorder(order_heuristic(mgr))

# Declare reach set as [0.8] x [-.8, 0] box
target =  pspace.conc2pred(mgr, 'x',  [-.4, .4], 8, innerapprox=False)
target &= pspace.conc2pred(mgr, 'y', [-.4,.4], 8, innerapprox=False)

# dubins = composite.children[0] * composite.children[1] * composite.children[2]
# cpre = ControlPre(dubins, (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')), ('v', 'omega'))
# game = ReachGame(cpre, target)
# starttime = time.time()
# basin, steps, controller = game.run(verbose=False)
# print("Monolithic Solve Time:", time.time() - starttime)

# dcpre = DecompCPre(composite, (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')), ('v', 'omega'))
# dgame = ReachGame(dcpre, target)
# dstarttime = time.time()
# dbasin, steps, controller = dgame.run(verbose=False)
# print("Decomp Solve Time:", time.time() - dstarttime)
# # assert dbasin == basin
# basin = dbasin

# print("Reach Size:", dubins.mgr.count(basin, len(basin.support)))
# print("Target Size:", dubins.mgr.count(target, len(basin.support)))
# print("Game Steps:", steps)

# # Plot reachable winning set
# plot3D_QT(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),  basin , 60)

# # Plot x transition relation for v = .5
# xdyn = mgr.exist(['v_0'],(composite.children[0].pred) & mgr.var('v_0'))
# plot3D_QT(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), xdyn, 128)

# # Plot y transition relation for v = .5
# ydyn = mgr.exist(['v_0'],(composite.children[1].pred) & mgr.var('v_0'))
# plot3D_QT(mgr, ('y', pspace),('theta', anglespace), ('ynext', pspace), ydyn, 128)

# # Plot theta component
# thetadyn = mgr.exist(['v_0', 'omega_0', 'omega_1'],(composite.children[2].pred) & mgr.var('v_0') & mgr.var('omega_0') & ~mgr.var('omega_1') )
# pixel2D(mgr, ('theta', anglespace), ('thetanext', anglespace), thetadyn, 128)


# """Simulate"""
# state = {'x': -1, 'y': .6, 'theta': -.2}
# for step in range(10):
#     print(step, state)

#     u = fn.first(controller.allows(state))
#     if u is None:
#         break
#     # u = [i for i in controller.allows(state)]
#     # if len(u) == 0:
#     #     break
#     # u = u[0]
#     state.update(u)
#     nextstate = dynamics(**state)
#     state = {'x': nextstate[0], 'y': nextstate[1], 'theta': nextstate[2]}


"""
Test code for moving to a case where everything is an interface. 
"""

targetmod = Interface(mgr, {'x': pspace, 'y': pspace, 'theta': anglespace}, {})
targetmod._pred = target
targetmod._nb = target
targetmod.check()

# dubins = composite.children[0] * composite.children[1] * composite.children[2]
# cpre = ControlPre(dubins, (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')), ('v', 'omega'))
cpre = DecompCPre(composite, (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')), ('v', 'omega'))

# assert cpre.modulepre(targetmod).pred == cpre(target)

for i in range(8):
    pass
    # targetmod = cpre.modulepre(targetmod, no_inputs=True)
    # target = cpre(target, no_inputs=True)

    # assert targetmod.pred == target


from redax.ops import sinkprepend
for i in range(8):
    # pass
    targetmod = cpre.modulepre(targetmod, no_inputs=True, collapser=sinkprepend)
    # target = cpre(target, no_inputs=True)

    # assert targetmod.pred == target