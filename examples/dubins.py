r"""
Dubins vehicle example script
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
from redax.utils.heuristics import order_heuristic

"""
Specify dynamics and overapproximations
"""

L= 1.4
vmax = .5
def dynamics(x, y, theta, v, omega):
    return x + v*np.cos(theta), y + v*np.sin(theta), theta + (1/L) * v * np.sin(omega)

# Overapproximations of next states for each component
# v and omega are points and not windows
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
    theta: Tuple(float, float)
    v: float
    omega: float

    Returns
    -------
    Tuple(float, float)
        Overapproximation of possible values of FIXME:

    """
    return theta[0] + (1/L) * v * np.sin(omega), theta[1] + (1/L) * v*np.sin(omega)

outorder = {0: 'xnext', 1: 'ynext', 2: 'thetanext'}

"""
Initialize binary circuit manager
"""
mgr = BDD()
mgr.configure(reordering=False)

"""
Declare spaces and types
"""
# Declare continuous state spaces
pspace      = DynamicCover(-2, 2)
anglespace  = DynamicCover(-np.pi, np.pi, periodic=True)
# Declare discrete control spaces
vspace      = EmbeddedGrid(2, vmax/2, vmax)
angaccspace = EmbeddedGrid(3, -1.5, 1.5)


"""
Declare modules
"""
dubins_x        = Interface(mgr, {'x': pspace, 'theta': anglespace, 'v': vspace},
                                 {'xnext': pspace})
dubins_y        = Interface(mgr, {'y': pspace, 'theta': anglespace, 'v': vspace},
                                 {'ynext': pspace})
dubins_theta    = Interface(mgr, {'theta': anglespace, 'v': vspace, 'omega': angaccspace},
                                 {'thetanext': anglespace})

composite = CompositeModule([dubins_x, dubins_y, dubins_theta])

"""
Abstract the continuous dynamics with randomly generated boxes
"""
bits = 7
precision = {'x': bits, 'y':bits, 'theta': bits,
             'xnext': bits, 'ynext': bits, 'thetanext': bits}
abs_starttime = time.time()
np.random.seed(1337)
for numapplied in range(20000):

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
    iobox['xnext']     = xwindow(iobox['x'], iobox['v'], iobox['theta'])
    iobox['ynext']     = ywindow(iobox['y'], iobox['v'], iobox['theta'])
    iobox['thetanext'] = thetawindow(iobox['theta'], iobox['v'], iobox['omega'])

    # Refine abstraction with granularity specified in the precision variable
    composite = composite.io_refined(iobox, nbits=precision)

    if numapplied > 300000000000:
        target =  pspace.conc2pred(mgr, 'x',  [-.4, .4], 6, innerapprox=False)
        target &= pspace.conc2pred(mgr, 'y', [-.4,.4], 6, innerapprox=False)
        targetmod = Interface(mgr, {'x': pspace, 'y': pspace, 'theta': anglespace}, {}, guar=mgr.true, assum=target)
        targetmod.check()

        dcpre = DecompCPre(composite, (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')), ('v', 'omega'))
        dgame = ReachGame(dcpre, targetmod)
        dstarttime = time.time()
        basin, steps, controller = dgame.run(verbose=False)
        print("Reach Size:", basin.count_nb(3 * bits))
        print("Decomp Solve Time:", time.time() - dstarttime)
        # plot3D_QT(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),  basin.pred, 60)
        # plot3D(mgr, ('x', pspace),('y', pspace), ('theta', anglespace), basin.coarsened(x=6, y=6, theta=6).pred, view=(30, -144), fname="dubinsbasin{}.png".format(numapplied))



print("Abstraction Time: ", time.time() - abs_starttime)
composite.check()

# Heuristic used to reduce the size or "compress" the abstraction representation.
mgr.reorder(order_heuristic(mgr))

"""
Controller Synthesis with a reach objective
"""
# Declare reach set as [0.8] x [-.8, 0] box in the x-y space.
target =  pspace.conc2pred(mgr, 'x',  [-.4, .4], 6, innerapprox=False)
target &= pspace.conc2pred(mgr, 'y', [-.4,.4], 6, innerapprox=False)
targetmod = Interface(mgr, {'x': pspace, 'y': pspace, 'theta': anglespace}, {}, guar=mgr.true, assum=target)
targetmod.check()

# Two algorithms for synthesizing the controller
decomposed_synthesis = False
if decomposed_synthesis:
    # Synthesize using a decomposed model that never recombines multiple components together.
    # Typically more efficient than the monolithic case
    dcpre = DecompCPre(composite, (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')), ('v', 'omega'))
    dgame = ReachGame(dcpre, targetmod)
    dstarttime = time.time()
    basin, steps, controller = dgame.run(verbose=False)
    print("Decomp Solve Time:", time.time() - dstarttime)
else:
    # Synthesize using a monolithic model that is the parallel composition of components
    monostart = time.time()
    dubins = composite.children[0] * composite.children[1] * composite.children[2]
    print("Monolithic Construction Time:", time.time() - monostart)
    cpre = ControlPre(dubins, (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')), ('v', 'omega'))
    game = ReachGame(cpre, targetmod)
    starttime = time.time()
    basin, steps, controller = game.run(verbose=False)
    print("Monolithic Solve Time:", time.time() - starttime)


# Print statistics about reachability basin
print("Reach Size:", basin.count_nb(3 * bits))
print("Target Size:", targetmod.count_nb(3 * bits))
print("Game Steps:", steps)


"""
Plotting and visualization
"""

# # Plot reachable winning set
# plot3D_QT(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),  basin.pred, 60)
# plot3D(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),  basin.pred, view=(30, -144), fname="finedubinbasin")

# # Plot x transition relation for v = .5
# xdyn = mgr.exist(['v_0'],(composite.children[0].pred) & mgr.var('v_0'))
# plot3D_QT(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), xdyn, 128)

# plot3D_QT(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), xdyn, 128)

# for i in [3,4,5,6,7]:
#     xdyn = mgr.exist(['v_0'],(composite.children[0].coarsened(x=i, theta=i, xnext=i).pred) & mgr.var('v_0'))
#     plot3D(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), xdyn, view=(30, -144), fname="xcomp{}.png".format(i))



# # Plot y transition relation for v = .5
# ydyn = mgr.exist(['v_0'],(composite.children[1].pred) & mgr.var('v_0'))
# plot3D_QT(mgr, ('y', pspace),('theta', anglespace), ('ynext', pspace), ydyn, 128)

# # Plot theta component
# thetadyn = mgr.exist(['v_0', 'omega_0', 'omega_1'],(composite.children[2].pred) & mgr.var('v_0') & mgr.var('omega_0') & ~mgr.var('omega_1') )
# pixel2D(mgr, ('theta', anglespace), ('thetanext', anglespace), thetadyn, 128)


"""
Simulate trajectories
"""
state = {'x': -1, 'y': .6, 'theta': -.2}
for step in range(10):
    print(step, state)

    # Sample a valid safe control input from the controller.allows(state) iterator
    u = fn.first(controller.allows(state))
    if u is None:
        print("No more valid control inputs. Terminating simulation.")
        break
    state.update(u)

    # Iterate dynamics
    nextstate = dynamics(**state)

    state = {'x': nextstate[0],
             'y': nextstate[1],
             'theta': nextstate[2]}

