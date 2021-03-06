r"""
Dubins vehicle example script
"""

import time

import numpy as np
import funcy as fn

from redax.module import Interface, CompositeInterface
from redax.spaces import DynamicCover, EmbeddedGrid, FixedCover
from redax.synthesis import ReachGame, ControlPre, DecompCPre, CompConstrainedPre
from redax.visualizer import plot3D, plot3D_QT, pixel2D
from redax.utils.overapprox import maxmincos, maxminsin
from redax.predicates.dd import BDD
from redax.utils.heuristics import order_heuristic
from redax.ops import coarsen

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
    y: Tuple(float, float)
    v: float
    theta: Tuple(float, float)

    Returns
    -------
    Tuple(float, float)
        Overapproximation of possible values of y + v sin(theta)

    """
    return theta[0] + (1/L) * v * np.sin(omega), theta[1] + (1/L) * v*np.sin(omega)

def setup():
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
    Declare interfaces
    """
    dubins_x        = Interface(mgr, {'x': pspace, 'theta': anglespace, 'v': vspace},
                                     {'xnext': pspace})
    dubins_y        = Interface(mgr, {'y': pspace, 'theta': anglespace, 'v': vspace},
                                     {'ynext': pspace})
    dubins_theta    = Interface(mgr, {'theta': anglespace, 'v': vspace, 'omega': angaccspace},
                                     {'thetanext': anglespace})

    return mgr, dubins_x, dubins_y, dubins_theta


def generate_random_io(pspace, anglespace):
    # Generate random input windows and points
    f_width = {'x':     .04*pspace.width(),
                'y':     .04*pspace.width(),
                'theta': .04*anglespace.width()}
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

    return iobox

def abstract_composite(composite: CompositeInterface, samples = 10000):
    """
    Abstract the continuous dynamics with randomly generated boxes
    """
    pspace = composite['x']
    anglespace = composite['theta']
    bits = 7
    precision = {'x': bits, 'y':bits, 'theta': bits,
                'xnext': bits, 'ynext': bits, 'thetanext': bits}
    abs_starttime = time.time()
    np.random.seed(1337)
    for _ in range(samples):

        iobox = generate_random_io(pspace, anglespace)

        # Refine abstraction with granularity specified in the precision variable
        composite = composite.io_refined(iobox, nbits=precision)

    print("Abstraction Time: ", time.time() - abs_starttime)
    composite.check()

    return composite


def make_target(mgr, composite: CompositeInterface):
    """
    Controller Synthesis with a reach objective
    """

    pspace = composite['x']
    anglespace = composite['theta']
    # Declare reach set as [0.8] x [-.8, 0] box in the x-y space.
    target =  pspace.conc2pred(mgr, 'x',  [1.0,1.5], 5, innerapprox=False)
    target &= pspace.conc2pred(mgr, 'y',  [1.0,1.5], 5, innerapprox=False)
    targetmod = Interface(mgr, {'x': pspace, 'y': pspace, 'theta': anglespace}, {}, guar=mgr.true, assum=target)
    targetmod.check()

    return targetmod


def run_reach(targetmod, composite, cpretype="decomp", steps=None):
    # Three choices for the controlled predecessor used for synthesizing the controller
    # Options: decomp, monolithic, compconstr
    # cpretype = "compconstr"

    assert cpretype in ["decomp", "monolithic", "compconstr"]

    if cpretype is  "decomp":

        # Synthesize using a decomposed model that never recombines multiple components together.
        # Typically more efficient than the monolithic case
        dcpre = DecompCPre(composite, (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')), ('v', 'omega'))
        dgame = ReachGame(dcpre, targetmod)
        dstarttime = time.time()
        basin, steps, controller = dgame.run(verbose=False, steps=steps)
        print("Decomp Solve Time:", time.time() - dstarttime)

    elif cpretype is "monolithic":

        # Synthesize using a monolithic model that is the parallel composition of components
        dubins = composite.children[0] * composite.children[1] * composite.children[2]
        cpre = ControlPre(dubins, (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')), ('v', 'omega'))
        game = ReachGame(cpre, targetmod)
        starttime = time.time()
        basin, steps, controller = game.run(verbose=False, steps=steps)
        print("Monolithic Solve Time:", time.time() - starttime)

    elif cpretype is "compconstr":

        maxnodes = 5000

        def condition(iface: Interface) ->  bool:
            """
            Checks for interface BDD complexity.
            Returns true if above threshold
            """
            if len(iface.pred) > maxnodes:
                print("Interface # nodes {} exceeds maximum {}".format(len(iface.pred), maxnodes))
                return True
            return False

        def heuristic(iface: Interface) -> Interface:
            """
            Coarsens sink interface along the dimension that shrinks the set the
            least until a certain size met.
            """

            assert iface.is_sink()

            while (len(iface.pred) > maxnodes):
                granularity = {k: len(v) for k, v in iface.pred_bitvars.items()
                                    if k in ['x', 'y', 'theta', 'xnext', 'ynext', 'thetanext']
                            }
                statebits = len(iface.pred.support)

                # List of (varname, # of coarsened interface nonblock input assignments)
                coarsened_ifaces = [ (k, coarsen(iface, bits={k: v-1}).count_nb(statebits))
                                            for k, v in granularity.items()
                                ]
                coarsened_ifaces.sort(key = lambda x: x[1], reverse=True)
                best_var = coarsened_ifaces[0][0]
                # print(coarsened_ifaces)
                # print("Coarsening along dimension {}".format(best_var))
                iface = coarsen(iface, bits={best_var: granularity[best_var]-1})

            return iface


        cpre = CompConstrainedPre(composite,
                                (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')),
                                ('v', 'omega'),
                                condition,
                                heuristic)
        game = ReachGame(cpre, targetmod)
        starttime = time.time()
        basin, steps, controller = game.run(steps=steps, verbose=False)
        print("Comp Constrained Solve Time:", time.time() - starttime)


    # Print statistics about reachability basin
    print("Reach Size:", basin.count_nb( len(basin.pred.support | targetmod.pred.support)))
    print("Reach BDD nodes:", len(basin.pred))
    print("Target Size:", targetmod.count_nb(len(basin.pred.support | targetmod.pred.support)))
    print("Game Steps:", steps)

    return basin, controller

def plots(mgr, basin, composite):

    """
    Plotting and visualization
    """

    pspace = composite['x']
    anglespace = composite['theta']

    # # Plot reachable winning set
    plot3D_QT(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),  basin.pred, 60)

    # # Plot x transition relation for v = .5
    # xdyn = mgr.exist(['v_0'],(composite.children[0].pred) & mgr.var('v_0'))
    # plot3D_QT(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), xdyn, 128)

    # # Plot y transition relation for v = .5
    # ydyn = mgr.exist(['v_0'],(composite.children[1].pred) & mgr.var('v_0'))
    # plot3D_QT(mgr, ('y', pspace),('theta', anglespace), ('ynext', pspace), ydyn, 128)

    # # Plot theta component
    # thetadyn = mgr.exist(['v_0', 'omega_0', 'omega_1'],(composite.children[2].pred) & mgr.var('v_0') & mgr.var('omega_0') & ~mgr.var('omega_1') )
    # pixel2D(mgr, ('theta', anglespace), ('thetanext', anglespace), thetadyn, 128)


def simulate(controller):
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


if __name__ is "__main__":
    mgr, dubins_x, dubins_y, dubins_theta = setup()

    composite = CompositeInterface([dubins_x, dubins_y, dubins_theta])
    composite = abstract_composite(composite, samples=15000)

    # Heuristic used to reduce the size or "compress" the abstraction representation.
    # Higher significant bits first
    mgr.reorder(order_heuristic(mgr))
    mgr.configure(reordering=False)

    targetmod = make_target(mgr, composite)

    basin, controller = run_reach(targetmod, composite)

    plots(mgr, basin, composite)
    simulate(controller)