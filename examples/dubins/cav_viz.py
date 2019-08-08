r"""
Variant of the Dubins vehicle example script for CAV 2019 presentation visualizations
"""

import sys
import time
import os
directory = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import funcy as fn

from redax.module import Interface, CompositeInterface
from redax.spaces import DynamicCover, EmbeddedGrid, FixedCover
from redax.synthesis import ReachGame, ControlPre, PruningCPre, DecompCPre
from redax.visualizer import plot3D, plot3D_QT, pixel2D, scatter2D
from redax.utils.overapprox import maxmincos, maxminsin, shiftbox, bloatbox
from redax.predicates.dd import BDD
from redax.utils.heuristics import order_heuristic
from redax.ops import coarsen


"""
Specify dynamics and overapproximations
"""

### Dynamics Parameters
L= 1.4
vmax = .5
# vmax = .3
angturn = 1.5
# angturn = 1.0

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
    angaccspace = EmbeddedGrid(3, -angturn, angturn)

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

def coarse_abstract(f: Interface, concrete, bits=6):
    r"""
    Constructs an abstraction of interface f with two passes with
    overlapping input sets.

    First pass with 2^bits bins along each dimension.
    Second pass with same granuarity but shifted by .5.

    The abstraction is saved with 2^(bits+1) bins along each dimension.
    """
    iter_precision = {'x': bits, 'y':bits, 'theta': bits,
                 'xnext': bits, 'ynext': bits, 'thetanext': bits}
    save_precision = {k: v+1 for k,v in iter_precision.items()}
    outvar = fn.first(f.outputs)


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

def generate_random_io(pspace, anglespace):
    # Generate random input windows and points
    f_width = {'x':     .05*pspace.width(),
                'y':     .05*pspace.width(),
                'theta': .05*anglespace.width()}
    f_left  = {'x':     pspace.lb + np.random.rand() * (pspace.width() - f_width['x']),
                'y':     pspace.lb + np.random.rand() * (pspace.width() - f_width['y']),
                'theta': anglespace.lb + np.random.rand() * (anglespace.width())}
    f_right = {k: f_width[k] + f_left[k] for k in f_width}
    iobox   = {'v':     np.random.randint(1, 3) * vmax/2,
                'omega': np.random.randint(-1, 2) * 1.5}
    iobox.update({k: (f_left[k], f_right[k]) for k in f_width})

    # Generate output windows
    iobox['xnext']     = xwindow(iobox['x'], iobox['v'], iobox['theta'])
    iobox['ynext']     = ywindow(iobox['y'], iobox['v'], iobox['theta'])
    iobox['thetanext'] = thetawindow(iobox['theta'], iobox['v'], iobox['omega'])

    return iobox

def rand_abstract_composite(composite: CompositeInterface, samples = 10000):
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
    for i in range(samples):

        iobox = generate_random_io(pspace, anglespace)

        # Refine abstraction with granularity specified in the precision variable
        composite = composite.io_refined(iobox, nbits=precision)

        # if i % 500 == 499:
        #     xdyn = mgr.exist(['v_0'],(composite.children[0].coarsened(x=6, theta=6, xnext=6).pred) & mgr.var('v_0'))
        #     plot3D(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), xdyn, view=(30, -144), fname="xcompsamp{}".format(i+1), opacity=80)

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
    # target =  pspace.conc2pred(mgr, 'x',  [1.0,1.5], 5, innerapprox=False)
    # target &= pspace.conc2pred(mgr, 'y',  [1.0,1.5], 5, innerapprox=False)
    target =  pspace.conc2pred(mgr, 'x',  [-.4,.4], 5, innerapprox=False)
    target &= pspace.conc2pred(mgr, 'y',  [-.4,.4], 5, innerapprox=False)
    targetmod = Interface(mgr, {'x': pspace, 'y': pspace, 'theta': anglespace}, {}, guar=mgr.true, assum=target)
    targetmod.check()

    return targetmod



def run_reach(targetmod, composite, cpretype="decomp", steps=None):
    # Three choices for the controlled predecessor used for synthesizing the controller
    # Options: decomp, monolithic, compconstr
    # cpretype = "compconstr"

    assert cpretype in ["decomp", "monolithic", "compconstr"]

    prebasinsizehist = []
    prebasinnodehist = []
    basinsizehist = []
    basinnodehist = []

    if cpretype ==  "decomp":

        def printreach(Z: Interface):
            i = printreach.counter
            plot3D(mgr, ('x', composite['x']), ('y', composite['y']), ('theta', composite['theta']),
                    coarsen(Z, x=6,y=6,theta=6).pred,
                    fname="{}/rotate_dubinsreach_{}".format(directory, printreach.counter),
                    view=(35 - i , -40 - i * 4 ),
                    title = "Z_{}".format(printreach.counter)
                    )
            printreach.counter += 1
            return Z
        printreach.counter = 0

        # Synthesize using a decomposed model that never recombines multiple components together.
        # Typically more efficient than the monolithic case
        dcpre = DecompCPre(composite, (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')), ('v', 'omega'), pre_process=printreach)
        dgame = ReachGame(dcpre, targetmod)
        dstarttime = time.time()
        basin, steps, controller = dgame.run(verbose=False, steps=steps)
        print("Decomp Solve Time:", time.time() - dstarttime)

    else:
        import pdb; pdb.set_trace()

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

    def camerapan(N, vert = (30,0), horiz = (-45,-90)):
        horiz_pan = np.linspace(horiz[0], horiz[1], N)
        vert_pan = np.linspace(vert[0], vert[1], N)
        for i in range(N):
            yield (vert_pan[i], horiz_pan[i])

    pspace = composite['x']
    anglespace = composite['theta']

    target = make_target(mgr, composite)
    # # Pan target visualization
    for i, view in enumerate(camerapan(15, (89.9,35), (-89.9, -40))):
        plot3D(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace), target.pred, view=view, title="Target Visualization", fname="target{}".format(i))

    # # Plot reachable winning set
    # plot3D(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),  basin.pred, opacity=44, fname="dubins_nodes{}".format(len(basin.pred)))
    # plot3D_QT(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),  basin.pred, opacity=44)

    # # Plot x transition relation for v = .5
    # xdyn = mgr.exist(['v_0'],(composite.children[0].pred) & mgr.var('v_0'))
    # plot3D_QT(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), xdyn, opacity=128, raisebiterror=False)

    # # Plot y transition relation for v = .5
    # ydyn = mgr.exist(['v_0'],(composite.children[1].pred) & mgr.var('v_0'))
    # plot3D(mgr, ('y', pspace),('theta', anglespace), ('ynext', pspace), ydyn, opacity=128, raisebiterror=False)

    # Plot theta component
    # thetadyn = mgr.exist(['v_0', 'omega_0', 'omega_1'],(composite.children[2].pred) & mgr.var('v_0') & mgr.var('omega_0') & ~mgr.var('omega_1') )
    # scatter2D(mgr, ('theta', anglespace), ('thetanext', anglespace), thetadyn)
    # pixel2D(mgr, ('theta', anglespace), ('thetanext', anglespace), thetadyn)


    # # Plot reachable winning set
    # plot3D_QT(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),  basin.pred, 60)
    # plot3D(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),  basin.pred, view=(30, -144), fname="finedubinbasin")

    # # Coarsen x dynamics
    # for i in [3]:
    #     for j in [3]:
    #         xdyn = mgr.exist(['v_0'],(composite.children[0].coarsened(x=i, theta=i, xnext=j).pred) & mgr.var('v_0'))
    #         plot3D(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), xdyn, view=(30, -144), fname="xcomp{}_{}".format(i,j), opacity=80)
        # print("Xdyn Nodes:", len(composite.children[0].coarsened(x=i, theta=i, xnext=i).pred))

    # Plot coarsened reachable winning set
    # for i in [4,5,6,7]:
    #     plot3D(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),
    #                 coarsen(basin, x=i, y=i, theta=i).pred, view=(30, -144),
    #                 fname="basin{}".format(i),
    #                 opacity=80
    #           )
    #     print("Basin Nodes:", len(coarsen(basin, x=i, y=i, theta=i).pred))
    #     print("Basin IO:", coarsen(basin, x=i, y=i, theta=i).count_nb(21))


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


if __name__ == "__main__":

    print(sys.argv)
    mgr, dubins_x, dubins_y, dubins_theta = setup()

    if '-samp' in sys.argv:
        idx = sys.argv.index('-samp') + 1
        samples = int(sys.argv[idx])

        composite = CompositeInterface([dubins_x, dubins_y, dubins_theta])
        composite = rand_abstract_composite(composite, samples=samples)
    else:
        bits = 6
        abs_starttime = time.time()
        dubins_x = coarse_abstract(dubins_x, xwindow, bits=bits)
        dubins_y = coarse_abstract(dubins_y, ywindow, bits=bits)
        dubins_theta = coarse_abstract(dubins_theta, thetawindow, bits=bits)
        composite = CompositeInterface([dubins_x, dubins_y, dubins_theta])
        print("Abstraction Time: ", time.time() - abs_starttime)

    # Heuristic used to reduce the size or "compress" the abstraction representation.
    # Higher significant bits first
    mgr.reorder(order_heuristic(mgr))
    mgr.configure(reordering=False)

    if "-merge" in sys.argv:
        idx = sys.argv.index('-merge') + 1
        mergetype = int(sys.argv[idx])

        starttime = time.time()
        if mergetype == 0: # x y theta
            pass
        elif mergetype == 1: # xy theta
            composite = CompositeInterface([composite.children[0] * composite.children[1], composite.children[2]])
        elif mergetype == 2: # xtheta y
            composite = CompositeInterface([composite.children[0] * composite.children[2], composite.children[1]])
        elif mergetype == 3: # ytheta x
            composite = CompositeInterface([composite.children[1] * composite.children[2], composite.children[0]])
        elif mergetype == 4: # xytheta
            composite = CompositeInterface([composite.children[0] * composite.children[1] * composite.children[2]])
        else:
            raise ValueError("Invalid option")

        print("Abstraction merge time: {}s".format(time.time()-starttime))

    target = make_target(mgr, composite)

    if "-cpre" in sys.argv:
        idx = sys.argv.index('-cpre') + 1
        cpretype = sys.argv[idx]
    else:
        cpretype = "decomp"

    if "-steps" in sys.argv:
        idx = sys.argv.index('-steps') + 1
        steps = int(sys.argv[idx])
    else:
        steps = None

    # Run reach game
    basin, controller = run_reach(target, composite, steps=steps, cpretype=cpretype)


    if "-p" in sys.argv:
        plots(mgr, basin, composite)
    if "-sim" in sys.argv:
        simulate(controller)
    print("\n")


    # TEMPORARY STUFF FOR GENERATING VISUALIZATION
    # pspace = composite['x']
    # anglespace = composite['theta']
    # xdyn = mgr.exist(['v_0'], composite.children[0].pred & mgr.var('v_0'))
    # xdyn = mgr.exist(['v_0'],(composite.children[0].coarsened(x=6, theta=6, xnext=6).pred) & mgr.var('v_0'))
    # plot3D(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), xdyn, view=(30, -144), fname="xcomp_samp{}".format(samples), opacity=60)
    # plot3D_QT(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), xdyn, opacity=60, fname="asdf.png")

    # plot3D(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),  coarsen(basin, x=6,y=6,theta=6).pred, view=(30, -144), fname="dubinbasin_samp{}".format(samples), opacity=60)
