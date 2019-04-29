"""
Run collision avoidance example for dubins vehicles.
"""

import time

import numpy as np

from redax.module import Interface, CompositeInterface
from redax.ops import rename, coarsen, ihide
from redax.synthesis import DecompCPre, ReachAvoidGame, ReachGame, CompConstrainedPre
from redax.visualizer import pixel2D, scatter2D, plot3D_QT, plot3D
from redax.utils.heuristics import order_heuristic, order_heuristic_vars

from dubins import setup, rand_abstract_composite, xwindow, ywindow, thetawindow, coarse_abstract

def get_safe(composite: CompositeInterface):
    mgr = composite.mgr
    pspace = composite['x']
    anglespace = composite['theta']

    width = .2

    # x coordinates are close
    x_collision = mgr.false
    for i in np.arange(-1.95, 1.8, .5*width):
        occupied_x1 =  pspace.conc2pred(mgr, 'x', [i,i+width], 6, innerapprox=False)
        occupied_x2 =  pspace.conc2pred(mgr, 'x2', [i,i+width], 6, innerapprox=False)
        x_collision |= occupied_x1 & occupied_x2

    # y coordinates are close
    y_collision = mgr.false
    for i in np.arange(-1.95, 1.8, .5*width):
        occupied_y1 = pspace.conc2pred(mgr, 'y', [i,i+width], 6, innerapprox=False)
        occupied_y2 = pspace.conc2pred(mgr, 'y2', [i,i+width], 6, innerapprox=False)
        y_collision |= occupied_y1 & occupied_y2

    # scatter2D(mgr, ('x', pspace), ('x2', pspace), x_collision)
    # scatter2D(mgr, ('y', pspace), ('y2', pspace), y_collision)

    # If both vehicles are close along x, y then there is a collision
    collision = x_collision & y_collision

    # Negate collision for safe region
    safe = Interface(mgr, {'x': pspace, 'y': pspace, 'theta': anglespace, 'x2': pspace, 'y2':pspace, 'theta2':anglespace},
                            {},
                            guar=mgr.true,
                            assum=~collision
                            )
    safe.check()

    return safe

def get_target(composite: CompositeInterface):
    mgr = composite.mgr
    pspace = composite['x']
    anglespace = composite['theta']

    target1 =  pspace.conc2pred(mgr, 'x', [-1.5,-.5], 6, innerapprox=False)
    target1 &= pspace.conc2pred(mgr, 'y', [-1.5,-.5], 6, innerapprox=False)

    # scatter2D(mgr, ('x', pspace), ('y', pspace), target1)

    target2 =  pspace.conc2pred(mgr, 'x2', [.5,1.5], 6, innerapprox=False)
    target2 &= pspace.conc2pred(mgr, 'y2', [.5,1.5], 6, innerapprox=False)

    # scatter2D(mgr, ('x2', pspace), ('y2', pspace), target2)

    target = Interface(mgr, {'x': pspace, 'y': pspace, 'theta': anglespace, 'x2': pspace, 'y2':pspace, 'theta2':anglespace},
                            {},
                            guar=mgr.true,
                            assum=target1 & target2
                            )
    target.check()

    return target

def run_reachavoid(safe, target, composite, verbose=False, steps=None, winningonly=True):

    maxnodes = 50000

    def condition(iface: Interface) ->  bool:
        """
        Checks for interface BDD complexity.
        Returns true if above threshold
        """
        if len(iface.pred) > maxnodes:
            if verbose:
                print("Basin predicate # nodes {}".format(len(iface.pred)))
            return True
        return False

    def heuristic(iface: Interface) -> Interface:
        """
        Coarsens sink interface along the dimension that shrinks the set the
        least until a certain size met.
        """
        assert iface.is_sink()

        statebits = len(iface.pred.support)

        while (len(iface.pred) > maxnodes):
            granularity = {k: len(v) for k, v in iface.pred_bitvars.items()
                                if k in ['x', 'y', 'theta', 'xnext', 'ynext', 'thetanext',
                                         'x2', 'y2', 'theta2', 'x2next', 'y2next', 'theta2next']
                         }

            # List of (varname, # of coarsened interface nonblock input assignments)
            coarsened_ifaces = [ (k, coarsen(iface, bits={k: v-1}).count_nb(statebits))
                                        for k, v in granularity.items()
                            ]
            coarsened_ifaces.sort(key = lambda x: x[1], reverse=True)
            if verbose:
                print(coarsened_ifaces)
            best_var = coarsened_ifaces[0][0]
            iface = coarsen(iface, bits={best_var: granularity[best_var]-1})
            if verbose:
                print("Coarsened along {}. now {} nodes and basin size {}".format(best_var,
                                                                                  len(iface.pred),
                                                                                  iface.count_nb(statebits)
                                                                            )
                     )

        return iface

    cpre = CompConstrainedPre(composite,
                              (('x', 'xnext'), ('x2', 'x2next'),
                              ('y', 'ynext'), ('y2', 'y2next'),
                              ('theta', 'thetanext'), ('theta2', 'theta2next')),
                              ('v', 'v2', 'omega', 'omega2'),
                              condition,
                              heuristic)
    game = ReachAvoidGame(cpre, safe, target)
    starttime = time.time()
    basin, steps, controller = game.run(steps=steps, verbose=verbose, winningonly=winningonly)
    print("Comp Constrained Solve Time:", time.time() - starttime)

    # Print statistics about reachability basin
    print("Reach Size:", basin.count_nb( len(basin.pred.support | target.pred.support)))
    print("Reach BDD nodes:", len(basin.pred))
    print("Target Size:", target.count_nb(len(basin.pred.support | target.pred.support)))
    print("Game Steps:", steps)

    return basin, controller

def plot_basin(mgr, composite, basin):
    mgr = composite.mgr
    pspace = composite['x']
    anglespace = composite['theta']

    plot3D_QT(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),    ihide(basin, {'x2', 'y2', 'theta2'}).pred, 60)
    plot3D_QT(mgr, ('x2', pspace), ('y2', pspace), ('theta2', anglespace),    ihide(basin, {'x', 'y', 'theta'}).pred, 60)

    # plot3D_QT(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace),    ihide(controller.C, {'v','v2','omega','omega2','x2', 'y2', 'theta2'}).pred, 60)
    # plot3D_QT(mgr, ('x2', pspace), ('y2', pspace), ('theta2', anglespace), ihide(controller.C, {'v','v2','omega','omega2','x', 'y', 'theta'}).pred, 60)

if __name__ is "__main__":

    mgr, dubins_x, dubins_y, dubins_theta = setup()

    # composite = CompositeInterface([dubins_x, dubins_y, dubins_theta])
    # composite = rand_abstract_composite(composite, samples=30000)
    # dubins_x, dubins_y, dubins_theta = composite.children[0], composite.children[1], composite.children[2]

    # Original dynamics
    bits = 6
    dubins_x = coarse_abstract(dubins_x, xwindow, bits=bits)
    dubins_y = coarse_abstract(dubins_y, ywindow, bits=bits)
    dubins_theta = coarse_abstract(dubins_theta, thetawindow, bits=bits)

    # Create another vehicle with duplicated dynamics
    dubins_x2 = rename(dubins_x, x='x2', v='v2', theta='theta2', xnext='x2next')
    dubins_y2 = rename(dubins_y, y='y2', v='v2', theta='theta2', ynext='y2next')
    dubins_theta2 = rename(dubins_theta, theta='theta2', omega='omega2', v='v2', thetanext='theta2next')

    # Hard code a BDD variable ordering. Higher significant bits first
    # mgr.reorder(order_heuristic(mgr))
    mgr.reorder(order_heuristic_vars(mgr, ['x', 'y', 'x2', 'y2', 'theta', 'theta2', 'v', 'v2', 'omega', 'omega2', 'xnext', 'ynext', 'x2next', 'y2next', 'thetanext', 'theta2next']) )
    mgr.configure(reordering=False)

    # # Set up collision region and reach target
    composite = CompositeInterface([dubins_theta, dubins_theta2, dubins_x, dubins_y, dubins_x2, dubins_y2])
    target = get_target(composite)
    safe   = get_safe(composite)
    # safe._assum = mgr.true # OVERRIDE SO ANYTHING IS ALLOWED. NO COLLISIONS AT ALL

    print("Target Size:", target.count_nb((bits+1)*6))

    # Set up and run game
    basin, controller = run_reachavoid(safe,
                                       target,
                                       composite,
                                       verbose=True,
                                       steps=5,
                                       winningonly=True)

    # plot_basin(mgr, composite, basin)