import os
import time
import sys

import numpy as np

from redax.predicates.dd import BDD
from redax.module import Interface, CompositeInterface
from redax.spaces import DynamicCover, EmbeddedGrid, DiscreteSet, OutOfDomainError
from redax.synthesis import SafetyGame, DecompCPre, ReachGame, PruningCPre
from redax.controllers import MemorylessController
from redax.visualizer import scatter2D, plot3D, plot3D_QT, pixel2D
from redax.ops import ihide

from dynamics import lander_box_dynamics, plot_io_bounds

directory = os.path.dirname(os.path.abspath(__file__))

T = 9
np.random.seed(1337)

# Abstract
precision = {'x': 8,     'y': 8,     'vx': 6,     'vy': 6,     'theta': 5,     'omega': 4,
             'xnext': 8, 'ynext': 8, 'vxnext': 6, 'vynext': 6, 'thetanext': 5, 'omeganext': 4}

statebits = sum([precision[s] for s in ['x', 'y', 'vx', 'vy', 'theta', 'omega']])

states = ['x', 'y', 'vx', 'vy', 'theta', 'omega']
nextstates = [i+'next' for i in states]

# Helper for substituting different precision from the default.
def replace(default, replacement):
    return {k: replacement[k] if k in replacement else default[k]
                                for k, v in default.items()
            }

# Each component depends on certain variables with different precision.
sysview = {'x': replace(precision, {'vx':6, 'theta': 4}),
           'y': replace(precision, {'vy':6, 'theta': 4}),
           'vx': replace(precision, {'theta':5, 'omega':4}),
           'vy': replace(precision, {'theta':5, 'omega':4}),
           'theta': precision,
           'omega': precision}

xspace = DynamicCover(-1, 1)
yspace = DynamicCover(0, 1.3)
vxspace = DynamicCover(-1, 1)
vyspace = DynamicCover(-1, 1)
thetaspace  = DynamicCover(-np.pi/5, np.pi/5, periodic=False)
omegaspace = DynamicCover(-.6, .6)
thrust = EmbeddedGrid(4, -.01, .99)
side = EmbeddedGrid(4, -.51, .51)

def shiftbox(box, fraction=0.0):
    width = box[1] - box[0]
    return box[0] + fraction*width, box[1] + fraction*width

def bloatbox(box, factor = .001):
    assert factor > -1.0
    center = box[0] + .5 * (box[1]-box[0])
    return box[0] - factor*(center - box[0]), box[1] + factor*(box[1] - center)

def _name(i):
    return i.split('_')[0]


def _idx(i):
    return i.split('_')[1]

def nd_ratio(mod):
    iobits = len(mod.pred.support)
    nbbits = len(mod.nonblock().support)

    if mod.count_nb(nbbits) == 0:
        return np.inf

    return mod.count_io(iobits) / mod.count_nb(nbbits)

    # print("System", i, "predicate nodes", len(subsys[i].pred), "ND ratio:", ratio)
    # print("\n")


def setup(init=False):

    mgr = BDD()
    mgr.configure(reordering=False)  # Reordering causes too much time overhead for minimal gain.
    subsys = {}
    subsys['x'] = Interface(mgr, {'x': xspace, 'vx': vxspace, 'theta': thetaspace, 't': thrust},
                                      {'xnext': xspace})
    subsys['y'] = Interface(mgr, {'y': yspace, 'vy': vyspace, 'theta': thetaspace, 't': thrust},
                                      {'ynext': yspace})
    subsys['vx'] = Interface(mgr, {'vx': vxspace, 'theta': thetaspace, 'omega': omegaspace, 't': thrust, 's': side},
                                       {'vxnext': vxspace})
    subsys['vy'] = Interface(mgr, {'vy': vyspace, 'theta': thetaspace, 'omega': omegaspace, 't': thrust, 's': side},
                                       {'vynext': vyspace})
    subsys['theta'] = Interface(mgr, {'theta': thetaspace, 'omega': omegaspace, 's': side},
                                          {'thetanext': thetaspace})
    subsys['omega'] = Interface(mgr, {'omega': omegaspace, 's': side},
                                          {'omeganext': omegaspace})

    # Coarse, but exhaustive abstraction of submodules.
    if init:
        iter_coarseness = {'x': {'x':7, 'vx': 5, 'theta': 2},
                           'y': {'y':7, 'vy': 5, 'theta': 2},
                           'vx':  {'vx': 5, 'theta': 4, 'omega': 3},
                           'vy': {'vy': 5, 'theta': 4,'omega': 3},
                           'theta': {'theta': 5, 'omega': 4},
                           'omega': {'omega': 4}
                        }

        initabs_start = time.time()
        for i in states:
            print("Refining", i, "module")
            print("Iter Coarseness:", iter_coarseness[i])
            print("Recording Coarseness:", {k: v for k, v in sysview[i].items() if k in subsys[i].vars})
            subsys[i] = coarse_abstract(subsys[i], iter_coarseness[i], sysview[i])
            print(len(mgr), "manager nodes after first pass")
            subsys[i].check()
            print("Subsys", i, "ND ratio:", nd_ratio(subsys[i]))

            subsys[i] = coarse_abstract(subsys[i], iter_coarseness[i], sysview[i], shift_frac=0.5)
            print(len(mgr), "manager nodes after second pass")
            subsys[i].check()
            print("Subsys", i, "ND ratio:", nd_ratio(subsys[i]), "\n")

            mgr.reorder(order_heuristic(mgr, ['x', 'y', 'vx', 'vy', 'theta', 'omega', 't', 's', 'xnext', 'ynext', 'vxnext', 'vynext', 'thetanext', 'omeganext']))

        print("Coarse Initial Abstraction Time (s): ", time.time() - initabs_start)

    return mgr, subsys


def coarse_abstract(f: Interface, iter_coarseness, save_precision, shift_frac = 0.0):
    """
    iter_coarseness:
        How many bits along each input dimension
    save_precision:
        Bits to record in the underlying grid
    shift:
        Shift boxes along each dimension by this fraction. Useful for the offset grid iterator.
    """
    default_boxes = {'x': (-.01, .01), 'y': (.49, .51), 'vx': (-.01,.01), 'vy': (-.01,.01), 'theta': (-.01,.01), 'omega': (-.01,.01), 't': -.01, 's':.25}
    iter = f.input_iter(precision=iter_coarseness)
    for iobox in iter:
        # Lift missing arguments to a full dimension with default_boxes.
        for arg in default_boxes:
            if arg not in iobox:
                iobox[arg] = default_boxes[arg]

        # Bloat inputs for numerical precision, just in case.
        for k, v in iobox.items():
            if isinstance(v, tuple):
                iobox[k] = bloatbox(shiftbox(v, shift_frac), .001)

        # Simulate and overapproximate outputs
        stateboxes = {k: v for k, v in iobox.items() if k not in  ['s', 't']}
        s = lander_box_dynamics(steps=T, a=(iobox['t'], iobox['s']), **stateboxes, discrete=False)
        out = {i: bloatbox(j, factor=-0.01) for i, j in zip(nextstates, s)}  # Shrink output box ever so slightly.
        iobox.update(out)

        # Refine
        iobox = {k: v for k, v in iobox.items() if k in f.vars}  # Filter unnecessary input/output slices.
        f = f.io_refined(iobox, nbits=save_precision)

    f.check()
    print("Done coarse abs: ", f.outputs)

    return f


def synthesize_reach(f: CompositeInterface, target: Interface, steps=None):

    elimorder = [i for i in nextstates]
    # elimorder = ['vxnext', 'xnext', 'vynext', 'ynext', 'thetanext', 'omeganext']
    # elimorder = ['xnext', 'vxnext', 'ynext', 'vynext', 'thetanext', 'omeganext']
    print("Variable Elimination Order (right to left): ", elimorder)
    cpre = PruningCPre(f, (('x', 'xnext'), ('y', 'ynext'), ('vx', 'vxnext'), ('vy', 'vynext'), ('theta', 'thetanext'), ('omega', 'omeganext')),
                         ('t', 's'))

    print("Solving Reach Game")
    game = ReachGame(cpre, target)
    game_starttime = time.time()
    basin, steps, controller = game.run(verbose=True,
                                        steps=steps)
    print("Solve time: ", time.time() - game_starttime)
    print("Solve steps: ", steps)
    print("Trivial region: ", basin == target)
    print("Target Size:", target.count_nb(statebits))
    print("Basin Size:", basin.count_nb(statebits))
    print("Basin Growth: ", basin.count_nb(statebits) / target.count_nb(statebits))

    return controller, steps

def simulate(controller: MemorylessController, exclude=None, drop=0):

    from lunar_lander import LunarLanderContinuous
    env = LunarLanderContinuous()
    env.seed(1337)

    try:

        # Simulate and control
        import funcy as fn

        state = fn.first(fn.drop(drop, controller.winning_states(exclude=exclude)))
        state = {k: .5*(v[0] + v[1]) for k, v in state.items()}
        env.reset([state[v] for v in states])

        statehist = None
        for step in range(20):
            u = fn.first(controller.allows(state))
            if u is None:
                print("No safe control inputs for state {}".format(state))
                env.close()
                f.children[0].mgr.configure(reordering=False)  # Needed because mgr.pick_iter toggles reordering to True
                return statehist
            picked_u = {'t': u['t'], 's': u['s']}

            state.update(picked_u)
            print("Step: ", step, state)

            for _ in range(T):
                s, _, _, _ = env.step(np.array([picked_u['t'], picked_u['s']]))
                env.render()
                time.sleep(.03)

            if step == 0:
                statehist = s
            else:
                statehist = np.vstack([statehist, s])

            state = {v: s[idx] for idx, v in enumerate(states)}

    except OutOfDomainError:
        print("Out of system domain")
        env.close()
        return statehist

    except:
        env.close()
        raise

    env.close()
    return statehist

def order_heuristic(mgr, var_priority = None):
    """
    Most signifiant bits are ordered first in BDD.

    var_priority is a list of variable names. Resolves ties if two bits, are of the same priority, it resolves based.
    e.g. var_priority = ['x','y','z'] would impose an order ['x_0', 'y_0', 'z_0']
    """

    def _name(i):
        return i.split('_')[0]

    def _idx(i):
        return int(i.split('_')[1])

    vars = list(mgr.vars)

    max_granularity = max([_idx(i) for i in vars])
    order_seed = []
    for i in range(max_granularity + 1):
        level_bits = [v for v in vars if _idx(v) == i]
        if var_priority is not None:
            level_bits.sort(key = lambda x: {k:v for v, k in enumerate(var_priority)}[ _name(x)])
        order_seed.extend(level_bits)

    return {var: idx for idx, var in enumerate(order_seed)}

if __name__ == "__main__":

    # Run setup only on startup or if user sets -c cleanse option.
    # TODO: Fix these options flags.
    init = True
    try:

        # Load previous
        if "-l" in sys.argv:
            # Setup w/ initialized variables
            mgr, subsys = setup(init=False)
            for var, bits in precision.items():
                for b in range(bits):
                    mgr.declare(var + '_' + str(b))
            mgr.declare('s_0', 's_1', 't_0', 't_1')

            for f in subsys:
                print("Loading: {0}/subsys/{1}.dddmp".format(directory, f))
                subsys[f]._assum = mgr.load("{0}/subsys/{1}_assum.dddmp".format(directory, f))
                subsys[f]._guar = mgr.load("{0}/subsys/{1}_guar.dddmp".format(directory, f))
            mgr.reorder(order_heuristic(mgr, ['x', 'y', 'vx', 'vy', 'theta', 'omega', 't', 's', 'xnext', 'ynext', 'vxnext', 'vynext', 'thetanext', 'omeganext']))
            mgr.configure(reordering=False)

        if "-c" in sys.argv:
            mgr, subsys = setup(init=True)
            abs_iters = 0

        subsys
        mgr

    except NameError: # Initiate startup and initialization steps
        mgr, subsys = setup(init=True)
        abs_iters = 0

        # Save abstractions to files
        for f in subsys:
            mgr._dump_dddmp(subsys[f].assum, "{0}/subsys/{1}_assum.dddmp".format(directory, f))
            mgr._dump_dddmp(subsys[f].guar, "{0}/subsys/{1}_guar.dddmp".format(directory, f))

    epoch = 0

    # Reach Game
    if True:
        f = CompositeInterface(tuple(subsys[i] for i in states))
        controller = None
        target = f['x'].conc2pred(mgr, 'x', (-.1, .1), 7, innerapprox=True)
        target &= f['y'].conc2pred(mgr, 'y', (1.2, 1.28), 7, innerapprox=False)
        target &= f['vy'].conc2pred(mgr, 'vy', (-.8, .1), 6, innerapprox=True)
        target &= f['theta'].conc2pred(mgr, 'theta', (-.15, .15), 5, innerapprox=True)
        # target &= f['y'].conc2pred(mgr, 'y', (.05, .15), 7, innerapprox=False)
        # target &= f['vy'].conc2pred(mgr, 'vy', (-.1, .3), 6, innerapprox=True)

        print("Target Size:", mgr.count(target, statebits))
        target = Interface(mgr,
                           {k:v for k,v in f.inputs.items() if k in states},
                           {},
                           guar=mgr.true,
                           assum=target)

        w = target
        c = mgr.false
        c = Interface(mgr,
                      {k:v for k,v in f.inputs.items() if k in states or k in ('t','s')},
                      {},
                      guar=mgr.true,
                      assum=mgr.false)

        iterreach_start = time.time()
        for gameiter in range(0,7):

            print("\nStep:", gameiter)

            # Monitoring the set of states in the lunar lander reach basin
            # elimvars = [v for v in target.pred.support if _name(v) not in ['x', 'y']]
            # pwin = mgr.exist(elimvars, w.pred)
            pwin = ihide(w, ['theta', 'vx', 'vy', 'omega'])
            print("XY Proj states:", mgr.count(pwin.pred, precision['x'] + precision['y']))
            print("XY proj target:", ihide(target, ['theta', 'vx', 'vy', 'omega']).count_nb(precision['x'] + precision['y']))
            pixel2D(mgr, ('x', subsys['x']['x']),
                         ('y', subsys['y']['y']),
                         pwin.pred,
                         70,
                         fname=directory+"/figs/lunarbasin_withxycoarsen_{0}".format(gameiter)
                    )
            del pwin

            # Decrease complexity of winning set by eliminating a least significant bit.
            while (len(w.pred) > (gameiter*.4+4)*10**5):
                coarsenbits = {k: max([_idx(n)  for n in w.pred.support if _name(n) == k], default=None) for k in states}# if k not in ['x','y']}
                coarsenbits = [k + "_" + v for k, v in coarsenbits.items() if v is not None]
                coarsenbits.sort(key=lambda x: mgr.count(mgr.forall([x], w.pred), statebits), reverse=True) # Want to shrink the least, while simplifying
                print("Coarsening with", coarsenbits[0])
                print("Default num states", mgr.count(w.pred, statebits), "node complexity ", len(w.pred))
                w._assum = mgr.forall([coarsenbits[0]], w.pred)
                print("After elim", coarsenbits[0], "num states", mgr.count(w.pred, statebits), "node complexity", len(w.pred))
                c._assum &= w.assum

            pwin = ihide(w, ['theta', 'vx', 'vy', 'omega'])
            print("XY Proj states after coarsening:",pwin.count_nb(precision['x'] + precision['y']))
            del pwin


            del controller
            controller, steps = synthesize_reach(f, w + target, steps=1)
            c._assum = c.assum | (controller.C.assum & (~controller.cpre.elimcontrol(c.assum)))  # Add new inputs

            w = ihide(controller.C, ['s','t'])
            print("Controller nodes: {0}".format(len(c.pred)))
            print("Basin nodes: {0}".format(len(w.pred)))

        controller.C = c


        print("Iter Reach Time: {}".format(time.time() - iterreach_start))

        exclude = f['x'].conc2pred(mgr, 'x', (-.13, .13), 7, innerapprox=True)
        exclude = exclude & f['y'].conc2pred(mgr, 'y', (.9,1.29), 7, innerapprox=False)
        exclude = Interface(mgr,
                           {k:v for k,v in f.inputs.items() if k in states},
                           {},
                           guar=mgr.true,
                           assum=target.pred | exclude)
        simulate(controller, exclude=exclude)


    if False:  # Plotting dynamics components

        # # Plot omega component
        side_bdd = side.conc2pred(mgr, 's', -.51 + 1.02/3.0 * 0)
        elimvars = side_bdd.support
        pred = mgr.exist(elimvars, subsys['omega'].pred & side_bdd)
        pixel2D(mgr, ('omega', subsys['omega']['omega']),
                    ('omeganext', subsys['omega']['omeganext']),
                    pred , 70)

        # # Plot theta component
        side_bdd = side.conc2pred(mgr, 's', -.51 * 1.02/3.0 * 1)
        elimvars = side_bdd.support
        pred = mgr.exist(elimvars, subsys['theta'].pred & side_bdd)
        plot3D_QT(mgr, ('theta', subsys['theta']['theta']),
                    ('omega', subsys['theta']['omega']),
                    ('thetanext', subsys['theta']['thetanext']),
                    pred , 70)

        # # Plot vy component
        side_bdd = side.conc2pred(mgr, 's', -.51 + 1.02/3.0 * 0)
        thrust_bdd = thrust.conc2pred(mgr, 't', -.01 + 1/3.0 * 3)
        omega_bdd = omegaspace.conc2pred(mgr, 'omega', (0.1, 0.2), 5, innerapprox=False)
        elimvars = (side_bdd & thrust_bdd & omega_bdd).support
        pred = mgr.exist(elimvars, subsys['vy'].pred & side_bdd & thrust_bdd & omega_bdd)
        plot3D_QT(mgr, ('vy', subsys['vy']['vy']),
                    ('theta', subsys['vy']['theta']),
                    ('vynext', subsys['vy']['vynext']),
                    pred , 70)

        # # # Plot vx component wrt vx, theta
        side_bdd = side.conc2pred(mgr, 's', -.51 + 1.02/3.0 * 2)
        thrust_bdd = thrust.conc2pred(mgr, 't', -.01 + 1/3.0 * 3)
        omega_bdd = omegaspace.conc2pred(mgr, 'omega', (.01, .02), 5, innerapprox=False)
        elimvars = (side_bdd & thrust_bdd & omega_bdd).support
        pred = mgr.exist(elimvars, subsys['vx'].pred & side_bdd & thrust_bdd & omega_bdd)
        plot3D_QT(mgr, ('vx', subsys['vx']['vx']),
                    ('theta', subsys['vx']['theta']),
                    ('vxnext', subsys['vx']['vxnext']),
                    pred , 70)

        # for i in ['vx', 'vy']:
        #     for omega, theta in [(4,5),  (4,4), (3,5), (3,4)]:
        #         csys = subsys[i].coarsened(omega=omega, theta=theta)
        #         iobits = len(csys.pred.support)
        #         nbbits = len(csys.nonblock().support)
        #         ratio = csys.count_io(iobits) / csys.count_nb(nbbits)
        #         print("System", i , "granularity", (omega,theta) , "predicate nodes", len(csys.pred), "ND ratio:", ratio)

        # for i in ['x', 'y']:
        #     for theta in [3, 4]:
        #         csys = subsys[i].coarsened(theta=theta)
        #         iobits = len(csys.pred.support)
        #         nbbits = len(csys.nonblock().support)
        #         ratio = csys.count_io(iobits) / csys.count_nb(nbbits)
        #         print("System", i , "granularity", (theta) , "predicate nodes", len(csys.pred), "ND ratio:", ratio)

        # # # Plot x component wrt x, vx. Ideally should see some dependence on vx.
        side_bdd = side.conc2pred(mgr, 's', -.51 + 1.02/3.0 * 1)
        thrust_bdd = thrust.conc2pred(mgr, 't', -.01 + 1/3.0 * 0)
        # theta_bdd = thetaspace.conc2pred(mgr, 'theta', (-.35, -.30), 5, innerapprox=False)
        theta_bdd = thetaspace.conc2pred(mgr, 'theta', (.26, .27), 3, innerapprox=False)
        elimvars = (side_bdd & thrust_bdd & theta_bdd).support
        pred = mgr.exist(elimvars, subsys['x'].pred & side_bdd & thrust_bdd & theta_bdd)# & omega_bdd)
        # plot3D_QT(mgr, ('x', subsys['x']['x']),
        #             ('vx', subsys['x']['vx']),
        #             ('xnext', subsys['x']['xnext']),
        #             pred , 70)

        # vx_bdd = vxspace.conc2pred(mgr, 'vx', (.89, .9), 6, innerapprox=False)
        # pred = mgr.exist([i for i in pred.support if _name(i) == 'vx'], pred & vx_bdd)

        x_bdd = vxspace.conc2pred(mgr, 'x', (.001, .0015), 8, innerapprox=False)
        pred = mgr.exist(x_bdd.support, pred & x_bdd)
        pixel2D(mgr, ('vx', subsys['x']['vx']),
                    ('xnext', subsys['x']['xnext']),
                    pred, 70)

