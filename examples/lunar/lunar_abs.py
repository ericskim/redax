import time

import numpy as np
try:
    from dd.cudd import BDD
except ImportError:
    from dd.autoref import BDD

import gym
from lunar_lander import LunarLander
from dynamics import lander_box_dynamics, plot_io_bounds


from redax.module import AbstractModule, CompositeModule
from redax.spaces import DynamicCover, EmbeddedGrid, FixedCover, DiscreteSet
from redax.synthesis import SafetyGame, ControlPre, DecompCPre, ReachGame
from redax.controllers import MemorylessController
from redax.visualizer import scatter2D, plot3D, plot3D_QT

T = 12

# Abstract
precision = {'x': 7,     'y': 7,     'vx': 6,     'vy': 7,     'theta': 6,     'omega': 5,
             'xnext': 7, 'ynext': 7, 'vxnext': 6, 'vynext': 7, 'thetanext': 6, 'omeganext': 5}
statebits = sum([precision[s] for s in ['x', 'y', 'vx', 'vy', 'theta', 'omega']])

states = ['x', 'y', 'vx', 'vy', 'theta', 'omega']
nextstates = [i+'next' for i in states]

# Helper for substituting different precision from the default.
def replace(default, replacement):
    return {k: replacement[k] if k in replacement else default[k]
                                for k, v in default.items()
            }

# Each component depends on certain variables with different precision.
sysview = {'x': replace(precision, {'vx':6, 'theta': 3, 'omega':3}),
           'y': replace(precision, {'vy':6, 'theta': 3, 'omega':3}),
           'vx': replace(precision, {'theta':6, 'omega':4}),
           'vy': replace(precision, {'theta':6, 'omega':4}),
           'theta': precision,
           'omega': precision}


def setup(init=False):
    xspace = DynamicCover(-.4, .4)
    yspace = DynamicCover(0, 1)
    vxspace = DynamicCover(-8, 8)
    vyspace = DynamicCover(-8, 8)
    thetaspace  = DynamicCover(-np.pi/2, np.pi/2, periodic=False)
    omegaspace = DynamicCover(-3, 3)
    actionspace = DiscreteSet(4)

    mgr = BDD()
    mgr.configure(reordering=False)  # Reordering causes too much time overhead for minimal gain.
    subsys = {}
    # subsys['x'] = AbstractModule(mgr, {'x': xspace, 'vx': vxspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'xnext': xspace})
    # subsys['y'] = AbstractModule(mgr, {'y': yspace, 'vy': vyspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'ynext': yspace})
    subsys['x'] = AbstractModule(mgr, {'x': xspace, 'vx': vxspace, 'theta': thetaspace, 'a': actionspace}, {'xnext': xspace})
    subsys['y'] = AbstractModule(mgr, {'y': yspace, 'vy': vyspace, 'theta': thetaspace, 'a': actionspace}, {'ynext': yspace})    
    subsys['vx'] = AbstractModule(mgr, {'vx': vxspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'vxnext': vxspace})
    subsys['vy'] = AbstractModule(mgr, {'vy': vyspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'vynext': vyspace})
    subsys['theta'] = AbstractModule(mgr, {'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'thetanext': thetaspace})
    subsys['omega'] = AbstractModule(mgr, {'omega': omegaspace, 'a': actionspace}, {'omeganext': omegaspace})

    # Coarse, but exhaustive abstraction of submodules.
    if init:
        iter_coarseness = {var: 4 for var in states}
        iter_coarseness = {'x': replace(iter_coarseness, {'vx':5, 'theta': 3, 'omega':3}),
                           'y': replace(iter_coarseness, {'vy':5, 'theta': 3, 'omega':3}),
                           'vx': iter_coarseness,
                           'vy': iter_coarseness,
                           'theta': replace(iter_coarseness, {'theta': 6, 'omega':6}),
                           'omega': replace(iter_coarseness, {'omega':7})}
        initabs_start = time.time()
        for i in states:
            subsys[i] = coarse_abstract(subsys[i], iter_coarseness[i], sysview[i])

        mgr.reorder(order_heuristic(mgr))
        print("Initial Abstraction Time (s): ", time.time() - initabs_start)

    return mgr, subsys

def bloat(box, factor = .001):
    assert factor > -1.0
    center = box[0] + .5 * (box[1]-box[0])
    return box[0] - factor*(center - box[0]), box[1] + factor*(box[1] - center)

def coarse_abstract(f: AbstractModule, iter_coarseness, save_precision):
    var_to_idx = {'xnext': 0, 'ynext': 1, 'vxnext': 2, 'vynext': 3, 'thetanext': 4, 'omeganext': 5}
    default_boxes = {'x': (-.01, .01), 'y': (.49, .51), 'vx': (-.01,.01), 'vy': (-.01,.01), 'theta': (-.01,.01), 'omega': (-.01,.01)}
    iter = f.input_iter(precision=iter_coarseness)
    for iobox in iter:
        # Lift arguments to a full dimension with default_boxes
        for arg in default_boxes:
            if arg not in iobox:
                iobox[arg] = default_boxes[arg]

        # Bloat inputs for numerical precision, just in case.
        for k, v in iobox.items():
            if isinstance(v, tuple):
                iobox[k] = bloat(v)

        # Overapproximate
        s = lander_box_dynamics(steps=T, **iobox)
        out = {var: s[var_to_idx[var]] for var in f.outputs}  # Extract output dimension box
        out = {k, bloat(v, factor=-.002) for k, v in out.items()}
        iobox.update(out)
        iobox = {k: v for k, v in iobox.items() if k in f.vars}  # Filter unnecessary input/output slices. 

        f = f.io_refined(iobox, nbits=save_precision)
        f.check()

    print("Done coarse abs: ", f.outputs)

    return f

def abstract(subsys, samples=0, maxboxscale = 1.0):

    subsys['x'].mgr.configure(reordering=False)
    
    abs_starttime = time.time()
    for numapplied in range(samples):
        # Shrink window widths over time
        scale = maxboxscale / np.log10(numapplied+20)

        # Generate random input windows and points
        f_width = {'x':     scale*subsys['x']['x'].width(),
                   'y':     scale*subsys['y']['y'].width(),
                   'vx': scale*subsys['vx']['vx'].width(),
                   'vy': scale*subsys['vy']['vy'].width(),
                   'theta': scale*subsys['theta']['theta'].width(),
                   'omega': scale*subsys['omega']['omega'].width()}
        f_left = {k: subsys[k][k].lb + np.random.rand() * (subsys[k][k].width() - f_width[k]) for k in states}
        f_right = {k: f_width[k] + f_left[k] for k in states}
        iobox = {'a': numapplied % 4}
        iobox.update({k: (f_left[k], f_right[k]) for k in states})

        # Simulate and overapprox
        stateboxes = {k: v for k, v in iobox.items() if k is not 'a'}
        s = lander_box_dynamics(steps=T, a=iobox['a'], **stateboxes)
        
        out = {i: j for i, j in zip(nextstates, s)}
        out = {k, bloat(v, factor=-.002) for k, v in out.items()} ## Shrink output box ever so slightly.
        iobox.update(out)

        # plot_io_bounds(**stateboxes, steps=T, a=iobox['a'])
        # print(stateboxes)

        # Refine modules each with custom precision viewpoints
        # for s in ['vx', 'vy', 'theta', 'omega']:
        for s in states:
            filtered_io = {k: v for k, v in iobox.items() if k in subsys[s].vars}
            subsys[s] = subsys[s].io_refined(filtered_io, nbits=sysview[s])

        if numapplied % 2000 == 1999:
            print("Sample: {0}  Time: {1}, # Nodes: {2}  Scale: {3}".format(numapplied+1, time.time() - abs_starttime, len(subsys['x'].mgr), scale))

    print("Abs Time: ", time.time() - abs_starttime)

    return subsys

def synthesize_safe(f, safe, steps=None):

    cpre = DecompCPre(f, (('x', 'xnext'), ('y', 'ynext'), ('vx', 'vxnext'), ('vy', 'vynext'), ('theta', 'thetanext'), ('omega', 'omeganext')), ('a'))

    print("Solving Safety Game")
    game = SafetyGame(cpre, safe)
    game_starttime = time.time()
    inv, steps, controller = game.run(verbose=True, steps=steps)
    print("Solve time: ", time.time() - game_starttime)
    print("Solve steps: ", steps)
    print("Safe Size:", mgr.count(safe, statebits))
    print("Invariant Size:", mgr.count(inv, statebits))
    print("Safe Fraction: ", mgr.count(inv, statebits) / mgr.count(safe, statebits))

    assert inv == cpre.elimcontrol(controller.C)

    return controller, steps

def synthesize_reach(f, target):

    cpre = DecompCPre(f, (('x', 'xnext'), ('y', 'ynext'), ('vx', 'vxnext'), ('vy', 'vynext'), ('theta', 'thetanext'), ('omega', 'omeganext')), ('a'))

    print("Solving Reach Game")
    game = ReachGame(cpre, target)
    game_starttime = time.time()
    basin, steps, controller = game.run(verbose=True)
    print("Solve time: ", time.time() - game_starttime)
    print("Solve steps: ", steps)
    print("Trivial region: ", basin == target)
    print("Target Size:", mgr.count(target, statebits))
    print("Basin Size:", mgr.count(basin, statebits))
    print("Basin Growth: ", mgr.count(basin, statebits) / mgr.count(target, statebits))

    return controller, steps

def simulate(controller: MemorylessController, exclude=None):

    if not controller.isempty():

        from lunar_lander import LunarLander
        env = LunarLander()
        env.seed(1337)

        # Simulate and control
        import funcy as fn

        state = fn.first(controller.winning_states(exclude=exclude))
        state = {k: .5*(v[0] + v[1]) for k, v in state.items()}
        ordered_state = [state[v] for v in ['x', 'y', 'vx', 'vy', 'theta', 'omega']]
        env.reset(ordered_state)

        for step in range(20):
            u = fn.first(controller.allows(state))
            if u is None:
                print("No safe control inputs for state {}".format(state))
                env.close()
                f.children[0].mgr.configure(reordering=False)  # BUG: mgr.pick_iter toggles reordering to True
                return
            picked_u = {'a': u['a']}

            state.update(picked_u)
            print("Step: ", step, state)

            for i in range(T):
                s, _, _, _ = env.step(picked_u['a'])
                env.render()
                time.sleep(.03)
            
            state = {v: s[idx] for idx, v in enumerate(states)}

        env.close()
        return

def order_heuristic(mgr):
    
    def _name(i):
        return i.split('_')[0]

    def _idx(i):
        return int(i.split('_')[1])

    vars = list(mgr.vars)

    max_granularity = max([_idx(i) for i in vars])
    order_seed = []
    for i in range(max_granularity + 1):
        order_seed.extend([v for v in vars if _idx(v) == i])

    return {var: idx for idx, var in enumerate(order_seed)}

if __name__ is "__main__":
    
    # Run setup only on startup or if user sets cleanse option.
    init = False
    try:
        subsys
        mgr
        import sys
        if "-c" in sys.argv:
            mgr, subsys = setup(init=init)
            abs_iters = 0
    except NameError:
        mgr, subsys = setup(init=init)
        abs_iters = 0

    epoch = 0

    # mgr.configure(reordering=False)
    # mgr.reorder(order_heuristic(mgr))

    while(epoch < 6):
        print("Epoch:", epoch)
        print(mgr)
        N = 8000
        boxscale = .8*(epoch+1)**(-1/4)
        subsys = abstract(subsys, samples=N, maxboxscale=boxscale)
        abs_iters += N

        for state in states:
            print("System", state, "predicate size", len(subsys[state].pred))

        if epoch == 0:
            print("Nodes before reordering:", len(mgr))
            mgr.configure(reordering=False)
            mgr.reorder(order_heuristic(mgr))
            print("Nodes after reordering:", len(mgr))


        # # Solve reach operations
        # f = CompositeModule(tuple(subsys[i] for i in states))
        # target = f['x'].conc2pred(mgr, 'x', (0, .3), 5, innerapprox=True)
        # target &= f['y'].conc2pred(mgr, 'y', (.5, .8), 5, innerapprox=True)
        # target &= f['theta'].conc2pred(mgr, 'theta', (-.2, .2), 5, innerapprox=True)
        # target &= f['vy'].conc2pred(mgr, 'vy', (1,2.5), 5, innerapprox=True)
        # print("Target Size:", mgr.count(target, statebits))
        # controller, steps = synthesize_reach(f, target)
        # simulate(controller, exclude=target)
        # print("Mgr after simulate:", mgr.configure())

        # Solve safety operations
        # f = CompositeModule(tuple(subsys[i] for i in states))
        # safe = f['x'].conc2pred(mgr, 'x', (-.3, .3), 6, innerapprox=True)
        # safe &= f['y'].conc2pred(mgr, 'y', (.20, .80), 6, innerapprox=True)
        # safe &= f['theta'].conc2pred(mgr, 'theta', (-1, 1), 6, innerapprox=True)
        # print("Safe Size:", mgr.count(safe, statebits))
        # controller, steps = synthesize_safe(f, safe)
        # simulate(controller)

        epoch += 1

        print("\n\n")

    # Plot omega component
    # a_bdd = ~mgr.var('a_0') & mgr.var('a_1')
    # pred = mgr.exist(['a_0', 'a_1'], subsys['omega'].pred & a_bdd)
    # scatter2D(mgr, ('omega', subsys['omega']['omega']),
    #                ('omeganext', subsys['omega']['omeganext']),
    #                pred , 70)

    # Plot theta component
    # a_bdd = mgr.var('a_0') & ~mgr.var('a_1')
    # pred = mgr.exist(['a_0', 'a_1'], subsys['theta'].pred & a_bdd)
    # plot3D_QT(mgr, ('theta', subsys['theta']['theta']),
    #                ('omega', subsys['theta']['omega']),
    #                ('thetanext', subsys['theta']['thetanext']),
    #                pred , 70)

    # Plot vy component
    a_bdd = mgr.var('a_0') & ~mgr.var('a_1')
    omega_bdd = mgr.var('omega_0') & ~mgr.var('omega_1') & ~mgr.var('omega_2') & ~mgr.var('omega_3') & ~mgr.var('omega_4')# &  ~mgr.var('omega_5')
    elimvars = ['a_'+str(i) for i in [0,1]] + ['omega_'+str(i) for i in range(5)]
    pred = mgr.exist(elimvars, subsys['vy'].pred & a_bdd & omega_bdd)
    plot3D_QT(mgr, ('vy', subsys['vy']['vy']),
                   ('theta', subsys['vy']['theta']),
                   ('vynext', subsys['vy']['vynext']),
                   pred , 70)

    # Plot vx component wrt vx, theta
    a_bdd = mgr.var('a_0') & ~mgr.var('a_1')
    omega_bdd = mgr.var('omega_0') & ~mgr.var('omega_1') & ~mgr.var('omega_2') & ~mgr.var('omega_3') & ~mgr.var('omega_4')# &  ~mgr.var('omega_5')
    elimvars = ['a_'+str(i) for i in [0,1]] + ['omega_'+str(i) for i in range(5)]
    pred = mgr.exist(elimvars, subsys['vx'].pred & a_bdd & omega_bdd)
    plot3D_QT(mgr, ('vx', subsys['vx']['vx']),
                ('theta', subsys['vx']['theta']),
                ('vxnext', subsys['vx']['vxnext']),
                pred , 70)

    # Plot x component wrt x, vx. Ideally should see some dependence on vx.
    a_bdd = mgr.var('a_0') & ~mgr.var('a_1')
    # omega_bdd = mgr.var('omega_0') & ~mgr.var('omega_1') & ~mgr.var('omega_2') & ~mgr.var('omega_3') & ~mgr.var('omega_4')# &  ~mgr.var('omega_5')
    theta_bdd = mgr.var('theta_0') & ~mgr.var('theta_1') & ~mgr.var('theta_2') & ~mgr.var('theta_3') & ~mgr.var('theta_4')# &  ~mgr.var('omega_5')
    elimvars = (a_bdd & omega_bdd).support
    pred = mgr.exist(elimvars, subsys['x'].pred & a_bdd & theta_bdd)# & omega_bdd)
    plot3D_QT(mgr, ('x', subsys['x']['x']),
                ('vx', subsys['x']['vx']),
                ('xnext', subsys['x']['xnext']),
                pred , 70)
