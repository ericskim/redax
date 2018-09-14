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
from redax.visualizer import plot2D, plot3D, plot3D_QT

T = 8

# Abstract
precision = {'x': 6,     'y': 6,     'vx': 6,     'vy': 6,     'theta': 6,     'omega': 6,
                'xnext': 6, 'ynext': 6, 'vxnext': 6, 'vynext': 6, 'thetanext': 6, 'omeganext': 6}
statebits = sum([precision[s] for s in ['x', 'y', 'vx', 'vy', 'theta', 'omega']])

def setup(init=False):
    xspace = DynamicCover(-1, 1)
    yspace = DynamicCover(0, 1)
    vxspace = DynamicCover(-2, 2)
    vyspace = DynamicCover(-3, 3)
    thetaspace  = DynamicCover(-np.pi/4, np.pi/4, periodic=False)
    omegaspace = DynamicCover(-1.5, 1.5)
    actionspace = DiscreteSet(4)

    mgr = BDD()
    mgr.configure(reordering=False)  # Reordering causes too much time overhead for minimal gain.
    f_x = AbstractModule(mgr, {'x': xspace, 'vx': vxspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'xnext': xspace})
    f_y = AbstractModule(mgr, {'y': yspace, 'vy': vyspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'ynext': yspace})
    f_vx = AbstractModule(mgr, {'vx': vxspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'vxnext': vxspace})
    f_vy = AbstractModule(mgr, {'vy': vyspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'vynext': vyspace})
    f_theta = AbstractModule(mgr, {'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'thetanext': thetaspace})
    f_omega = AbstractModule(mgr, {'omega': omegaspace, 'a': actionspace}, {'omeganext': omegaspace})

    # Coarse, but exhaustive abstraction of submodules.
    if init:
        coarseness = {var: 4 for var in ['x', 'y', 'vx', 'vy', 'theta', 'omega']}
        initabs_start = time.time()
        f_x = initial_abstract(f_x, coarseness)
        mgr.reorder(order_heuristic(mgr))
        f_y = initial_abstract(f_y, coarseness)
        mgr.reorder(order_heuristic(mgr))
        f_vx = initial_abstract(f_vx, coarseness)
        mgr.reorder(order_heuristic(mgr))
        f_vy = initial_abstract(f_vy, coarseness)
        mgr.reorder(order_heuristic(mgr))
        f_theta = initial_abstract(f_theta, coarseness)
        mgr.reorder(order_heuristic(mgr))
        f_omega = initial_abstract(f_omega, coarseness)
        mgr.reorder(order_heuristic(mgr))
        print("Initial Abstraction Time (s): ", time.time() - initabs_start)

    f = CompositeModule((f_x, f_y, f_vx, f_vy, f_theta, f_omega))

    return mgr, f

def bloat(box):
    center = box[0] + .5 * (box[1]-box[0])
    return box[0] - .001*(center - box[0]), box[1] + .001*(box[1] - center)

def initial_abstract(f: AbstractModule, coarseness):
    var_to_idx = {'xnext': 0, 'ynext': 1, 'vxnext': 2, 'vynext': 3, 'thetanext': 4, 'omeganext': 5}
    default_boxes = {'x': (-.01, .01), 'y': (.49, .51), 'vx': (-.01,.01), 'vy': (-.01,.01), 'theta': (-.01,.01), 'omega': (-.01,.01)}
    iter = f.input_iter(precision=coarseness)
    for iobox in iter:
        # Lift arguments to a full dimension with default_boxes
        for arg in default_boxes:
            if arg not in iobox:
                iobox[arg] = default_boxes[arg]

        for k, v in iobox.items():
            if isinstance(v, tuple):
                iobox[k] = bloat(v)

        s = lander_box_dynamics(steps=T, **iobox)
        out = {var: s[var_to_idx[var]] for var in f.outputs}  # Extract output dimension box
        iobox.update(out)
        iobox = {k: v for k, v in iobox.items() if k in f.vars}  # Filter unnecessary input/output slices. 

        f = f.io_refined(iobox, nbits=precision)
        f.check()

    print("Done coarse abs: ", f.outputs)

    return f

def abstract(f, samples=0, maxboxscale = 1.0):

    f.children[0].mgr.configure(reordering=False)
    abs_starttime = time.time()
    for numapplied in range(samples):
        # Shrink window widths over time
        scale = maxboxscale / np.log10(numapplied+20)

        # Generate random input windows and points
        f_width = {'x':     scale*f['x'].width(),
                   'y':     scale*f['y'].width(),
                   'vx': .8*scale*f['vx'].width(),
                   'vy': .8*scale*f['vy'].width(),
                   'theta': .5*scale*f['theta'].width(),
                   'omega': .5*scale*f['omega'].width()}
        f_left = {'x':     f['x'].lb + np.random.rand() * (f['x'].width() - f_width['x']),  # (xspace.width() - f_width['x']),
                  'y':     f['y'].lb + np.random.rand() * (f['y'].width() - f_width['y']),  # (yspace.width() - f_width['y']),
                  'vx': f['vx'].lb + np.random.rand() * (f['vx'].width() - f_width['vx']),
                  'vy': f['vy'].lb + np.random.rand() * (f['vy'].width() - f_width['vy']),
                  'theta': f['theta'].lb + np.random.rand() * (f['theta'].width() - f_width['theta']),
                  'omega': f['omega'].lb + np.random.rand() * (f['omega'].width() - f_width['omega'])}
        f_right = {k: f_width[k] + f_left[k] for k in f_width}
        iobox = {'a': np.random.randint(0, 4)}
        iobox.update({k: (f_left[k], f_right[k]) for k in f_width})

        # Simulate and overapprox
        states = {k: v for k, v in iobox.items() if k is not 'a'}
        s = lander_box_dynamics(steps=T, a=iobox['a'], **states)
        out = {i: j for i, j in zip(f.outputs, s)}
        iobox.update(out)

        # plot_io_bounds(**states, steps=T, a=iobox['a'])
        # print(states)

        # Refine
        f = f.io_refined(iobox, nbits = precision)

        if numapplied % 400 == 399:
            print("Sample: {0}  Time: {1}, # Nodes: {2}  Scale: {3}".format(numapplied+1, time.time() - abs_starttime, len(f.children[0].mgr), scale))

    print("Abs Time: ", time.time() - abs_starttime)

    return f

def varorder(m: BDD):
    """
    BDD variable ordering in manager
    """
    from bidict import bidict

    lvls = bidict(m.var_levels).inv
    return [lvls[i] for i in range(len(lvls))]

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

    # FIXME: THIS FUNCTION SOMEHOW MUTATES THE MANAGER STATE TO ALLOW REORDERING.
    # pick_iter causes the reordering to flip back to true.

    if not controller.isempty():

        from lunar_lander import LunarLander
        env = LunarLander()
        env.seed(1337)

        # Simulate and control
        import funcy as fn

        state = fn.first(controller.winning_states(exclude=exclude))
        state = {k: .5*(v[0] + v[1]) for k, v in state.items()}
        state = {k: 0 for k in state}
        state['vy'] = 1.0
        state['y'] = .5
        ordered_state = [state[v] for v in ['x', 'y', 'vx', 'vy', 'theta', 'omega']]
        env.reset(ordered_state)

        for step in range(20):
            u = fn.first(controller.allows(state))
            if u is None:
                print("No safe control inputs for state {}".format(state))
                env.close()
                f.children[0].mgr.configure(reordering=False)  # BUG: mgr.pick_iter toggles reordering  to True
                return
            picked_u = {'a': u['a']}

            state.update(picked_u)
            print("Step: ", step, state)

            for i in range(T):
                s, _, _, _ = env.step(picked_u['a'])
                env.render()
                time.sleep(.05)
            
            state = {v: s[idx] for idx, v in enumerate(['x', 'y', 'vx', 'vy', 'theta', 'omega'])}

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
    init = True
    try:
        f
        mgr
        import sys
        if "-c" in sys.argv:
            mgr, f = setup(init=init)
            abs_iters = 0
    except NameError:
        mgr, f = setup(init=init)
        abs_iters = 0

    epoch = 0

    # mgr.configure(reordering=False)
    # mgr.reorder(order_heuristic(mgr))

    while(epoch < 5):
        print("Epoch:", epoch)
        print(mgr)
        N = 3000
        boxscale = (epoch+1)**(-1/4)
        f = abstract(f, samples=N, maxboxscale=boxscale)
        abs_iters += N

        if epoch == 0:
            print("Manager Nodes: ", len(mgr))
            mgr.configure(reordering=False)
            mgr.reorder(order_heuristic(mgr))

        print("Manager config:", mgr.configure())

        # # Solve reach operations
        # target = f['x'].conc2pred(mgr, 'x', (0, .6), 5, innerapprox=True)
        # target &= f['y'].conc2pred(mgr, 'y', (.5, .8), 5, innerapprox=True)
        # target &= f['theta'].conc2pred(mgr, 'theta', (-.2, .2), 5, innerapprox=True)
        # target &= f['vy'].conc2pred(mgr, 'vy', (1,2.5), 5, innerapprox=True)
        # controller, steps = synthesize_reach(f, target)
        # simulate(controller, exclude=target)
        # print("Mgr after simulate:", mgr.configure())

        # Solve safety operations
        safe = f['x'].conc2pred(mgr, 'x', (-.75, .75), 6, innerapprox=True)
        safe &= f['y'].conc2pred(mgr, 'y', (.25, .75), 6, innerapprox=True)
        safe &= f['theta'].conc2pred(mgr, 'theta', (-.5, .5), 6, innerapprox=True)
        controller, steps = synthesize_safe(f, safe, steps=2)
        # simulate(controller)

        # if not controller.isempty():
        #     break
        epoch += 1

        print("\n\n")

    # a_bdd = mgr.var('a_0') & mgr.var('a_1')
    # prednb = mgr.exist(['a_0', 'a_1'],(f.children[4].nonblock()) & a_bdd)
    # nbpred = mgr.exist(['a_0', 'a_1'],(f.children[4]._nb) & a_bdd)
    # assert prednb == nbpred
    # plot2D(mgr, ('theta', f['theta']) , ('omega', f['omega']), prednb)

    # # # Plot theta component
    a_bdd = mgr.var('a_0') & mgr.var('a_1')
    pred = mgr.exist(['a_0', 'a_1'], f.children[4].pred & a_bdd)
    plot3D_QT(mgr, ('theta', f['theta']),
                   ('omega', f['omega']),
                   ('thetanext', f['thetanext']),
                   pred , 70)
    mgr.configure(reordering=False)

    # # # Plot vx component 
    a_bdd = mgr.var('a_0') & ~mgr.var('a_1')
    omega_bdd = mgr.var('omega_0') & ~mgr.var('omega_1') & mgr.var('omega_2') & mgr.var('omega_3') & ~mgr.var('omega_4') &  ~mgr.var('omega_5')
    elimvars = ['a_'+str(i) for i in [0,1]] + ['omega_'+str(i) for i in range(6)]
    pred = mgr.exist(elimvars, f.children[3].pred & a_bdd & omega_bdd)
    plot3D_QT(mgr, ('vy', f['vy']),
                   ('theta', f['theta']),
                   ('vynext', f['vynext']),
                   pred , 70)
    mgr.configure(reordering=False)


    for i in range(6):
        print(len(f.children[i].pred))