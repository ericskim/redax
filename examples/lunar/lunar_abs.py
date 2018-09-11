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

T = 20

# Abstract
precision = {'x': 6,     'y': 6,     'vx': 6,     'vy': 6,     'theta': 6,     'omega': 6,
                'xnext': 6, 'ynext': 6, 'vxnext': 6, 'vynext': 6, 'thetanext': 6, 'omeganext': 6}
statebits = sum([precision[s] for s in ['x', 'y', 'vx', 'vy', 'theta', 'omega']])

def setup():
    xspace = DynamicCover(-1, 1)
    yspace = DynamicCover(0, 1)
    vxspace = DynamicCover(-2.5, 2.5)
    vyspace = DynamicCover(-5, 3)
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
    f_x = initial_abstract(f_x)
    f_y = initial_abstract(f_y)
    f_vx = initial_abstract(f_vx)
    f_vy = initial_abstract(f_vy)
    f_theta = initial_abstract(f_theta)
    f_omega = initial_abstract(f_omega)

    f = CompositeModule((f_x, f_y, f_vx, f_vy, f_theta, f_omega))

    return mgr, f

def initial_abstract(f: AbstractModule):
    coarseness = {'x': 4,     'y': 4,     'vx': 4,     'vy': 4,     'theta': 4,     'omega': 3}
    var_to_idx = {'xnext': 0, 'ynext': 1, 'vxnext': 2, 'vynext': 3, 'thetanext': 4, 'omeganext': 5}
    default_boxes = {'x': (-.01, .01), 'y': (.49, .51), 'vx': (-.01,.01), 'vy': (-.01,.01), 'theta': (-.01,.01), 'omega': (-.01,.01)}
    iter = f.input_iter(precision=coarseness)

    for iobox in iter:
        # Lift arguments to a full dimension with default_boxes
        for arg in default_boxes:
            if arg not in iobox:
                iobox[arg] = default_boxes[arg]

        s = lander_box_dynamics(steps=T, **iobox)
        out = {var: s[var_to_idx[var]] for var in f.outputs}  # Extract output dimension box
        iobox.update(out)
        iobox = {k: v for k, v in iobox.items() if k in f.vars}  # Filter unnecessary input/output slices. 

        f = f.io_refined(iobox, nbits=precision)
        f.check()

    print("Done coarse abs: ", f.outputs)

    return f

def abstract(f, samples=100):

    abs_starttime = time.time()
    for numapplied in range(samples):
        # Shrink window widths over time
        scale = .75 / np.log10(numapplied+20)

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

        # Refine
        f = f.io_refined(iobox, nbits = precision)

        if numapplied % 2000 == 1999:
            print("Sample: {0}  Time: {1}  Scale: {2}".format(numapplied+1, time.time() - abs_starttime, scale))

    print("Abs Time: ", time.time() - abs_starttime)

    f.check()
    return f

def synthesize_safe(f, safe):

    cpre = DecompCPre(f, (('x', 'xnext'), ('y', 'ynext'), ('vx', 'vxnext'), ('vy', 'vynext'), ('theta', 'thetanext'), ('omega', 'omeganext')), ('a'))

    # Solve game and plot 2D invariant region
    print("Solving Safety Game")
    game = SafetyGame(cpre, safe)
    game_starttime = time.time()
    inv, steps, controller = game.run(verbose=True)
    print("Solve time: ", time.time() - game_starttime)
    print("Solve steps: ", steps)
    print("Safe Size:", mgr.count(safe, statebits))
    print("Invariant Size:", mgr.count(inv, statebits))
    print("Safe Fraction: ", mgr.count(inv, statebits) / mgr.count(safe, statebits))

    assert inv == cpre.elimcontrol(controller.C)

    return controller, steps

def synthesize_reach(f, target):

    cpre = DecompCPre(f, (('x', 'xnext'), ('y', 'ynext'), ('vx', 'vxnext'), ('vy', 'vynext'), ('theta', 'thetanext'), ('omega', 'omeganext')), ('a'))

    # Solve game and plot 2D invariant region
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

def simulate(controller: MemorylessController, exclude):

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
        try: 
            for step in range(10):
                u = fn.first(controller.allows(state))
                if u is None:
                    raise RuntimeError("No safe control inputs for state {}".format(state))
                picked_u = {'a': u['a']}

                state.update(picked_u)
                print("Step: ", step, state)

                for i in range(T):
                    s, _, _, _ = env.step(picked_u['a'])
                    env.render()
                    time.sleep(.1)
                
                state = {v: s[idx] for idx, v in enumerate(['x', 'y', 'vx', 'vy', 'theta', 'omega'])}
        except:
            env.close()
            raise

        env.close()

def order_heuristic_seed(mgr):
    
    def _name(i):
        return i.split('_')[0]

    def _idx(i):
        return int(i.split('_')[1])

    vars = mgr.vars

    max_granularity = max([_idx(i) for i in vars])
    order_seed = []
    for i in range(max_granularity + 1):
        order_seed.extend([v for v in vars if _idx(v) == i])

    return {var: idx for idx, var in enumerate(order_seed)}

if __name__ is "__main__":
    
    try:
        f
        mgr
        import sys
        if "-c" in sys.argv:
            mgr, f = setup()
            abs_iters = 0
    except NameError:
        mgr, f = setup()
        abs_iters = 0

    epoch = 0

    while(epoch < 2):
        print("Epoch:", epoch)
        N = 20000
        f = abstract(f, samples=N)
        abs_iters += N

        if epoch == 0:
            print(len(mgr))
            mgr.configure(reordering=False)
            mgr.reorder(order_heuristic_seed(mgr))
            print(len(mgr))


        # # Solve safety operations
        # target = f['x'].conc2pred(mgr, 'x', (-.3, .3), 5, innerapprox=True)
        # target &= f['y'].conc2pred(mgr, 'y', (.3, .7), 5, innerapprox=True)
        # controller, steps = synthesize_reach(f)

        # # Solve safety operations
        # safe = f['x'].conc2pred(mgr, 'x', (-.4, .4), 5, innerapprox=True)
        # safe &= f['y'].conc2pred(mgr, 'y', (.3, .7), 5, innerapprox=True)
        # controller, steps = synthesize_safe(f)

        # if not controller.isempty():
        #     break
        epoch += 1

        print("\n\n")

    # target = f['x'].conc2pred(mgr, 'x', (-.3, .3), 5, innerapprox=True)
    # target &= f['y'].conc2pred(mgr, 'y', (.7, .9), 5, innerapprox=True)
    # simulate(controller, exclude=target)

    # a_bdd = mgr.var('a_0') & mgr.var('a_1')
    # prednb = mgr.exist(['a_0', 'a_1'],(f.children[4].nonblock()) & a_bdd)
    # nbpred = mgr.exist(['a_0', 'a_1'],(f.children[4]._nb) & a_bdd)
    # plot2D(mgr, ('theta', f['theta']) , ('omega', f['omega']), prednb)
    # plot2D(mgr, ('theta', f['theta']) , ('omega', f['omega']), nbpred)

    # # # Plot theta component
    # pred = mgr.exist(['a_0', 'a_1'], f.children[4].pred & a_bdd)
    # plot3D_QT(mgr, ('theta', f['theta']),
    #                ('omega', f['omega']),
    #                ('thetanext', f['thetanext']), 
    #                pred , 70)

    # Print out how many elements in a submodule are nonblocking
    # for i in range(4):
    #     print(mgr.count(f.children[i]._nb, 23))

