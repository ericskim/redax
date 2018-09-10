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
from redax.synthesis import SafetyGame, ControlPre, DecompCPre
from redax.visualizer import plot2D, plot3D, plot3D_QT

T = 15

def setup():
    xspace = DynamicCover(-1, 1)
    yspace = DynamicCover(0, 1)
    vxspace = DynamicCover(-2.5, 2.5)
    vyspace = DynamicCover(-5, 3)
    thetaspace  = DynamicCover(-np.pi/4, np.pi/4, periodic=False)
    # thetaspace  = DynamicCover(-np.pi, np.pi, periodic=True)
    omegaspace = DynamicCover(-1.5, 1.5)
    actionspace = DiscreteSet(4)

    mgr = BDD()
    mgr.configure(reordering=True)
    f_x = AbstractModule(mgr, {'x': xspace, 'vx': vxspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'xnext': xspace})
    f_y = AbstractModule(mgr, {'y': yspace, 'vy': vyspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'ynext': yspace})
    f_vx = AbstractModule(mgr, {'vx': vxspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'vxnext': vxspace})
    f_vy = AbstractModule(mgr, {'vy': vyspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'vynext': vyspace})
    f_theta = AbstractModule(mgr, {'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'thetanext': thetaspace})
    f_omega = AbstractModule(mgr, {'omega': omegaspace, 'a': actionspace}, {'omeganext': omegaspace})

    f = CompositeModule((f_x, f_y, f_vx, f_vy, f_theta, f_omega))

    return mgr, f


def abstract(f, samples=100):

    # Abstract
    precision = {'x': 5,     'y': 5,     'vx': 6,     'vy': 6,     'theta': 5,     'omega': 5,
                 'xnext': 5, 'ynext': 5, 'vxnext': 6, 'vynext': 6, 'thetanext': 5, 'omeganext': 5}

    abs_starttime = time.time()
    for numapplied in range(samples):
        # Shrink window widths over time
        scale = 1/np.log10(3.0*numapplied+20)

        # Generate random input windows and points
        f_width = {'x':     .3*.8,#xspace.width(),
                   'y':     .3*.6,#yspace.width(),
                   'vx': .2**f['vx'].width(),
                   'vy': .2*f['vy'].width(),
                   'theta': .1*f['theta'].width(),
                   'omega': .1*f['omega'].width()}
        f_left = {'x':     -.4 + np.random.rand() * (.4 - f_width['x']),  # (xspace.width() - f_width['x']),
                  'y':     .2 + np.random.rand() * (.8 - f_width['y']),  # (yspace.width() - f_width['y']),
                  'vx': f['vx'].lb + np.random.rand() * (f['vx'].width() - f_width['vx']),
                  'vy': f['vx'].lb + np.random.rand() * (f['vx'].width() - f_width['vy']),
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

        if numapplied % 1000 == 999:
            print("Sample: {0}  Time: {1}".format(numapplied+1, time.time() - abs_starttime))

    print("Abs Time: ", time.time() - abs_starttime)

    # f.check()
    return f

def synthesizesafe(f):

    # Solve safety operations
    safe = f['x'].conc2pred(mgr, 'x', (-.4, .4), 6, innerapprox=True)
    safe &= f['y'].conc2pred(mgr, 'y', (.2, .8), 6, innerapprox=True)

    cpre = DecompCPre(f, (('x', 'xnext'), ('y', 'ynext'), ('vx', 'vxnext'), ('vy', 'vynext'), ('theta', 'thetanext'), ('omega', 'omeganext')), ('a'))
    # fmono = f.children[0] | f.children[1] | f.children[2] | f.children[3] | f.children[4] | f.children[5]
    # cpre = ControlPre(fmono, (('x', 'xnext'), ('y', 'ynext'), ('vx', 'vxnext'), ('vy', 'vynext'), ('theta', 'thetanext'), ('omega', 'omeganext')), ('a'))

    # Solve game and plot 2D invariant region
    print("Solving Game")
    game = SafetyGame(cpre, safe)
    game_starttime = time.time()
    inv, steps, controller = game.run(verbose=True)
    print("Solve time: ", time.time() - game_starttime)
    print("Solve steps: ", steps)
    print("Trivial region: ", inv == mgr.false)
    print("Safe Size:", mgr.count(safe, 32))
    print("Invariant Size:", mgr.count(inv, 32))
    print("Safe Fraction: ", mgr.count(inv, 32) / mgr.count(safe, 32))

    assert inv == cpre.elimcontrol(controller.C)

    return controller

def simulate(controller):

    if not controller.isempty():

        from lunar_lander import LunarLander
        env = LunarLander()
        env.seed(1337)

        # Simulate and control
        import funcy as fn
        state = fn.first(controller.winning_states())
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
                print(step, state)

                for i in range(T):
                    s, _, _, _ = env.step(picked_u['a'])
                    env.render()
                    time.sleep(.1)
                
                state = {v: s[idx] for idx, v in enumerate(['x', 'y', 'vx', 'vy', 'theta', 'omega'])}
        except:
            env.close()
            raise

        env.close()



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

    while(epoch < 3):
        print("Epoch:", epoch)
        N = 5000
        f = abstract(f, samples=N)
        abs_iters += N

        # f.check()
        controller = synthesizesafe(f)

        if not controller.isempty():
            break
        epoch += 1

    # simulate(controller)

    a_bdd = mgr.var('a_0') & ~mgr.var('a_1')
    prednb = mgr.exist(['a_0', 'a_1'],(f.children[4].nonblock()) & a_bdd)
    nbpred = mgr.exist(['a_0', 'a_1'],(f.children[4]._nb) & a_bdd)
    # plot2D(mgr, ('theta', f['theta']) , ('omega', f['omega']), prednb)
    # plot2D(mgr, ('theta', f['theta']) , ('omega', f['omega']), nbpred)

    pred = mgr.exist(['a_0', 'a_1'], f.children[4].pred & a_bdd)
    plot3D_QT(mgr, ('theta', f['theta']) , 
                   ('omega', f['omega']),
                   ('thetanext', f['thetanext']), 
                   pred , 70)


