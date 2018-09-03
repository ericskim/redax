import time

import numpy as np
try:
    from dd.cudd import BDD
except ImportError:
    from dd.autoref import BDD

import gym
from lunar_lander import LunarLander

from redax.module import AbstractModule, CompositeModule
from redax.spaces import DynamicCover, EmbeddedGrid, FixedCover, DiscreteSet
from redax.synthesis import SafetyGame, ControlPre, DecompCPre
from redax.visualizer import plot3D, plot3D_QT

# x = [-1,1]
# y = [0,1]
# angle = [-pi, pi]
# Not sure what reasonable velocities
# Linear velocity is around [-10,10] for each positional dimension
# Angular velocity is also roughly [-10, 10] but could easily go higher 
xspace = DynamicCover(-1, 1)
yspace = DynamicCover(0, 1)
vxspace = DynamicCover(-1, 1)
vyspace = DynamicCover(-1, 1)
thetaspace  = DynamicCover(-np.pi, np.pi, periodic=True)
omegaspace = DynamicCover(-1, 1)
actionspace = DiscreteSet(4)

mgr = BDD()
# f_x = AbstractModel(mgr, {x, xvel, theta, omega}, {xnext}) <--- ideal signature with named spaces or "wires"
f_x = AbstractModule(mgr, {'x': xspace, 'xvel': vxspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'xnext': xspace})
f_y = AbstractModule(mgr, {'y': yspace, 'yvel': vyspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'ynext': yspace})
f_vx = AbstractModule(mgr, {'xvel': vxspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'xvelnext': vxspace})
f_vy = AbstractModule(mgr, {'yvel': vyspace, 'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'yvelnext': vyspace})
f_theta = AbstractModule(mgr, {'theta': thetaspace, 'omega': omegaspace, 'a': actionspace}, {'thetanext': thetaspace})
f_omega = AbstractModule(mgr, {'omega': omegaspace, 'a': actionspace}, {'omeganext': omegaspace})

# f = f_x | f_y | f_vx | f_vy | f_theta | f_omega
f = CompositeModule((f_x, f_y, f_vx, f_vy, f_theta, f_omega))

# Edit lunar lander to take reset states, get rid of rewards.
# simulate a bunch of different boxes
# Find out the reachable set bound by sampling along corners of 6D set and multiplying by a safety factor
# Ten time steps?
from lunar_lander import LunarLander
env = LunarLander()
env.seed(1337)

precision = {'x': 6, 'y': 6, 'xvel': 5, 'yvel': 5, 'theta': 5, 'omega': 5, 
             'xnext': 6, 'ynext': 6, 'xvelnext': 5, 'yvelnext': 5, 'thetanext': 5, 'omeganext': 5}
             
# Abstract
abs_starttime = time.time()
for numapplied in range(50000):
    # Shrink window widths over time
    scale = 1/np.log10(3.0*numapplied+10)

    # Generate random input windows and points
    f_width = {'x':     np.random.rand()*scale*.8,#xspace.width(),
               'y':     np.random.rand()*scale*.4,#yspace.width(),
               'xvel': np.random.rand()*scale*vxspace.width(),
               'yvel': np.random.rand()*scale*vyspace.width(),
               'theta': .2*thetaspace.width(),
               'omega': np.random.rand()*scale*omegaspace.width()}
    f_left = {'x':     -.4 + np.random.rand() * (.4 - f_width['x']),# (xspace.width() - f_width['x']),
              'y':     .3 + np.random.rand() * (.7 - f_width['y']), # (yspace.width() - f_width['y']),
              'xvel': vxspace.lb + np.random.rand() * (vxspace.width() - f_width['xvel']),
              'yvel': vyspace.lb + np.random.rand() * (vyspace.width() - f_width['yvel']),
              'theta': thetaspace.lb + np.random.rand() * (thetaspace.width()),
              'omega': omegaspace.lb + np.random.rand() * (omegaspace.width() - f_width['omega'])}
    f_right = {k: f_width[k] + f_left[k] for k in f_width}
    iobox = {'a':     np.random.randint(0, 4)}
    iobox.update({k: (f_left[k], f_right[k]) for k in f_width})

    # Simulate and overapprox
    states = {k: v for k, v in iobox.items() if k is not 'a'}
    s = env.reset(tuple(.5 * (l + r) for l, r in states.values()))
    for _ in range(30):
        s, r, done, info = env.step(iobox['a'])
    
    # time.sleep(1)

    out = {'xnext': s[0], 'ynext': s[1], 'xvelnext': s[2], 'yvelnext': s[3], 'thetanext': s[4], 'omeganext': s[5]}
    out = {o[0]: (o[1] - .6*w, o[1] + .6*w) for w, o in zip(f_width.values(), out.items())}
    iobox.update(out)

    # Refine
    f = f.io_refined(iobox, nbits = precision)

    if numapplied % 1000 == 999:
        print("Sample: {0}  Time: {1}".format(numapplied+1, time.time() - abs_starttime))

print("Abs Time: ", time.time() - abs_starttime)


# Solve safety operations
safe = xspace.conc2pred(mgr, 'x', (-.4, .4), 6, innerapprox=True)
safe &= yspace.conc2pred(mgr, 'y', (.3, .7), 6, innerapprox=True)

cpre = DecompCPre(f, (('x', 'xnext'), ('y', 'ynext'), ('xvel', 'xvelnext'), ('yvel', 'yvelnext'), ('theta', 'thetanext'), ('omega', 'omeganext')), ('a'))

# Solve game and plot 2D invariant region
print("Solving Game")
game = SafetyGame(cpre, safe)
game_starttime = time.time()
inv, steps, controller = game.run(verbose=True)
print("Solve time: ", time.time() - game_starttime)
print("Solve steps: ", steps)
print("Trivial region: ", inv == mgr.false)
print("Safe Size:", mgr.count(safe, 6*6))
print("Invariant Size:", mgr.count(inv, 6*6))