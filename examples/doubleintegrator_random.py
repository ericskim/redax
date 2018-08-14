"""
Double integrator example where the abstraction is constructed via random sampling of boxes.
"""

import time

import numpy as np
try:
    from dd.cudd import BDD
except ImportError:
    from dd.autoref import BDD

from sydra.controlmodule import to_control_module
from sydra.module import AbstractModule
from sydra.spaces import DynamicCover
from sydra.synthesis import SafetyGame
from sydra.visualizer import plot2D, plot3D, plot3D_QT

ts = .2
k = .1
g = 9.8

mgr = BDD()


def dynamics(p, v, a):
    vsign = 1 if v > 0 else -1
    return p + v * ts, v + a*ts - vsign*k*(v**2)*ts - g*ts

pspace = DynamicCover(-10, 10)
vspace = DynamicCover(-16, 16)
aspace = DynamicCover(0, 20)

# Smaller component modules
pcomp = AbstractModule(mgr,
                       {'p': pspace,
                        'v': vspace},
                       {'pnext': pspace}
        )
vcomp = AbstractModule(mgr,
                       {'v': vspace,
                        'a': aspace},
                       {'vnext': vspace}
        )

bounds = {'p': [-10,10], 'v': [-16,16]}

# Monolithic system
system = pcomp | vcomp

# Declare grid precision
precision = {'p': 6, 'v': 6, 'a': 6, 'pnext': 6, 'vnext': 6}
bittotal = sum(precision.values())
outorder = {0: 'pnext', 1: 'vnext'}
possible_transitions = (pcomp | vcomp).count_io_space(bittotal)

# Sample generator
numapplied = 0
abs_starttime = time.time()
while(numapplied < 2000):

    # Shrink window widths over time
    width = 18 * 1/np.log10(2*numapplied+10)

    # Generate random input windows
    f_width = {'p': np.random.rand()*.5*width,
               'v': np.random.rand()*width,
               'a': np.random.rand()*.5*width}
    f_left  = {'p': -10 +  np.random.rand() * (20 - f_width['p']),
               'v': -16 + np.random.rand() * (32 - f_width['v']),
               'a': 0 + np.random.rand() * (20 - f_width['a'])}
    f_right = {k: f_width[k] + f_left[k] for k in f_width}
    iobox   = {k: (f_left[k], f_right[k]) for k in f_width}

    # Generate output overapproximation
    ll = dynamics(**f_left)
    ur = dynamics(**f_right)
    outbox = {outorder[i]: (ll[i], ur[i]) for i in range(2)}
    iobox.update(outbox)

    # Apply constraint to parallel updates
    pcomp = pcomp.io_refined({k: v for k, v in iobox.items() if k in pcomp.vars}, nbits=precision)
    vcomp = vcomp.io_refined({k: v for k, v in iobox.items() if k in vcomp.vars}, nbits=precision)

    numapplied += 1

    if numapplied % 1000 == 0:
        system = pcomp | vcomp

        iotrans = system.count_io(bittotal)
        print("(samples, I/O % trans., bddsize, time(s))"
              " --- ({0}, {1:.3}, {2}, {3})".format(numapplied, 
                                                   100*iotrans/possible_transitions,
                                                   len(system.pred),
                                                   time.time() - abs_starttime)
             )
        name = "{0} samples".format(numapplied)
        # plot3D(system.mgr, ('v', vspace), ('a', aspace), ('vnext', vspace), vcomp.pred, 
        #         opacity=100, view=(25,-100), title=name, fname=name)


system = pcomp | vcomp

# Control system declaration
csys = to_control_module(system, (('p', 'pnext'), ('v', 'vnext')))

# Declare safe set
safe = pspace.conc2pred(mgr, 'p', [-8,8], 6, innerapprox=True) 

# Solve game and plot 2D invariant region
game = SafetyGame(csys, safe)
synth_starttime = time.time()
inv, steps, controller = game.step()
print("Solver Time: ", time.time() - synth_starttime)
print("Safe Size:", system.mgr.count(safe, 12))
print("Invariant Size:", system.mgr.count(inv, 12))
# plot2D(system.mgr, ('v', vspace), ('p', pspace), inv)

# plot3D_QT(system.mgr, ('p', vspace), ('v', aspace), ('pnext', vspace), pcomp.pred, 128)
# plot3D_QT(system.mgr, ('v', vspace), ('a', aspace), ('vnext', vspace), vcomp.pred, 128)

"""Simulate"""
# state = {'p': -4, 'v': 2}
# for step in range(10):
#     u = [i for i in controller.allows(state)]  # Pick first allowed control
#     if len(u) == 0:
#         break
#     u = {'a': u[0]['a'][0]}
#     state.update(u)
#     print(step, state)
#     nextstate = dynamics(**state)
#     state = {'p': nextstate[0], 'v': nextstate[1]}
