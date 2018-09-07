"""
Double integrator example where the abstraction is constructed via random sampling of boxes.
"""

import time

init_time = time.time()

import numpy as np
try:
    from dd.cudd import BDD
except ImportError:
    from dd.autoref import BDD

import funcy as fn

from redax.controlmodule import to_control_module
from redax.module import AbstractModule, CompositeModule
from redax.spaces import DynamicCover
from redax.synthesis import SafetyGame, ControlPre, DecompCPre
from redax.visualizer import plot2D, plot3D, plot3D_QT


ts = .2
k = .1
g = 9.8

mgr = BDD()

mgr.configure(reordering=True)

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

# Composite system
composite = CompositeModule((pcomp, vcomp))

# Declare grid precision
p_precision = 7
v_precision = 7
precision = {'p': p_precision, 'v': v_precision, 'a': 7, 'pnext': p_precision, 'vnext': v_precision}
bittotal = sum(precision.values())
outorder = {0: 'pnext', 1: 'vnext'}
possible_transitions = (pcomp | vcomp).count_io_space(bittotal)

print("Setup time: ", time.time() - init_time)

# Sample generator
numapplied = 0
abs_starttime = time.time()
np.random.seed(1336)
while(numapplied < 1200):

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

    composite = composite.io_refined(iobox, nbits = precision)

    assert composite.children[0] == pcomp
    assert composite.children[1] == vcomp

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

print("Abstraction Time: ", time.time() - abs_starttime)

system = pcomp | vcomp


# Control system declaration
for nbits in [6]:

    cpre = ControlPre(system, (('p', 'pnext'), ('v', 'vnext')), ('a'))
    dcpre = DecompCPre(composite, (('p', 'pnext'), ('v', 'vnext')), ('a'))

    # Declare safe set
    safe = pspace.conc2pred(mgr, 'p', [-8,8], 6, innerapprox=True)

    # Solve game and plot 2D invariant region
    game = SafetyGame(cpre, safe)
    synth_starttime = time.time()
    inv, steps, controller = game.run()
    print("Solver Time: ", time.time() - synth_starttime)

    dgame = SafetyGame(dcpre, safe)
    dsynth_starttime = time.time()
    dinv, steps, controller = dgame.run()
    print("Dsolver: ", time.time() - dsynth_starttime)
    assert dinv == inv

    print("Solving Bits: ", nbits)
    print("Solver Steps: ", steps)
    print("Safe Size:", system.mgr.count(safe, p_precision + v_precision))
    print("Invariant Size:", system.mgr.count(inv,  p_precision + v_precision))
    # plot2D(system.mgr, ('v', vspace), ('p', pspace), inv)
# plot3D_QT(system.mgr, ('p', vspace), ('v', aspace), ('pnext', vspace), pcomp.pred, 128)
# plot3D_QT(system.mgr, ('v', vspace), ('a', aspace), ('vnext', vspace), vcomp.pred, 128)


sim_starttime = time.time()
"""Simulate"""
# state = {'p': -4, 'v': 2}
state = fn.first(controller.winning_states())
state = {k: .5*(v[0] + v[1]) for k, v in state.items()}
for step in range(10):
    u = fn.first(controller.allows(state))
    if u is None:
        break
    picked_u = {'a': u['a'][0]} # Pick lower bound of first allowed control voxel

    state.update(picked_u)
    print(step, state)
    nextstate = dynamics(**state)
    state = {'p': nextstate[0], 'v': nextstate[1]}
print("Simulation Time: ", time.time() - sim_starttime)