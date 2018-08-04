"""
Double integrator example where the abstraction is initialized with a coarse grid 
and further refined by randomly sampling of boxes. 
"""

from dd.cudd import BDD

import numpy as np 

from vpax.module import AbstractModule
from vpax.spaces import DynamicPartition
from vpax.synthesis import SafetyGame
from vpax.visualizer import plot2D, plot3D
from vpax.controlmodule import to_control_module

import matplotlib.pyplot as plt

ts = .2
k = .1
g = 9.8

mgr = BDD()

def dynamics(p,v,a):
    vsign = 1 if v > 0 else -1 
    return p + v * ts , v + a * ts - vsign*k*(v**2)*ts - g*ts

pspace = DynamicPartition(-10, 10)
vspace = DynamicPartition(-16, 16)
aspace = DynamicPartition(0, 20)

# Monolithic module
system = AbstractModule(mgr, 
                        {'p': pspace,
                         'v': vspace,
                         'a': aspace},
                        {'pnext': pspace,
                         'vnext': vspace}
        )

bounds = {'p': [-10, 10], 'v': [-16, 16]}

# Declare grid precision 
precision = {'p': 6, 'v': 6, 'a': 6, 'pnext': 6, 'vnext': 6}
bittotal = sum(precision.values()) 
outorder = {0: 'pnext', 1: 'vnext'}

# Sample generator 
numapplied = 0
out_of_domain_violations = 0
possible_transitions = system.count_io_space(bittotal)
print("# I/O Transitions: ", possible_transitions)
for iobox in system.input_iter({'p': 3, 'v': 4, 'a': 3}):

    f_left = {k: v[0] for k,v in iobox.items()}
    f_right = {k: v[1] for k,v in iobox.items()}

    # Generate output overapproximation 
    ll = dynamics(**f_left)
    ur = dynamics(**f_right)
    outbox = {outorder[i]: (ll[i], ur[i]) for i in range(2)}
    iobox.update(outbox)
    
    # Apply 3d constraint
    try:
        system.apply_abstract_transitions(iobox, nbits = precision)
    except AssertionError:
        out_of_domain_violations += 1
        continue

    # Apply 2d constraint to slices. Identical to parallel update.
    system.apply_abstract_transitions({k: v for k, v in iobox.items() if k in {'p', 'v', 'pnext'}}, nbits = precision)
    system.apply_abstract_transitions({k: v for k, v in iobox.items() if k in {'v', 'a', 'vnext'}}, nbits = precision)


    numapplied += 1

    if numapplied % 500 == 0:
        print("(samples, I/O % transitions) --- ({0}, {1})".format(numapplied, 100*system.count_io(bittotal)/possible_transitions))

print("# samples after exhaustive grid search: {0}".format(numapplied))

while(numapplied < 4000): 

    # Shrink window widths over time 
    width = 40 * 1/np.log10(2*numapplied+10)

    # Generate random input windows 
    f_width = {'p': np.random.rand()*width,
               'v': np.random.rand()*width,
               'a': np.random.rand()*width}
    f_left = {'p': -10 + np.random.rand() * (20 - f_width['p']),
              'v': -16 + np.random.rand() * (32 - f_width['v']),
              'a': 0 + np.random.rand() * (20 - f_width['a'])}
    f_right = {k: f_width[k] + f_left[k] for k in f_width}
    iobox = {k: (f_left[k], f_right[k]) for k in f_width}

    # Generate output overapproximation 
    ll = dynamics(**f_left)
    ur = dynamics(**f_right)
    outbox = {outorder[i]: (ll[i], ur[i]) for i in range(2)}
    iobox.update(outbox)
    
    try: 
        # Apply 3d constraint, even though the system has lower dimensions
        # system.apply_abstract_transitions(iobox, nbits = precision)

        # Apply 2d constraint to slices. Identical to parallel update.
        system.apply_abstract_transitions({k:v for k,v in iobox.items() if k in {'p', 'v', 'pnext'}}, nbits = precision)
        system.apply_abstract_transitions({k:v for k,v in iobox.items() if k in {'v', 'a', 'vnext'}}, nbits = precision)

    except AssertionError:
        out_of_domain_violations +=1
        continue

    numapplied += 1

    if numapplied % 500 == 0:
        iotrans = system.count_io(bittotal)
        print("(samples, I/O % transitions,bddsize) --- ({0}, {1:.3}, {2})".format(numapplied, 
                                                            100*iotrans/possible_transitions,
                                                            len(system.pred)))
    
print("# I/O Transitions: ", system.count_io(bittotal))
print("# Out of Domain errors:", out_of_domain_violations) 

# Control system declaration 
csys = to_control_module(system, (('p', 'pnext'), ('v','vnext')))

# Declare safe set 
safe = pspace.conc2pred(mgr, 'p', [-8, 8], 6, innerapprox = True)

game = SafetyGame(csys, safe)
inv, steps, controller = game.step()

print("Safe Size:", system.mgr.count(safe, 12))
print("Invariant Size:", system.mgr.count(inv, 12))
print("Game Steps:", steps)

plot2D(system.mgr, ('v', vspace), ('p', pspace), inv)
# plot3D(system.mgr, ('v', vspace), ('p', pspace), ('a', aspace), inv)
