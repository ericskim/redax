"""
Double integrator example where the abstraction is initialized with a coarse grid 
and further refined by randomly sampling of boxes. 
"""

from dd.cudd import BDD

import numpy as np 

from vpax.module import AbstractModule
from vpax.symbolicinterval import DynamicPartition
from vpax.synthesizer import ControlPre, SafetyGame
from vpax.visualizer import plot2D
from vpax.controlmodule import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ts = .2
k = .03
g = 9.8

mgr = BDD()

def dynamics(p,v,a):
    vsign = 1 if v > 0 else -1 
    return p + v*ts , v + a*ts - vsign*k*(v**2)*ts - g*ts

pspace = DynamicPartition(-10,10)
vspace = DynamicPartition(-16,16)
aspace = DynamicPartition(0,20)

# Monolithic module
system = AbstractModule(mgr, 
                        {'p': pspace,
                         'v': vspace,
                         'a': aspace},
                        {'pnext': pspace,
                         'vnext': vspace}
        )

        
bounds = {'p': [-10,10], 'v': [-16,16]}

# Visualization
# fig = plt.figure()
# ax = fig.gca(projection='3d') 
# voxelcolors = '#7A88CC' + '10' # color  + opacity 

# Declare grid precision 
precision = {'p': 6, 'v':6, 'a': 6, 'pnext': 6, 'vnext': 6}
bittotal = sum(precision.values()) 
outorder = {0: 'pnext', 1: 'vnext'}

def detect_violation(ll, ur):
    if ll[0] < bounds['p'][0] or ll[1] < bounds['v'][0]:
        return True
    if ur[0] > bounds['p'][1] or ur[1] > bounds['v'][1]:
        return True
    return False 

# Sample generator 
numapplied = 0
out_of_domain_violations = 0
possible_transitions = system.count_io(bittotal)
print("# I/O Transitions: ", possible_transitions)
for iobox in system.input_iter({'p': 3, 'v':4, 'a': 3}):

    f_left = {k: v[0] for k,v in iobox.items()}
    f_right = {k: v[1] for k,v in iobox.items()}

    # Generate output overapproximation 
    ll = dynamics(**f_left)
    ur = dynamics(**f_right)
    outbox = {outorder[i]: (ll[i], ur[i]) for i in range(2)}
    iobox.update(outbox)

    # Detect output domain violation 
    if detect_violation(ll,ur):
        out_of_domain_violations +=1
        continue
    
    # Apply 3d constraint 
    system.pred &= system.ioimplies2pred(iobox, precision = precision)

    # Apply 2d constraint to slices. Identical to parallel update.
    # system.pred &= system.ioimplies2pred({k:v for k,v in iobox.items() if k in {'p','v','pnext'}}, precision = precision)
    # system.pred &= system.ioimplies2pred({k:v for k,v in iobox.items() if k in {'v','a','vnext'}}, precision = precision)
    
    numapplied += 1

    # # Visualization 
    # p,v,a = np.indices(((2,2,2)))/1.0
    # p[0,:,:] += iobox['p'][0] 
    # p[1,:,:] *= iobox['p'][1]
    # v[:,0,:] += iobox['v'][0]
    # v[:,1,:] *= iobox['v'][1]
    # a[:,:,0] += iobox['a'][0]
    # a[:,:,1] *= iobox['a'][1]
    # ax.voxels(p,v,a, 
    #           np.array([[[True]]]), 
    #           facecolors =  np.array([[[voxelcolors]]])
    #           )

    if numapplied % 500 == 0:
        print("(samples, I/O % transitions) --- ({0}, {1})".format(numapplied, 100*system.count_io(bittotal)/possible_transitions))

print("# samples after exhaustive grid search: {0}".format(numapplied))

while(numapplied < 3577): 

    # Shrink window widths over time 
    width = 40 * 1/np.log10(2*numapplied+10)

    # Generate random input windows 
    f_width = {'p': np.random.rand()*width,
            'v':np.random.rand()*width,
            'a':np.random.rand()*width} 
    f_left = {'p': -10 +  np.random.rand() * (20 - f_width['p']), 
            'v': -16 + np.random.rand() * (32 - f_width['v']), 
            'a': 0 + np.random.rand() * (20 - f_width['a'])}
    f_right = {k: f_width[k] + f_left[k] for k in f_width}
    iobox = {k: (f_left[k], f_right[k]) for k in f_width}

    # Generate output overapproximation 
    ll = dynamics(**f_left)
    ur = dynamics(**f_right)
    outbox = {outorder[i]: (ll[i], ur[i]) for i in range(2)}
    iobox.update(outbox)

    # Detect output domain violation 
    if detect_violation(ll,ur):
        out_of_domain_violations +=1
        continue
    
    # Apply 3d constraint, even though the system has lower dimensions 
    system.pred &= system.ioimplies2pred(iobox, precision = precision)
    
    # Apply 2d constraint to slices. Identical to parallel update. 
    # system.pred &= system.ioimplies2pred({k:v for k,v in iobox.items() if k in {'p','v','pnext'}}, precision = precision)
    # system.pred &= system.ioimplies2pred({k:v for k,v in iobox.items() if k in {'v','a','vnext'}}, precision = precision)

    numapplied += 1

    if numapplied % 500 == 0:
        print("(samples, I/O % transitions) --- ({0}, {1})".format(numapplied, 100*system.count_io(bittotal)/possible_transitions))
    
# ax.set_xlim(-10,10)
# ax.set_ylim(-16,16)
# ax.set_zlim(0,20)
# plt.show() 
print("# I/O Transitions: ", system.count_io(bittotal))
print("# Out of Domain errors:", out_of_domain_violations) 

# Control system declaration 
csys = to_control_module(system, (('p', 'pnext'), ('v','vnext'))) 
cpre = ControlPre(csys)

# Declare safe set 
safe = pspace.box2pred(mgr, 'p', [-8,8], 6, innerapprox = True) 

inv = cpre(safe)
game = SafetyGame(cpre, safe)
inv, steps = game.step()

print("Safe Size:", system.mgr.count(safe, 12))
print("Invariant Size:", system.mgr.count( inv, 12))
print("Game Steps:", steps)

# plot2D(system.mgr, ('v', vspace), ('p', pspace), inv)
