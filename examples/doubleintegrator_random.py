"""
Double integrator example where the abstraction is constructed via random sampling of boxes.
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



fig = plt.figure()
ax = fig.gca(projection='3d') 
voxelcolors = '#7A88CC' + '10' # color  + opacity 

# Declare grid precision 
precision = {'p': 6, 'v':6, 'a': 6, 'pnext': 6, 'vnext': 6}
bittotal = sum(precision.values()) 
outorder = {0: 'pnext', 1: 'vnext'}

# Sample generator 
numapplied = 0
out_of_domain_violations = 0
while(numapplied < 4000): 

    # Shrink window widths over time 
    width = 18 * 1/np.log10(2*numapplied+10)

    # Generate random input windows 
    f_width = {'p': np.random.rand()*.5*width,
            'v':np.random.rand()*width,
            'a':np.random.rand()*.5*width} 
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
    if ll[0] < bounds['p'][0] or ll[1] < bounds['v'][0]:
        out_of_domain_violations += 1
        continue
    if ur[0] > bounds['p'][1] or ur[1] > bounds['v'][1]:
        out_of_domain_violations += 1 
        continue
    
    # Apply 3d constraint 
    system.pred &= system.ioimplies2pred(iobox, precision = precision)

    # Apply 2d constraint to slices. Identical to parallel update. 
    # system.pred &= system.ioimplies2pred({k:v for k,v in iobox.items() if k in {'p','v','pnext'}}, precision = precision)
    # system.pred &= system.ioimplies2pred({k:v for k,v in iobox.items() if k in {'v','a','vnext'}}, precision = precision)
    
    # Apply constraint to parallel updates 
    # pcomp.pred &= pcomp.ioimplies2pred({k:v for k,v in iobox.items() if k in {'p','v','pnext'}}, precision = precision)
    # vcomp.pred &= vcomp.ioimplies2pred({k:v for k,v in iobox.items() if k in {'v','a','vnext'}}, precision = precision)
    
    numapplied += 1

    # Visualization 
    p,v,a = np.indices(((2,2,2)))/1.0
    p[0,:,:] += iobox['p'][0] 
    p[1,:,:] *= iobox['p'][1]
    v[:,0,:] += iobox['v'][0]
    v[:,1,:] *= iobox['v'][1]
    a[:,:,0] += iobox['a'][0]
    a[:,:,1] *= iobox['a'][1]
    ax.voxels(p,v,a, 
              np.array([[[True]]]), 
              facecolors =  np.array([[[voxelcolors]]])
              )
    
    if numapplied % 500 == 0:
        # system = pcomp | vcomp 
        print("# samples", numapplied, " --- # I/O transitions", system.count_io(bittotal))
        # assert (pcomp | vcomp) == system

# system = pcomp | vcomp

ax.set_xlim(-10,10)
ax.set_ylim(-16,16)
ax.set_zlim(0,20)
plt.show()
print("# I/O Transitions: ", system.count_io(bittotal))
print("# Out of Domain errors:", out_of_domain_violations) 

# Control system declaration 
csys = to_control_module(system, (('p', 'pnext'), ('v','vnext'))) 
cpre = ControlPre(csys)

# Declare safe set 
safe = pspace.box2pred(mgr, 'p', [-8,8], 6, innerapprox = True)

# inv = cpre(safe)
game = SafetyGame(cpre, safe)
inv, steps = game.step()

print("Safe Size:", system.mgr.count(safe, 12))
print("Invariant Size:", system.mgr.count( inv, 12))

plot2D(system.mgr, ('v', vspace), ('p', pspace), inv)