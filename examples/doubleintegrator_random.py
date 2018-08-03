"""
Double integrator example where the abstraction is constructed via random sampling of boxes.
"""

from dd.cudd import BDD

import numpy as np 

from vpax.module import AbstractModule
from vpax.spaces import DynamicPartition
from vpax.synthesis import SafetyGame
from vpax.visualizer import plot2D, plot3D, plot3D_QT

from vpax.controlmodule import to_control_module

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ts = .2
k = .1
g = 9.8

mgr = BDD()

def dynamics(p,v,a):
    vsign = 1 if v > 0 else -1 
    return p + v*ts , v + a*ts - vsign*k*(v**2)*ts - g*ts

pspace = DynamicPartition(-10,10)
vspace = DynamicPartition(-16,16)
aspace = DynamicPartition(0,20) 

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

# fig = plt.figure()
# ax = fig.gca(projection='3d') 
# voxelcolors = '#7A88CC' + '10' # color  + opacity

# Declare grid precision 
precision = {'p': 6, 'v':6, 'a': 6, 'pnext': 6, 'vnext': 6}
bittotal = sum(precision.values())
outorder = {0: 'pnext', 1: 'vnext'}
possible_transitions = (pcomp | vcomp).count_io_space(bittotal)

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

    try: 
        # Apply 3d constraint
        system.apply_abstract_transitions(iobox, nbits =  precision)

        # Apply 2d constraint to slices. Identical to parallel update but cannot be decomposed later 
        system.apply_abstract_transitions({k:v for k,v in iobox.items() if k in {'p','v','pnext'}}, nbits =  precision)
        system.apply_abstract_transitions({k:v for k,v in iobox.items() if k in {'v','a','vnext'}}, nbits =  precision)
        
        # Apply constraint to parallel updates 
        pcomp.apply_abstract_transitions({k:v for k,v in iobox.items() if k in pcomp.vars}, nbits =  precision)
        vcomp.apply_abstract_transitions({k:v for k,v in iobox.items() if k in vcomp.vars}, nbits =  precision)

        
    except AssertionError:
        out_of_domain_violations +=1
        continue

    numapplied += 1

    # Visualization 
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

    if numapplied % 1000 == 0:
        system = pcomp | vcomp 
        print("# samples", numapplied, " --- # I/O transitions", system.count_io(bittotal))
        print("(samples, I/O % transitions) --- ({0}, {1})".format(numapplied, 100*system.count_io(bittotal)/possible_transitions))
        plot3D_QT(mgr, ('v', vspace), ('a', aspace), ('vnext', vspace), vcomp.pred, 128)


system = pcomp | vcomp

print("# I/O Transitions: ", system.count_io(bittotal))
print("# Out of Domain errors:", out_of_domain_violations) 

# Control system declaration 
csys = to_control_module(system, (('p', 'pnext'), ('v','vnext')))

# Declare safe set 
safe = pspace.conc2pred(mgr, 'p', [-8,8], 6, innerapprox = True) 

# Solve game and plot 2D invariant region 
game = SafetyGame(csys, safe)
inv, steps = game.step()
print("Safe Size:", system.mgr.count(safe, 12))
print("Invariant Size:", system.mgr.count( inv, 12))
plot2D(system.mgr, ('v', vspace), ('p', pspace), inv) 