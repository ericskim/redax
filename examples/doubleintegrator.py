from dd.cudd import BDD

import numpy as np 

from vpax.module import input, output

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ts = .2
k = .03
g = 9.8

mgr = BDD()

@input(mgr, 'p', (-10,10), precision = 6)
@input(mgr, 'a', (-20,20), precision = 6)
@input(mgr, 'v', (-20,20), precision = 6)
@output(mgr, (0, 'p_next'), (-10,10), precision = 6)
@output(mgr, (1, 'v_next'), (-20,20), precision = 6)
def system(p, v, a) -> (float, float):
    vsign = 1 if v > 0 else -1 
    return p + v*ts, v + a*ts - vsign*k*(v**2)*ts - g*ts

"""
inputs = {'p': DynamicInterval(-10,10),
          'v': DynamicInterval(-20,20),
          'a': DynamicInterval(-20,20)}
outputs = {'pnext': DynamicInterval(-10,10),
           'vnext': DynamicInterval(-20,20)}

system = AbstractModule(mgr, inputs, outputs)
grid_iterator = system.inputiterator(precision = {'p': 2, 'v': 3, 'a': 3})
for i in grid_iterator: 
    out_OA = OA(i) 
    system.apply_io_constraint(i,out_OA, precision)

# Hides an output 
system.hide('pnext') 



#pupdate.register_concrete(func, hooks)?

"""

@input(mgr, 'p', (-10,10), precision = 6)
@input(mgr, 'v', (-20,20), precision = 6)
@output(mgr, (0, 'p_next'), (-10,10), precision = 6)
def pupdate(p,v) -> (float):
    return p + v*ts 

@input(mgr, 'v', (-20,20), precision = 6)
@input(mgr, 'a', (-20,20), precision = 6)
@output(mgr, (0, 'v_next'), (-20,20), precision = 6)
def vupdate(v,a) -> (float):
    vsign = 1 if v > 0 else -1 
    return v + a*ts - vsign*k*(v**2)*ts - g*ts

# sys = pupdate | vupdate

bounds = {'p': [-10,10], 'v': [-20,20]}

fig = plt.figure()
ax = fig.gca(projection='3d') 

# Sample generator
numapplied = 0
print(system.count_io())
while(numapplied < 200): 
    # Generate random windows 
    f_width = {'p': np.random.rand()*12,
            'v':np.random.rand()*12,
            'a':np.random.rand()*12}
    f_left = {'p': -10 +  np.random.rand() * (20 - f_width['p']), 
            'v': -20 + np.random.rand() * (40 - f_width['v']), 
            'a': -20 + np.random.rand() * (40 - f_width['a'])}
    f_right = {k: f_width[k] + f_left[k] for k in f_width}
    inbox = {k: (f_left[k], f_right[k]) for k in f_width}


    # Generate output overapproximation 
    ll = system(**f_left)
    ur = system(**f_right)

    # # Detect output domain violation 
    # if ll[0] < bounds['p'][0] or ll[1] < boundss['v'][0]:
    #     continue
    # if ur[0] > bounds['p'][1] or ur[1] > bounds['v'][1]:
    #     continue

    # Constrain 
    # TODO: Constrain along slices of the I/O space for lower dim objects 
    outbox = {i: (ll[i], ur[i]) for i in range(2)}
    try:
        system.apply_io_constraint((inbox, outbox))
    except:
        continue
    numapplied += 1
    
    p,v,a = np.indices(((2,2,2)))/1.0
    p[0,:,:] += inbox['p'][0] 
    p[1,:,:] *= inbox['p'][1]
    v[:,0,:] += inbox['v'][0]
    v[:,1,:] *= inbox['v'][1]
    a[:,:,0] += inbox['a'][0]
    a[:,:,1] *= inbox['a'][1]

    color = '#7A88CC' + '03' # color  + opacity 
    ax.voxels(p,v,a, 
              np.array([[[True]]]), 
              facecolors =  np.array([[[color]]])
              )

    if numapplied % 500 == 0:
        print(system.count_io())

ax.set_xlim(-10,10)
ax.set_ylim(-20,20)
ax.set_zlim(-20,20)
plt.show()

print("Final: ", system.count_io())
