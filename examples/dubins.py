"""
Dubins vehicle example 
"""


from dd.cudd import BDD

import numpy as np 

from vpax.module import AbstractModule
from vpax.spaces import DynamicPartition, FixedPartition, EmbeddedGrid
from vpax.controlmodule import to_control_module
from vpax.synthesis import ReachGame
from vpax.visualizer import plot3D_QT, plot3D

import time

"""
Specify dynamics and overapproximations 
"""

L= 1.4
vmax = .5
def dynamics(x,y,theta,v,omega):
    return x + v*np.cos(theta), y + v*np.sin(theta), theta + (1/L) * v * np.sin(omega)

def containszero(left, right):
    """
    Determines if 0 is contained in a periodic interval [left,right]. Possible that right < left. 
    """
    
    # Map to interval [-pi,pi]
    left = ((left + np.pi) % (np.pi*2) ) - np.pi
    left = left - 2*np.pi if left > np.pi else left
    right = ((right + np.pi) % (np.pi*2) ) - np.pi
    right = right - 2*np.pi if right > np.pi else right

    if right < left:
        if left <= 0:
            return True
    else: 
        if left <= 0 and right >= 0:
            return True
    
    return False

def maxmincos(left, right):
    """
    Given an interval, compute the maximum and minimum values of cos in that window 
    """
    if containszero(left,right) is True:
        maxval = 1
    else:
        maxval = max([np.cos(left), np.cos(right)])

    if containszero(left + np.pi, right + np.pi) is True:
        minval = -1
    else:
        minval = min([np.cos(left), np.cos(right)])

    return (minval, maxval)

def maxminsin(left, right):
    return maxmincos(left - np.pi/2, right - np.pi/2)

# Overapproximations of next states for each component
# v and omega are points and not windows
def xwindow(x,v,theta):
    mincos, maxcos = maxmincos(theta[0], theta[1])
    return x[0] + v * mincos, x[1] + v * maxcos
def ywindow(y,v,theta):
    minsin, maxsin = maxminsin(theta[0], theta[1])
    return y[0] + v * minsin, y[1] + v * maxsin
def thetawindow(theta, v, omega):
    return theta[0] + (1/L) * v * np.sin(omega), theta[1] + (1/L) * v*np.sin(omega)

outorder = {0: 'xnext', 1: 'ynext', 2: 'thetanext'}

"""
Declare modules 
"""

mgr = BDD() 

pspace = DynamicPartition(-2,2)
anglespace = DynamicPartition(-np.pi, np.pi, periodic=True)
vspace = EmbeddedGrid(vmax/2, vmax, 2)
angaccspace = EmbeddedGrid(-1.5, 1.5, 3)

dubins_x = AbstractModule(mgr, 
                          {'x': pspace,
                           'theta': anglespace,
                           'v': vspace}, 
                           {'xnext': pspace})
dubins_y = AbstractModule(mgr, 
                          {'y': pspace,
                           'theta': anglespace,
                           'v': vspace}, 
                           {'ynext': pspace})
dubins_theta =  AbstractModule(mgr, 
                              {'theta': anglespace,
                              'v': vspace,
                              'omega': angaccspace}
                              ,{'thetanext': anglespace})

dubins = (dubins_x | dubins_y | dubins_theta)

precision = {'x': 6,
             'y':6,
             'theta': 5,
             'xnext': 6,
             'ynext': 6,
             'thetanext': 5}
bittotal = sum(precision.values()) + 3 # +3 for the embedded grid bits 
possible_transitions = dubins.count_io(bittotal)

coarseiter = 0
coarse_x_errors = 0
coarse_y_errors = 0
coarse_theta_errors = 0 
errorboxes = []
for iobox in dubins.input_iter(precision = {'x':4, 'y':4, 'theta':3}):
    # Generate output windows
    iobox['xnext']     = xwindow(**{k:v for k,v in iobox.items() if k in dubins_x.inputs})
    iobox['ynext']     = ywindow(**{k:v for k,v in iobox.items() if k in dubins_y.inputs})
    iobox['thetanext'] = thetawindow(**{k:v for k,v in iobox.items() if k in dubins_theta.inputs})

    # Add transitions 
    try:
        dubins_x.pred &= dubins_x.ioimplies2pred({k:v for k,v in iobox.items() if k in dubins_x.vars},
                                             precision = precision)
    except AssertionError:
        coarse_x_errors += 1
        continue

    try:
        dubins_y.pred &= dubins_y.ioimplies2pred({k:v for k,v in iobox.items() if k in dubins_y.vars},
                                            precision = precision)
    except AssertionError:
        coarse_y_errors +=1

    try: 
        dubins_theta.pred &= dubins_theta.ioimplies2pred({k:v for k,v in iobox.items() if k in dubins_theta.vars},
                                             precision = precision)
    except AssertionError:
        coarse_theta_errors += 1 

    coarseiter += 1
    if coarseiter % 2000 == 0: 
        dubins = (dubins_x | dubins_y | dubins_theta) 
        iotrans = dubins.count_io(bittotal)
        print("(samples, I/O % transitions, bddsize) --- ({0}, {1:.3}, {2})".format(coarseiter, 
                                                            100*iotrans/possible_transitions,
                                                            len(dubins.pred)))


# Sample generator 
numapplied = 0
out_of_domain_violations = 0
abs_starttime = time.time()
while(numapplied < 40000):

    # Shrink window widths over time 
    scale = 1/np.log10(1.0*numapplied+10)
    
    # Generate random input windows and points
    f_width = {'x':     np.random.rand()*scale*pspace.width(),
               'y':     np.random.rand()*scale*pspace.width(),
               'theta': .2*anglespace.width()} 
    f_left = {'x':     pspace.lb +  np.random.rand() * (pspace.width() - f_width['x']),
              'y':     pspace.lb +  np.random.rand() * (pspace.width() - f_width['y']),
              'theta': anglespace.lb + np.random.rand() * (anglespace.width())}
    f_right = {k: f_width[k] + f_left[k] for k in f_width}
    iobox = {'v':     np.random.randint(1,3)*vmax/2,
             'omega': np.random.randint(-1,2)*1.5}
    iobox.update({k: (f_left[k], f_right[k]) for k in f_width})

    # Generate output windows
    iobox['xnext']     = xwindow(**{k:v for k,v in iobox.items() if k in dubins_x.inputs})
    iobox['ynext']     = ywindow(**{k:v for k,v in iobox.items() if k in dubins_y.inputs})
    iobox['thetanext'] = thetawindow(**{k:v for k,v in iobox.items() if k in dubins_theta.inputs})

    # Add transitions 
    try:
        dubins_x.pred &= dubins_x.ioimplies2pred({k:v for k,v in iobox.items() if k in dubins_x.vars},
                                             precision = precision)
        dubins_y.pred &= dubins_y.ioimplies2pred({k:v for k,v in iobox.items() if k in dubins_y.vars},
                                             precision = precision)
        dubins_theta.pred &= dubins_theta.ioimplies2pred({k:v for k,v in iobox.items() if k in dubins_theta.vars},
                                             precision = precision)
    except AssertionError:
        out_of_domain_violations += 1
        continue

    numapplied += 1
    
    if numapplied % 4000 == 0:
        dubins = (dubins_x | dubins_y | dubins_theta)
        iotrans = dubins.count_io(bittotal)
        print("(samples, I/O % trans., bddsize, s) --- ({0}, {1:.3}, {2}, {3})".format(numapplied, 
                                                            100*iotrans/possible_transitions,
                                                            len(dubins.pred),
                                                            time.time() - abs_starttime))
dubins = (dubins_x | dubins_y | dubins_theta)

csys = to_control_module(dubins, (('x', 'xnext'), ('y','ynext'), ('theta','thetanext')))

# Declare reach set 
target =  pspace.conc2pred(mgr, 'x', [-.6,.6], 5, innerapprox = True)
target &= pspace.conc2pred(mgr, 'y', [-.4,.4], 5, innerapprox = True)

game = ReachGame(csys, target)
starttime = time.time()
basin, steps = game.step()
print("Solve Time:", time.time() - starttime)
print("Reach Size:", dubins.mgr.count(basin, 17))
print("Target Size:", dubins.mgr.count(target, 17))
print("Game Steps:", steps)

plot3D_QT(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace), basin, 128)

# Plot x transition relation for v = .5 
# a = mgr.exist(['v_0'],(dubins_x.pred) & mgr.var('v_0') & ~dubins_x.constrained_inputs())
# plot3D_QT(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), a, 128) 

# Plot y transition relation for v = .5 
# a = mgr.exist(['v_0'],(dubins_y.pred) & mgr.var('v_0') & ~dubins_y.constrained_inputs())
# plot3D_QT(mgr, ('y', pspace),('theta', anglespace), ('ynext', pspace), a, 128)