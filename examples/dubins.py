"""
Dubins vehicle example 
"""

import time

import numpy as np
from dd.cudd import BDD

from vpax.controlmodule import to_control_module
from vpax.module import AbstractModule
from vpax.spaces import DynamicPartition, EmbeddedGrid, FixedPartition
from vpax.synthesis import ReachGame
from vpax.visualizer import plot3D, plot3D_QT

"""
Specify dynamics and overapproximations 
"""

L= 1.4
vmax = .5
def dynamics(x,y,theta,v,omega):
    return x + v*np.cos(theta), y + v*np.sin(theta), theta + (1/L) * v * np.sin(omega)

def containszero(left, right):
    """Determine if 0 is contained in a periodic interval [left,right]. Possible that right < left."""
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
    """Compute the maximum and minimum values of cos in an interval."""
    if containszero(left, right) is True:
        maxval = 1
    else:
        maxval = max([np.cos(left), np.cos(right)])

    if containszero(left + np.pi, right + np.pi) is True:
        minval = -1
    else:
        minval = min([np.cos(left), np.cos(right)])

    return (minval, maxval)

def maxminsin(left, right):
    """Compute the maximum and minimum values of sin in an interval."""
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

# Declare continuous state spaces
pspace      = DynamicPartition(-2,2)
anglespace  = DynamicPartition(-np.pi, np.pi, periodic=True)
# Declare discrete control spaces 
vspace      = EmbeddedGrid(vmax/2, vmax, 2)
angaccspace = EmbeddedGrid(-1.5, 1.5, 3)

# Declare modules
dubins_x        = AbstractModule(mgr, {'x': pspace, 'theta': anglespace, 'v': vspace}, 
                                      {'xnext': pspace})
dubins_y        = AbstractModule(mgr, {'y': pspace, 'theta': anglespace, 'v': vspace}, 
                                      {'ynext': pspace})
dubins_theta    = AbstractModule(mgr, {'theta': anglespace, 'v': vspace, 'omega': angaccspace}, 
                                      {'thetanext': anglespace})

dubins = (dubins_x | dubins_y | dubins_theta)

precision = {'x': 6, 'y':6, 'theta': 6,
             'xnext': 6, 'ynext': 6, 'thetanext': 6}
bittotal = sum(precision.values()) + 3 # +3 for the discrete embedded grid bits
possible_transitions = dubins.count_io_space(bittotal)

coarseiter = 0
coarse_errors = {'x': 0, 'y': 0, 'theta': 0}
abs_starttime = time.time()
for iobox in dubins.input_iter(precision={'x': 4, 'y': 4, 'theta': 3}):
    # Generate output windows
    iobox['xnext']     = xwindow(**{k: v for k, v in iobox.items() if k in dubins_x.inputs})
    iobox['ynext']     = ywindow(**{k: v for k, v in iobox.items() if k in dubins_y.inputs})
    iobox['thetanext'] = thetawindow(**{k: v for k, v in iobox.items() if k in dubins_theta.inputs})
    
    # Add new inputs and constrain output nondeterminism
    for var, sys in {'x': dubins_x, 'y': dubins_y, 'theta': dubins_theta}.items():
        filtered_iobox = {k: v for k, v in iobox.items() if k in sys.vars}
        if not sys.apply_abstract_transitions(filtered_iobox, nbits=precision):
            coarse_errors[var] += 1

    coarseiter += 1
    if coarseiter % 2000 == 0: 
        dubins = (dubins_x | dubins_y | dubins_theta) 
        iotrans = dubins.count_io(bittotal)
        print("(samples, I/O % trans., bddsize, time(s)) --- ({0}, {1:.3}, {2}, {3})".format(coarseiter, 
                                                    100*iotrans/possible_transitions,
                                                    len(dubins.pred),
                                                    time.time() - abs_starttime))                                                            

# Sample generator 
random_errors = {'x': 0, 'y': 0, 'theta': 0}
for numapplied in range(5000):
    
    # Shrink window widths over time 
    scale = 1/np.log10(1.0*numapplied+10)
    
    # Generate random input windows and points
    f_width = {'x':     np.random.rand()*scale*pspace.width(),
               'y':     np.random.rand()*scale*pspace.width(),
               'theta': .2*anglespace.width()} 
    f_left = {'x':     pspace.lb + np.random.rand() * (pspace.width() - f_width['x']),
              'y':     pspace.lb + np.random.rand() * (pspace.width() - f_width['y']),
              'theta': anglespace.lb + np.random.rand() * (anglespace.width())}
    f_right = {k: f_width[k] + f_left[k] for k in f_width}
    iobox = {'v':     np.random.randint(1, 3) * vmax/2,
             'omega': np.random.randint(-1, 2) * 1.5}
    iobox.update({k: (f_left[k], f_right[k]) for k in f_width})

    # Generate output windows
    iobox['xnext']     = xwindow(**{k: v for k, v in iobox.items() if k in dubins_x.inputs})
    iobox['ynext']     = ywindow(**{k: v for k, v in iobox.items() if k in dubins_y.inputs})
    iobox['thetanext'] = thetawindow(**{k: v for k, v in iobox.items() if k in dubins_theta.inputs})

    # Add new inputs and constrain output nondeterminism
    for var, sys in {'x': dubins_x, 'y': dubins_y, 'theta': dubins_theta}.items():
        filtered_iobox = {k: v for k, v in iobox.items() if k in sys.vars}
        if not sys.apply_abstract_transitions(filtered_iobox, nbits=precision):
            random_errors[var] += 1

    numapplied += 1
    
    if numapplied % 1000 == 0:
        dubins = (dubins_x | dubins_y | dubins_theta)
        iotrans = dubins.count_io(bittotal)
        print("(samples, I/O % trans., bddsize, time(s)) --- ({0}, {1:.3}, {2}, {3})".format(numapplied, 
                                                    100*iotrans/possible_transitions,
                                                    len(dubins.pred),
                                                    time.time() - abs_starttime))

dubins = (dubins_x | dubins_y | dubins_theta)

csys = to_control_module(dubins, (('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')))

# Declare reach set 
target =  pspace.conc2pred(mgr, 'x', [0, .8], 6, innerapprox=False)
target &= pspace.conc2pred(mgr, 'y', [-.8, 0], 6, innerapprox=False)

game = ReachGame(csys, target)
starttime = time.time()
basin, steps = game.step()
print("Solve Time:", time.time() - starttime)
print("Reach Size:", dubins.mgr.count(basin, 18))
print("Target Size:", dubins.mgr.count(target, 18))
print("Game Steps:", steps)

# # Plot reachable winning set
# plot3D_QT(mgr, ('x', pspace), ('y', pspace), ('theta', anglespace), basin, 128)

# # Plot x transition relation for v = .5
# xdyn = mgr.exist(['v_0'],(dubins_x.pred) & mgr.var('v_0'))
# plot3D_QT(mgr, ('x', pspace),('theta', anglespace), ('xnext', pspace), xdyn, 128)

# # Plot y transition relation for v = .5
# ydyn = mgr.exist(['v_0'],(dubins_y.pred) & mgr.var('v_0'))
# plot3D_QT(mgr, ('y', pspace),('theta', anglespace), ('ynext', pspace), ydyn, 128)
