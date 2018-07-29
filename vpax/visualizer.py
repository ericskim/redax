
# Visualize BDD sets in 2D or 3D

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 

def _name(i):
    return i.split('_')[0]

def _idx(i):
    return i.split('_')[1] 

def center(box):
    l,r = box
    return l + (r-l)/2.0

#Organize into bitvectors 
def plot2D(mgr, xspace, yspace, pred):
    """

    Args: 
        mgr:    dd manager
        xspace: (name: str, symbolicinterval)
        yspace: (name: str, symbolicinterval)
        pred:   dd's bdd object over xspace and yspace 

    Return:
        fig: matplotlib figure
        ax: matplotlib axis
    """
    xname, xgrid = xspace
    yname, ygrid = yspace

    xpts = []
    ypts = []

    # Add all BDD assignments to a list of points 
    for pt in mgr.pick_iter(pred):
        xvars = [k for k,v in pt.items() if _name(k) == xname ]
        yvars = [k for k,v in pt.items() if _name(k) == yname ]
        xvars.sort() # Sort by bit names
        yvars.sort() 
        
        xbv = [pt[bit] for bit in xvars]
        ybv = [pt[bit] for bit in yvars]

        xpts.append(xgrid.bv2box(xbv)) 
        ypts.append(ygrid.bv2box(ybv))
    
    fig, ax = plt.subplots()
    ax.scatter(xpts,ypts)

    ax.set_xlim(xgrid.lb, xgrid.ub)
    ax.set_ylim(ygrid.lb, ygrid.ub)

    plt.show() 
    return fig, ax 
