
# Visualize BDD sets in 2D or 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from vpax.spaces import _bv2int, _graytobin, DynamicPartition


def _name(i):
    return i.split('_')[0]


def _idx(i):
    return i.split('_')[1]


def center(box):
    l, r = box
    return l + (r-l)/2.0


def centerspace(space):
    return space.lb + (space.ub - space.lb)/2.0


def bv2int(bv):
    """Converts bitvector (list or tuple) into an integer"""
    nbits = len(bv)
    index = 0
    for i in range(nbits):
        if bv[i]:
            index += 2**(nbits - i - 1)
    return index


def dynamicperiodic(space):
    if isinstance(space, DynamicPartition) and space.periodic is True:
        return True
    return False


# Organize into bitvectors
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
        xvars = [k for k, v in pt.items() if _name(k) == xname]
        yvars = [k for k, v in pt.items() if _name(k) == yname]
        xvars.sort()  # Sort by bit names
        yvars.sort()

        xbv = [pt[bit] for bit in xvars]
        ybv = [pt[bit] for bit in yvars]

        xpts.append(xgrid.bv2conc(xbv))
        ypts.append(ygrid.bv2conc(ybv))

    fig, ax = plt.subplots()
    ax.scatter(xpts, ypts)

    ax.set_xlim(xgrid.lb, xgrid.ub)
    ax.set_xlabel(xname)
    ax.set_ylim(ygrid.lb, ygrid.ub)
    ax.set_ylabel(yname)

    plt.show()
    return fig, ax


def plot3D(mgr, xspace, yspace, zspace, pred, opacity=40):
    """Matplotlib based plotter with voxels"""
    voxelcolors = '#7A88CC' + format(opacity, "02x")
    xname, xgrid = xspace
    yname, ygrid = yspace
    zname, zgrid = zspace

    # Construct spaces
    support = pred.support
    xbits = len([bit for bit in support if _name(bit) == xname])
    ybits = len([bit for bit in support if _name(bit) == yname])
    zbits = len([bit for bit in support if _name(bit) == zname])
    xbins = 2**xbits
    ybins = 2**ybits
    zbins = 2**zbits
    # Voxel corners
    x, y, z = np.indices((xbins+1, ybins+1, zbins+1))
    x = (x * (xgrid.ub - xgrid.lb)/xbins) + xgrid.lb
    y = (y * (ygrid.ub - ygrid.lb)/ybins) + ygrid.lb
    z = (z * (zgrid.ub - zgrid.lb)/zbins) + zgrid.lb

    # Construct bitmask
    mask = np.full((xbins, ybins, zbins), False)
    for pt in mgr.pick_iter(pred):
        xvars = [k for k, v in pt.items() if _name(k) == xname]
        yvars = [k for k, v in pt.items() if _name(k) == yname]
        zvars = [k for k, v in pt.items() if _name(k) == zname]
        xvars.sort()  # Sort by bit names
        yvars.sort()
        zvars.sort()

        xbv = [pt[bit] for bit in xvars]
        ybv = [pt[bit] for bit in yvars]
        zbv = [pt[bit] for bit in zvars]

        x_idx = _bv2int(xbv) if not dynamicperiodic(xgrid) else _graytobin(_bv2int(xbv))
        y_idx = _bv2int(ybv) if not dynamicperiodic(ygrid) else _graytobin(_bv2int(ybv))
        z_idx = _bv2int(zbv) if not dynamicperiodic(zgrid) else _graytobin(_bv2int(zbv))

        mask[x_idx, y_idx, z_idx] = True

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(x, y, z, mask, facecolors=voxelcolors)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_zlabel(zname)
    plt.show()

    return fig, ax


def plot3D_QT(mgr, xspace, yspace, zspace, pred, opacity=255):
    """Somewhat buggy pyqtgraph plotting"""
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl

    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 200
    w.show()
    # w.setWindowTitle('pyqtgraph example: GLVolumeItem')

    # print(w.width())
    # print(w.height())
    # w.setFixedWidth(640)
    # w.setFixedHeight(480)
    w.resizeGL(1600, 1200)

    xname, xgrid = xspace
    yname, ygrid = yspace
    zname, zgrid = zspace

    # Construct spaces
    support = pred.support
    xbits = len([bit for bit in support if _name(bit) == xname])
    ybits = len([bit for bit in support if _name(bit) == yname])
    zbits = len([bit for bit in support if _name(bit) == zname])
    xbins = 2**xbits
    ybins = 2**ybits
    zbins = 2**zbits

    # Construct bitmask
    mask = np.full((xbins, ybins, zbins), False)
    xpts = []
    ypts = []
    zpts = []
    for pt in mgr.pick_iter(pred):
        xvars = [k for k, v in pt.items() if _name(k) == xname]
        yvars = [k for k, v in pt.items() if _name(k) == yname]
        zvars = [k for k, v in pt.items() if _name(k) == zname]
        xvars.sort()  # Sort by bit names
        yvars.sort()
        zvars.sort()

        xbv = [pt[bit] for bit in xvars]
        ybv = [pt[bit] for bit in yvars]
        zbv = [pt[bit] for bit in zvars]

        x_idx = _bv2int(xbv) if not dynamicperiodic(xgrid) else _graytobin(_bv2int(xbv))
        y_idx = _bv2int(ybv) if not dynamicperiodic(ygrid) else _graytobin(_bv2int(ybv))
        z_idx = _bv2int(zbv) if not dynamicperiodic(zgrid) else _graytobin(_bv2int(zbv))

        mask[x_idx, y_idx, z_idx] = True

    d2 = np.empty(mask.shape + (4,), dtype=np.ubyte)
    d2[..., 0] = mask.astype(np.float) * 255
    d2[..., 1] = mask.astype(np.float) * 255
    d2[..., 2] = mask.astype(np.float) * 255
    d2[..., 3] = mask.astype(np.float) * opacity

    v = gl.GLVolumeItem(d2, smooth=False)
    v.translate(-xbins//2,
                -ybins//2,
                -zbins//2)
    w.addItem(v)

    g = gl.GLGridItem()
    g.scale(xbins//10, ybins//10, zbins//10)
    w.addItem(g)

    ax = gl.GLAxisItem()
    w.addItem(ax)

    QtGui.QApplication.instance().exec_()
