
"""Visualize BDD sets in 2D or 3D."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Generator

import funcy as fn
from redax.spaces import DynamicCover
from redax.utils.bv import bv2int, graytobin, bv_var_name, bv_var_idx, var_name

#FIXME: Visualizers raise errors for true/false predicates!!!

def center(box):
    l, r = box
    return l + (r-l)/2.0


def centerspace(space):
    return space.lb + (space.ub - space.lb)/2.0


def dynamicperiodic(space) -> bool:
    if isinstance(space, DynamicCover) and space.periodic is True:
        return True
    return False


def partial_assign_complete(partial: dict, missing_bits: set) -> Generator[dict, None, None]:
    """ Generator that takes a partial assignment and yields all satisfying assignments."""

    # Missing bits can't be assigned!
    assigned_bits = set(partial.keys())
    assert assigned_bits.isdisjoint(missing_bits)

    if len(missing_bits) == 0:
        yield partial
    else:

        # Copies required because recursion mutates the missing_bit set
        _partial = partial.copy()
        _missing_bits = missing_bits.copy()

        # Pick an unassigned bit name
        bitvar = fn.first(_missing_bits)
        _missing_bits.remove(bitvar)

        # Iterate over bit assignments
        for val in [True, False]:
            _partial[bitvar] = val
            for totalassign in partial_assign_complete(_partial, _missing_bits):
                yield totalassign


def pixel2D(mgr, xspace, yspace, pred, title=None, fname=None, invertcolor=False, opacity=1.0, raisebiterror=False) -> None:
    """
    Plot a 2D set represented by a predicate as a pixel image.

    TODO: Return handle to matplotlib figure?

    Parameters
    ----------
    mgr : dd manager
    xspace : tuple (str, symbolic space)
        horizontal axis name and space
    yspace : tuple (str, symbolic space)
        vertical axis name and space
    pred : bdd
        Predicate representing set over xspace and yspace

    Returns
    -------
    matplotlib figure
    matplotlib axis

    """
    xname, xgrid = xspace
    yname, ygrid = yspace

    support = pred.support
    xbits = max([int(bv_var_idx(bit)) for bit in support if bv_var_name(bit) == xname], default=-1) + 1
    ybits = max([int(bv_var_idx(bit))for bit in support if bv_var_name(bit) == yname], default=-1) + 1
    xbins = 2**xbits
    ybins = 2**ybits

    xbitvarset = set(var_name(xname, i) for i in range(xbits))
    ybitvarset = set(var_name(yname, i) for i in range(ybits))
    bitvarset = xbitvarset.union(ybitvarset)

    assignment = fn.first(mgr.pick_iter(pred))
    assignment_bits = set(assignment.keys())
    missing_bits = bitvarset.difference(assignment_bits)
    if len(missing_bits) > 0 and raisebiterror:
        raise ValueError("Missing bits: {}".format(missing_bits))

    if invertcolor:
        mask = np.full((ybins, xbins), 111-(111*opacity), dtype=np.int8)
    else:
        mask = np.full((ybins, xbins), 111*opacity, dtype=np.int8)

    config = mgr.configure()
    # Add all BDD assignments to a list of points
    for partial in mgr.pick_iter(pred):
        for pt in partial_assign_complete(partial, missing_bits):
            xvars = [k for k, v in pt.items() if bv_var_name(k) == xname]
            yvars = [k for k, v in pt.items() if bv_var_name(k) == yname]
            xvars.sort()  # Sort by bit names
            yvars.sort()


            xbv = [False] * xbits
            for i in range(xbits):
                bitname = xname + '_' + str(i)
                xbv[i] = pt[bitname] if bitname in pt else False

            ybv = [False] * ybits
            for i in range(ybits):
                bitname = yname + '_' + str(i)
                ybv[i] = pt[bitname] if bitname in pt else False


            ycoord = bv2int(ybv)
            xcoord = bv2int(xbv)
            if ygrid.periodic:
                ycoord = graytobin(ycoord)
            if xgrid.periodic:
                xcoord = graytobin(xcoord)

            if invertcolor:
                mask[ycoord, xcoord] = 111*opacity
            else:
                mask[ycoord, xcoord] = 111-(111*opacity)
    mgr.configure(reordering=config['reordering'])  # Reinstate config state

    fig, ax = plt.subplots()
    if invertcolor:
        ax.pcolormesh(mask, edgecolors='.45', cmap = plt.cm.bone, vmin=0, vmax = 111, linewidths=.01)
    else:
        ax.pcolormesh(mask, edgecolors='.65', cmap = plt.cm.bone, vmin=0, vmax = 111, linewidths=.01)

    # ax.set_xlim(xgrid.lb, xgrid.ub)
    ax.set_xlabel(xname)
    # ax.set_ylim(ygrid.lb, ygrid.ub)
    ax.set_ylabel(yname)

    if title:
        ax.set_title(title)
    if fname is not None:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        fig.savefig(str(fname)+'.png', dpi=400, bbox_inches=extent.expanded(1.0, 1.0))
    else:
        plt.show()

    plt.close()
    return
    # return fig, ax

# Organize into bitvectors
def scatter2D(mgr, xspace, yspace, pred, title=None, fname=None, fig = None, ax = None, alpha=None, co=None, raisebiterror=False) -> None:
    """
    Plot a 2D set represented by a predicate as a scatter plot.

    Parameters
    ----------
    mgr : dd manager
    xspace : tuple (str, symbolic space)
        horizontal axis name and space
    yspace : tuple (str, symbolic space)
        vertical axis name and space
    pred : bdd
        Predicate representing set over xspace and yspace

    Returns
    -------
    matplotlib figure
    matplotlib axis

    """
    xname, xgrid = xspace
    yname, ygrid = yspace

    support = pred.support
    xbits = max([int(bv_var_idx(bit)) for bit in support if bv_var_name(bit) == xname], default=-1) + 1
    ybits = max([int(bv_var_idx(bit))for bit in support if bv_var_name(bit) == yname], default=-1) + 1

    xbitvarset = set(var_name(xname, i) for i in range(xbits))
    ybitvarset = set(var_name(yname, i) for i in range(ybits))
    bitvarset = xbitvarset.union(ybitvarset)

    assignment_bits = set(fn.first(mgr.pick_iter(pred)).keys())
    missing_bits = bitvarset.difference(assignment_bits)
    if len(missing_bits) > 0 and raisebiterror:
        raise ValueError("Missing bits: {}".format(missing_bits))


    xpts = []
    ypts = []

    config = mgr.configure()  # pick_iter alters config so save config state
    # Add all BDD assignments to a list of points
    for partial in mgr.pick_iter(pred):
        for pt in partial_assign_complete(partial, missing_bits):
            xvars = [k for k, v in pt.items() if bv_var_name(k) == xname]
            yvars = [k for k, v in pt.items() if bv_var_name(k) == yname]
            xvars.sort()  # Sort by bit names
            yvars.sort()


            xbv = [False] * xbits
            for i in range(xbits):
                bitname = xname + '_' + str(i)
                xbv[i] = pt[bitname] if bitname in pt else False

            ybv = [False] * ybits
            for i in range(ybits):
                bitname = yname + '_' + str(i)
                ybv[i] = pt[bitname] if bitname in pt else False

            xpts.append(xgrid.bv2conc(xbv))
            ypts.append(ygrid.bv2conc(ybv))
    mgr.configure(reordering=config['reordering'])  # Reinstate config state

    if fig is None:
        fig, ax = plt.subplots()
    ax.scatter(xpts, ypts, marker='s', alpha=alpha, c=co)

    ax.set_xlim(xgrid.lb, xgrid.ub)
    ax.set_xlabel(xname)
    ax.set_ylim(ygrid.lb, ygrid.ub)
    ax.set_ylabel(yname)
    if title:
        ax.set_title(title)
    if fname is not None:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(str(fname)+'.png', dpi=400)#, bbox_inches=extent.expanded(1.1, 1.2))
    else:
        plt.show()

    # return fig, ax


def plot3D(mgr, xspace, yspace, zspace, pred, raisebiterror=False,
           opacity=40, view=None, title=None, fname=None, **kwargs):
    """Matplotlib based plotter with voxels."""
    voxelcolors = '#7A88CC' + format(opacity, "02x")
    # voxelcolors = '#FFFFFF' + format(opacity, "02x")
    # voxelcolors = '#000000' + format(opacity, "02x")
    edgecolors =  '#000000' + format(opacity // 3, "02x")
    xname, xgrid = xspace
    yname, ygrid = yspace
    zname, zgrid = zspace

    # Construct spaces
    support = pred.support
    xbits = max([int(bv_var_idx(bit)) for bit in support if bv_var_name(bit) == xname], default=-1) + 1
    ybits = max([int(bv_var_idx(bit))for bit in support if bv_var_name(bit) == yname], default=-1) + 1
    zbits = max([int(bv_var_idx(bit)) for bit in support if bv_var_name(bit) == zname], default=-1) + 1
    xbins = 2**xbits
    ybins = 2**ybits
    zbins = 2**zbits

    xbitvarset = set(var_name(xname, i) for i in range(xbits))
    ybitvarset = set(var_name(yname, i) for i in range(ybits))
    zbitvarset = set(var_name(zname, i) for i in range(zbits))
    bitvarset = xbitvarset.union(ybitvarset).union(zbitvarset)

    # FIXME: Raises error for false interface!!!
    assignment_bits = set(fn.first(mgr.pick_iter(pred)).keys())
    missing_bits = bitvarset.difference(assignment_bits)
    if len(missing_bits) > 0 and raisebiterror:
        raise ValueError("Missing bits: {}".format(missing_bits))


    # Voxel corners
    x, y, z = np.indices((xbins+1, ybins+1, zbins+1))
    x = (x * (xgrid.ub - xgrid.lb)/xbins) + xgrid.lb
    y = (y * (ygrid.ub - ygrid.lb)/ybins) + ygrid.lb
    z = (z * (zgrid.ub - zgrid.lb)/zbins) + zgrid.lb

    config = mgr.configure()  # pick_iter alters config so save config state

    # Construct bitmask
    mask = np.full((xbins, ybins, zbins), False)
    for partial in mgr.pick_iter(pred):
        for pt in partial_assign_complete(partial, missing_bits):
            xvars = [k for k, v in pt.items() if bv_var_name(k) == xname]
            yvars = [k for k, v in pt.items() if bv_var_name(k) == yname]
            zvars = [k for k, v in pt.items() if bv_var_name(k) == zname]
            xvars.sort()  # Sort by bit names
            yvars.sort()
            zvars.sort()

            xbv = [False] * xbits
            for i in range(xbits):
                bitname = xname + '_' + str(i)
                xbv[i] = pt[bitname] if bitname in pt else False

            ybv = [False] * ybits
            for i in range(ybits):
                bitname = yname + '_' + str(i)
                ybv[i] = pt[bitname] if bitname in pt else False

            zbv = [False] * zbits
            for i in range(zbits):
                bitname = zname + '_' + str(i)
                zbv[i] = pt[bitname] if bitname in pt else False

            x_idx = bv2int(xbv) if not dynamicperiodic(xgrid) else graytobin(bv2int(xbv))
            y_idx = bv2int(ybv) if not dynamicperiodic(ygrid) else graytobin(bv2int(ybv))
            z_idx = bv2int(zbv) if not dynamicperiodic(zgrid) else graytobin(bv2int(zbv))

            mask[x_idx, y_idx, z_idx] = True

    mgr.configure(reordering=config['reordering'])  # Reinstate config state

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(x, y, z, mask, facecolors=voxelcolors, edgecolor=edgecolors)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_zlabel(zname)
    # ax.patch.set_facecolor('black')

    if view:
        ax.view_init(view[0], view[1])
    if title:
        ax.set_title(title)
    if fname:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(str(fname)+'.png', dpi=400, bbox_inches=extent.expanded(1.1, 1.2))
    else:
        plt.show()

    # return fig, ax


def plot3D_QT(mgr, xspace, yspace, zspace, pred, opacity=255, fname=None, raisebiterror=False):
    """Plot with Pyqtgraph.

    Assumes that bit variables come in the form "{name}_{index}" where indices
    start from 0.

    """
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
    xbits = max([int(bv_var_idx(bit)) for bit in support if bv_var_name(bit) == xname], default=-1) + 1
    ybits = max([int(bv_var_idx(bit)) for bit in support if bv_var_name(bit) == yname], default=-1) + 1
    zbits = max([int(bv_var_idx(bit)) for bit in support if bv_var_name(bit) == zname], default=-1) + 1
    xbins = 2**xbits
    ybins = 2**ybits
    zbins = 2**zbits

    xbitvarset = set(var_name(xname, i) for i in range(xbits))
    ybitvarset = set(var_name(yname, i) for i in range(ybits))
    zbitvarset = set(var_name(zname, i) for i in range(zbits))
    bitvarset = xbitvarset.union(ybitvarset).union(zbitvarset)

    assignment_bits = set(fn.first(mgr.pick_iter(pred)).keys())
    missing_bits = bitvarset.difference(assignment_bits)
    if len(missing_bits) > 0 and raisebiterror:
        raise ValueError("Missing bits: {}".format(missing_bits))

    # Construct bitmask
    mask = np.full((xbins, ybins, zbins), False)

    config = mgr.configure()  # pick_iter alters config so save config state

    for partial in mgr.pick_iter(pred):
        for pt in partial_assign_complete(partial, missing_bits):
            xvars = [k for k, v in pt.items() if bv_var_name(k) == xname]
            yvars = [k for k, v in pt.items() if bv_var_name(k) == yname]
            zvars = [k for k, v in pt.items() if bv_var_name(k) == zname]
            xvars.sort()  # Sort by bit names
            yvars.sort()
            zvars.sort()

            xbv = [False] * xbits
            for i in range(xbits):
                bitname = xname + '_' + str(i)
                xbv[i] = pt[bitname] if bitname in pt else False

            ybv = [False] * ybits
            for i in range(ybits):
                bitname = yname + '_' + str(i)
                ybv[i] = pt[bitname] if bitname in pt else False

            zbv = [False] * zbits
            for i in range(zbits):
                bitname = zname + '_' + str(i)
                zbv[i] = pt[bitname] if bitname in pt else False

            x_idx = bv2int(xbv) if not dynamicperiodic(xgrid) else graytobin(bv2int(xbv))
            y_idx = bv2int(ybv) if not dynamicperiodic(ygrid) else graytobin(bv2int(ybv))
            z_idx = bv2int(zbv) if not dynamicperiodic(zgrid) else graytobin(bv2int(zbv))

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
    v.rotate(180,0,0,1)

    g = gl.GLGridItem()
    g.scale(xbins//10, ybins//10, zbins//10)
    w.addItem(g)
    # g.rotate(90, 1, 0, 1)

    ax = gl.GLAxisItem()
    w.addItem(ax)

    QtGui.QApplication.instance().exec_()

    mgr.configure(reordering=config['reordering'])  # Reinstate config state