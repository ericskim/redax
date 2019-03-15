
from typing import Dict, Collection, Set

import funcy as fn
from .module import Interface
import redax.utils.bv as bv


def shared_refine(ifaces: Collection, safecheck=True):
    r"""
    Compute shared refinement of a collection of interfaces.


    Returns
    -------
    Interface:
        Shared refinement interface
    """

    # Managers must be identical
    mgr = fn.first(iface.mgr for iface in ifaces)
    if any(mgr != iface.mgr for iface in ifaces):
        raise RuntimeError("Managers do not match")

    # Same inputs and outputs
    input_signature = fn.first(iface.inputs for iface in ifaces)
    if any(input_signature != iface.inputs for iface in ifaces):
        raise RuntimeError("Interface inputs do not match")

    output_signature = fn.first(iface.outputs for iface in ifaces)
    if any(output_signature != iface.outputs for iface in ifaces):
        raise RuntimeError("Interface outputs do not match")

    nb = mgr.false
    pred = mgr.true

    for iface in ifaces:
        nb |= iface.assum
        pred &= iface.guar

    if safecheck:
        raise NotImplementedError     # TODO: Shared refinability check here

    return Interface(mgr,
                     input_signature,
                     output_signature,
                     guar=pred,
                     assum=nb
                     )


def coarsen(mod: Interface, bits=None, **kwargs) -> Interface:

    import redax.spaces as sp

    bits = dict() if bits is None else bits
    bits.update(kwargs)
    if any(not isinstance(mod[var], sp.DynamicCover) for var in bits):
        raise ValueError("Can only coarsen dynamic covers.")

    if any(b < 0 for b in bits.values()):
        raise ValueError("Negative bits are not allowed.")

    def finer_bits(var, num):
        # Identify bits that are finer than the desired precision
        return mod.pred_bitvars[var][num:]

    outbits = [finer_bits(k, v) for k, v in bits.items() if k in mod.outputs]
    outbits = bv.flatten(outbits)
    inbits = [finer_bits(k, v) for k, v in bits.items() if k in mod.inputs]
    inbits = bv.flatten(inbits)

    # Shrink nonblocking set
    nb = mod.mgr.forall(inbits, mod.assum)
    # Expand outputs with respect to input coarseness
    newpred = mod.mgr.exist(inbits, mod.guar)
    # Constrain outputs to align with nonblocking set
    newpred = mod.mgr.exist(outbits, newpred & mod.outspace())

    return Interface(mod.mgr,
                     mod.inputs,
                     mod.outputs,
                     guar=newpred,
                     assum=nb
                     )


def rename(mod: Interface, names: Dict = None, **kwargs) -> Interface:
    """
    Rename input and output ports.

    Parameters
    ----------
    names: dict, default = dict()
        Keys are str of old names, values are str of new names
    **kwargs:
        Same dictionary format as names.

    """
    names = dict([]) if names is None else names
    names.update(kwargs)

    newoutputs = mod._out.copy()
    newinputs = mod._in.copy()
    swapbits = dict()

    for oldname, newname in names.items():
        if oldname not in mod.vars:
            # raise ValueError("Cannot rename non-existent I/O " + oldname)
            continue
        if newname in mod.vars:
            raise ValueError("Don't currently support renaming to an existing variable")

        if oldname in mod.outputs:
            newoutputs[newname] = newoutputs.pop(oldname)
        elif oldname in mod.inputs:
            newinputs[newname] = newinputs.pop(oldname)

        newbits = [newname + '_' + i.split('_')[1] for i in mod.pred_bitvars[oldname]]
        mod.mgr.declare(*newbits)
        swapbits.update({i: j for i, j in zip(mod.pred_bitvars[oldname], newbits)})

    newguar = mod.guar if swapbits == {} else mod.mgr.let(swapbits, mod.guar)
    newassum = mod.assum if swapbits == {} else mod.mgr.let(swapbits, mod.assum)

    return Interface(mod.mgr,
                     newinputs,
                     newoutputs,
                     newguar,
                     newassum
                     )


def ohide(elim_vars: Collection, mod: Interface) -> Interface:
    r"""
    Hides an output variable and returns another interface.

    Parameters
    ----------
    elim_vars: Container
        Iterable container of output variable names

    Returns
    -------
    Interface:
        Another abstract interface with the removed outputs

    """

    if any(var not in mod._out for var in elim_vars):
        raise ValueError("Can only hide output variables")

    elim_bits: Set[str] = []
    for k in elim_vars:
        elim_bits += mod.pred_bitvars[k]
    elim_bits = set(elim_bits) & mod.guar.support

    newoutputs = {k: v for k, v in mod.outputs.items() if k not in elim_vars}

    return Interface(mod.mgr,
                     mod.inputs.copy(),
                     newoutputs,
                     guar=mod.mgr.exist(elim_bits, mod._guar & mod.outspace()),
                     assum=mod._assum
                     )


def ihide(elim_vars: Collection, mod: Interface) -> Interface:
    r"""

    """
    if any(var not in mod.inputs for var in elim_vars):
        raise ValueError("Can only hide input variables")

    if len(mod.outputs) > 0:
        raise ValueError("Can only hide inputs on sink modules")

    elim_bits = []
    for k in elim_vars:
        elim_bits += mod.pred_bitvars[k]

    newinputs = {k: v for k, v in mod.inputs.items() if k not in elim_vars}

    elim_bits = set(elim_bits) & mod.pred.support

    return Interface(mod.mgr,
                     newinputs,
                     mod.outputs.copy(),
                     assum=mod.mgr.exist(elim_bits, mod.assum & mod.iospace())
                     )


def compose(mod: Interface, other: Interface) -> Interface:
    r"""

    """

    if mod.mgr != other.mgr:
        raise ValueError("Module managers do not match")
    if not set(mod._out).isdisjoint(other.outputs):
        raise ValueError("Outputs are not disjoint")

    inout = set(other._out).intersection(mod._in)
    outin = set(other._in).intersection(mod._out)
    if len(inout) > 0 and len(outin) > 0:
        raise ValueError("Feedback composition is disallowed")

    # Identify upstream >> downstream modules, or if parallel comp
    if len(inout) > 0:
        upstream = other
        downstream = mod
    if len(outin) > 0:
        upstream = mod
        downstream = other
    if len(outin) == 0 and len(inout) == 0:
        upstream = mod
        downstream = other

    # Outputs are the union of both module outputs
    newoutputs = upstream._out.copy()
    newoutputs.update(downstream.outputs)

    # Compute inputs = (mod.inputs union other.inputs) and check for differences
    newinputs = upstream._in.copy()
    for k in downstream.inputs:
        # Common existing inputs must have same grid type
        if k in newinputs and newinputs[k] != downstream.inputs[k]:
            raise TypeError("Mismatch between input spaces {0}, {1}".format(newinputs[k],
                                                                            downstream.inputs[k]))
        newinputs[k] = downstream.inputs[k]

    # Shared vars that are both inputs and outputs
    # overlapping_vars = set(upstream._out) & set(downstream._in)
    for k in (set(upstream._out) & set(downstream._in)):
        newinputs.pop(k)

    # Compute nonblocking == forall outputvars . (mod.pred => other.nonblock())
    nonblocking = ~upstream.outspace() | ~upstream._guar | downstream._assum
    elim_bits = set(bv.flatten([upstream.pred_bitvars[k] for k in upstream._out]))
    elim_bits |= set(bv.flatten([downstream.pred_bitvars[k] for k in downstream._out]))
    elim_bits &= nonblocking.support
    nonblocking = upstream.mgr.forall(elim_bits, nonblocking)

    return Interface(upstream.mgr,
                     newinputs,
                     newoutputs,
                     guar=upstream._guar & downstream._guar,
                     assum=upstream._assum & nonblocking
                     )


def sinkprepend(iface: Interface, sink: Interface) -> Interface:
    """
    Composition of an interface connected in series with a sink.

    Identical to ohide(sharedvars, comp(iface, sinkmod)) where sharedvars is
    the intersection of iface's outputs and sinkmod's inputs. This is a faster
    implementation.

    Returns
    -------
    Interface:
        Sink interface obtained by composing and hiding the shared variables.

    """

    if not sink.is_sink():
        raise ValueError("Sink module is not actually a sink.")

    if not set(iface.outputs) <= set(sink.inputs):
        raise ValueError("Sink inputs must be a superset of iface outputs.")

    newinputs = iface._in.copy()
    for k in sink.inputs:
        # Common existing inputs must have same grid type
        if k in newinputs and newinputs[k] != sink.inputs[k]:
            raise TypeError("Mismatch between input spaces {0}, {1}".format(newinputs[k],
                                                                            sink.inputs[k]))
        newinputs[k] = sink.inputs[k]

    # Remove shared vars that are both inputs and outputs
    for k in (set(iface._out) & set(sink._in)):
        newinputs.pop(k)

    newnonblock = ~iface.guar | sink.assum
    elim_bits = set(bv.flatten([iface.pred_bitvars[k] for k in iface.outputs]))
    elim_bits |= set(bv.flatten([sink.pred_bitvars[k] for k in iface.outputs]))
    elim_bits &= newnonblock.support

    return Interface(iface.mgr,
                     newinputs,
                     dict(),
                     assum=iface.assum & iface.mgr.forall(elim_bits, newnonblock)
                     )


def parallelcompose(mod: Interface, other: Interface) -> Interface:
    r"""
    Parallel composition of interfaces.

    Returns
    -------
    Interface:
        Parallel composition of two interfaces

    """
    if mod.mgr != other.mgr:
        raise ValueError("Module managers do not match")

    # Check for disjointness
    if not set(mod._out).isdisjoint(other.outputs):
        raise ValueError("Outputs are not disjoint")
    if not set(mod._out).isdisjoint(other.inputs):
        raise ValueError("Module output feeds into other module input")
    if not set(mod._in).isdisjoint(other.outputs):
        raise ValueError("Module output feeds into other module input")

    # Take union of inputs and check for type differences
    newinputs = mod._in.copy()
    for k in other.inputs:
        # Common existing inputs must have same grid type
        if k in newinputs and newinputs[k] != other.inputs[k]:
            raise TypeError("Mismatch between input spaces {0} and {1}".format(newinputs[k], other.inputs[k]))
        newinputs[k] = other.inputs[k]

    # Take union of disjoint output sets.
    newoutputs = mod._out.copy()
    newoutputs.update(other.outputs)

    return Interface(mod.mgr,
                     newinputs,
                     newoutputs,
                     mod.guar & other.guar,
                     mod.assum & other.assum
                     )
