"""
.. module::module

Module container


"""

from typing import Dict, Generator, List, Union 
import itertools

import sydra.spaces as sp
from sydra.utils import flatten
from sydra.spaces import OutOfDomainError

class AbstractModule(object):
    r"""
    Wrapper for translating between concrete and discrete I/O values.

    Attributes
    ----------
    mgr: dd manager
        Manager for manipulating predicates as bdds
    inputs: dict
        Dictionary {str: symbolicintervals}
    outputs: dict
        Dictionary {str: symbolicintervals}
    pred: bdd
        Predicate encoding the module I/O relation

    """

    def __init__(self, mgr, inputs, outputs, pred=None, nonblocking=None):
        """
        Initialize the abstract module.

        The pred and nonblocking parameters should only be used for fast 
        initialization of the module.

        Parameters
        ----------
        mgr: bdd manager

        inputs: dict(str: sydra.spaces.SymbolicSet)
            Input variable name, SymbolicSet type
        outputs: dict(str: sydra.spaces.SymbolicSet)
            Output variable name, SymbolicSet type
        pred: bdd
            Predicate to initialize the input-output map
        nonblocking: bdd
            Predicate to initialize nonblocking inputs

        """
        if not set(inputs).isdisjoint(outputs):
            both_io = set(inputs).intersection(outputs)
            raise ValueError("Variables {0} cannot be both "
                             "an input and output".format(str(both_io)))

        self._in = inputs
        self._out = outputs
        self._mgr = mgr

        if any(var.isalnum() is False for var in self.vars):
            raise ValueError("Only alphanumeric strings are accepted as variable names")

        self._pred = self.mgr.false if pred is None else pred
        self._nb   = self.nonblock() if nonblocking is None else nonblocking

    def __repr__(self):
        if len(self.vars) > 0:
            maxvarlen = max(len(v) for v in self.vars)
            maxvarlen = max(maxvarlen, 20) + 4
        s = "{0:{1}}{2}\n".format("==Input Names==", maxvarlen, "==Input Spaces==")
        s += "\n".join(["{0:{1}}".format(k,maxvarlen) + v.__repr__() for k,v in self._in.items()])
        s += "\n"
        s += "{0:{1}}{2}\n".format("==Output Names==", maxvarlen, "==Output Spaces==")
        s += "\n".join(["{0:{1}}".format(k,maxvarlen) + v.__repr__() for k,v in self._out.items()])
        s += "\n"
        return s

    def __getitem__(self, var):
        r"""Access an input or output variable from its name."""
        if var not in self.vars:
            raise ValueError("Variable does not exist")
        return self._in[var] if var in self._in else self._out[var]

    def __eq__(self, other) -> bool:
        if not isinstance(other, AbstractModule):
            return False
        if self.mgr != other.mgr:
            return False
        if self._in != other.inputs or self._out != other.outputs:
            return False
        if self.pred != other.pred:
            return False
        if self._nb != other._nb:
            return False 
        return True

    def iospace(self):
        r"""Get input-output space predicate."""
        return self.inspace() & self.outspace()

    @property
    def mgr(self):
        return self._mgr

    @property
    def pred(self):
        return self._pred

    @property
    def vars(self):
        return set(self._in).union(self._out)

    @property
    def inputs(self):
        return self._in

    @property
    def outputs(self):
        return self._out

    @property
    def pred_bitvars(self) -> Dict[str, List[str]]:
        r"""Get dictionary with variable name keys and BDD bit names as values."""
        s = self.pred.support
        allocbits = {v: [] for v in self.vars}
        for bitvar in s:
            prefix, _ = bitvar.split("_")
            allocbits[prefix].append(bitvar)
            allocbits[prefix].sort()
        return allocbits

    def inspace(self):
        r"""
        Get input space predicate.

        Returns
        -------
        bdd: 
            Predicate corresponding to the Cartesian product of each
            input space.
        
        """
        space = self.mgr.true
        for var in self.inputs:
            space &= self.inputs[var].abs_space(self.mgr, var)
        return space

    def outspace(self):
        r"""
        Get output space predicate.

        Returns
        -------
        bdd: 
            Predicate corresponding to the Cartesian product of each
            output space.

        """
        space = self.mgr.true
        for var in self.outputs:
            space &= self.outputs[var].abs_space(self.mgr, var)
        return space

    def nonblock(self):
        r"""
        Compute a predicate of the inputs for which there exists an output.
        
        Equivalent to the predicate from hiding all module outputs. 

        Returns
        -------
        bdd:
            Predicate for :math:`\exists x'. (system /\ outspace(x'))`

        """
        elim_bits = []
        for k in self._out:
            elim_bits += self.pred_bitvars[k]
        return self.mgr.exist(elim_bits, self.pred & self.outspace())

    def refine_io(self, concrete: dict, **kwargs) -> bool:
        r"""
        Abstracts a concrete I/O transition and refines the module.

        Side effects:
            Mutates the pred attribute

        Proceeds in two steps:
            1. Adds fully nondeterministic outputs to any new inputs
                :math:`pred = pred \vee (\neg nonblocking \wedge abstract inputs)`
            2. Constraints input/output pairs corresponding to the concrete transitions
                :math:`pred = pred \wedge (abstract inputs \implies abstractoutputs)`

        Parameters
        ----------
        concrete: dict
            Concrete values for conc2pred() method of spaces.
        **kwargs:
            Custom arguments for conc2pred() method of spaces 

        Returns
        -------
        bool:
            False if the transition was not applied due to an out of domain
            error. True otherwise.

        See Also
        --------
        io_refined:
            Returns a new module instead of mutating it.

        """
        assert set(concrete) == set(self.inputs) | set(self.outputs)

        try:
            inputs = {k: v for k, v in concrete.items() if k in self.inputs}
            outputs = {k: v for k, v in concrete.items() if k in self.outputs}
            inpred = self.concrete_input_to_abs(inputs, **kwargs)
            outpred = self.concrete_output_to_abs(outputs, **kwargs)
        except:  # TODO: Should catch a custom out of boundaries error
            return False

        self._pred = ((~self._nb & inpred & self.inspace()) | self.pred) & (~inpred | outpred)
        self._nb = self._nb | inpred

        return True

    def io_refined(self, concrete: dict, silent: bool=True, **kwargs) -> 'AbstractModule':
        r"""
        Get a module refined with input-output data.

        Parameters
        ----------
        concrete: dict(Variable str: concrete values)
            Each abstract variable is
        silent: bool, optional
            If true, does not raise an error out of bounds errors
            If false, raises an error.
        **kwargs:
            Custom arguments for conc2pred methods in spaces

        Returns
        -------
        AbstractModule:
            New refined module

        See Also
        --------
        refine_io:
            Mutates the module instead of returning a new one.

        """
        assert set(concrete) >= set(self.inputs).union(set(self.outputs))

        try:
            inputs = {k: v for k, v in concrete.items() if k in self.inputs}
            outputs = {k: v for k, v in concrete.items() if k in self.outputs}
            inpred = self.concrete_input_to_abs(inputs, **kwargs)
            outpred = self.concrete_output_to_abs(outputs, **kwargs)
        except OutOfDomainError:  # TODO: Should catch a custom out of boundaries error
            if silent:
                return self
            raise
        except:
            raise

        return AbstractModule(self.mgr, 
                              self.inputs,
                              self.outputs,
                              ((~self._nb & inpred & self.inspace()) | self.pred) & (~inpred | outpred),
                              self._nb | inpred)

    def check(self):
        r"""Check consistency of internal private attributes.

        Raises
        ------
        AssertionError
            Internal inconsistencies detected

        """
        assert self._nb == self.nonblock()

        def varname(bit):
            return bit.split("_")[0]

        # Variable support check
        assert {varname(bit) for bit in self.pred.support} <= self.vars
        # Check that there aren't any transitions with invalid inputs/outputs
        assert ~self.iospace() & self.pred == self.mgr.false

    def concrete_input_to_abs(self, concrete: dict, **kwargs):
        r"""Convert concrete inputs to abstract ones.

        Applies an underapproximation for inputs that live in a continuous 
        domain.

        Parameters
        ----------
        concrete : dict
            Keys are input variable names, values are concrete instances of 
            that variable
        **kwargs:
            Arguments that are specific to each input space's conc2pred method

        Returns
        -------
        bdd:
            BDD corresponding to the concrete input box

        """
        in_bdd = self.inspace()
        for var in concrete.keys():
            custom_args = {k: v[var] for k, v in kwargs.items() if var in v}

            if isinstance(self[var], sp.ContinuousCover):
                custom_args.update({'innerapprox': True})

            in_bdd &= self.inputs[var].conc2pred(self.mgr,
                                                 var,
                                                 concrete[var],
                                                 **custom_args)

        return in_bdd

    def concrete_output_to_abs(self, concrete: dict, **kwargs):
        r"""Convert concrete outputs to abstract ones.

        Applies an overapproximation for outputs that live in a continuous 
        domain.

        Parameters
        ----------
        concrete : dict
            Keys are output variable names, values are concrete instances of
            that variable
        **kwargs:
            Arguments that are specific to each output space's conc2pred method

        Returns
        -------
        bdd:
            BDD corresponding to the concrete output box

        """
        out_bdd = self.outspace()
        for var in concrete.keys():
            custom_args = {k: v[var] for k, v in kwargs.items() if var in v}

            if isinstance(self[var], sp.ContinuousCover):
                custom_args.update({'innerapprox': False})

            out_bdd &= self.outputs[var].conc2pred(self.mgr,
                                                   var,
                                                   concrete[var],
                                                   **custom_args)

        return out_bdd

    def input_iter(self, precision: dict) -> Generator:
        r"""
        Generate for exhaustive search over the concrete input grid.

        #FIXME: Implementation assumes dictionary ordering is stable

        Parameters
        ----------
        precision: dict
            Keys are variables associated with dynamic covers. Values are an 
            integer number of bits.

        Yields
        -------
        dict:
            Keys are input variable names, values are concrete outputs

        """
        numin = len(self.inputs)
        names = [k for k, v in self.inputs.items() if isinstance(v, sp.DynamicCover)]
        names += [k for k, v in self.inputs.items() if not isinstance(v, sp.DynamicCover)]

        # FIXME: This solution is very adhoc!!! Need to find a better way to 
        # accommodate keyword arguments
        iters = [v.conc_iter(precision[k]) for k, v in self.inputs.items()
                 if isinstance(v, sp.DynamicCover)]
        iters += [v.conc_iter() for k, v in self.inputs.items()
                  if not isinstance(v, sp.DynamicCover)]

        for i in itertools.product(*iters):
            yield {names[j]: i[j] for j in range(numin)}

    def count_io(self, bits: int) -> float:
        r"""
        Count number of input-output pairs in abstraction.

        Parameters
        ----------
        bits: int
            Number of bits when counting

        Side Effects
        ------------
        None

        Returns
        -------
        float:
            Number of transitions

        """
        return self.mgr.count(self.pred, bits)

    def count_io_space(self, bits: int) -> float:
        return self.mgr.count(self.iospace(), bits)

    def hidden(self, elim_vars) -> 'AbstractModule':
        r"""
        Hides an output variable and returns another module.

        Parameters
        ----------
        elim_vars: iterator
            Iterable of output variable names

        Returns
        -------
        AbstractModule:
            Another abstract module with the removed outputs

        """
        if any(var not in self._out for var in elim_vars):
            raise ValueError("Can only hide output variables")

        elim_bits = []
        for k in elim_vars:
            elim_bits += self.pred_bitvars[k]

        newoutputs = {k: v for k, v in self.outputs.items() if k not in elim_vars}

        elim_bits = set(elim_bits) & self.pred.support
        return AbstractModule(self.mgr,
                              self.inputs.copy(),
                              newoutputs,
                              self.mgr.exist(elim_bits, self.pred & self.iospace())
                              )

    def __le__(self, other: 'AbstractModule') -> bool:
        r"""
        Check for a feedback refinement relation between two modules.
        
        If abs <= conc then we call abs an abstraction of the concrete system
        conc. 

        Returns
        -------
        bool:
            True if the feedback refinement relation holds.
            False if there is a type or module port mismatch

        """
        #TODO: Checks between Dynamic and fixed partitions

        # Incomparable
        if not isinstance(other, AbstractModule):
            return False
        if self.inputs != other.inputs:
            return False
        if self.outputs != other.outputs:
            return False

        # Abstract module must accept fewer inputs 
        if (~self._nb | other._nb != self.mgr.true):
            return False

        # Abstract system outputs must be overapproximations
        if (~(self._nb & other.pred) | self.pred) != self.mgr.true:
            return False

        return True

    def coarsened(self, bits=None, **kwargs) -> 'AbstractModule':
        r"""Remove less significant bits and coarsen the system representation.

        Input bits are universally abstracted ("forall")
        Output bits are existentially abstracted ("exists")

        Parameters
        ----------
        bits: dict(str: int)
            Maps variable name to the maximum number of bits to keep. All
            excluded variables aren't coarsened.

        """
        bits = dict() if bits is None else bits
        bits.update(kwargs)
        if any(not isinstance(self[var], sp.DynamicCover) for var in bits):
            raise ValueError("Can only coarsen dynamic covers.")

        if any(b < 0 for b in bits.values()):
            raise ValueError("Negative bits are not allowed.")

        # Identify bits that are finer than the desired precision 
        def fine_bits(var, num):
            return self.pred_bitvars[var][num:]

        outbits = [fine_bits(k, v) for k, v in bits.items() if k in self.outputs]
        outbits = flatten(outbits)
        inbits = [fine_bits(k, v) for k, v in bits.items() if k in self.inputs]
        inbits = flatten(inbits)

        # Shrink nonblocking set
        nb = self.mgr.forall(inbits, self.nonblock())
        # Expand outputs with respect to input coarseness
        newpred = self.mgr.exist(inbits, self.pred)
        # Constrain outputs to align with nonblocking set
        newpred = self.mgr.exist(outbits, nb & newpred & self.outspace())

        return AbstractModule(self.mgr, self.inputs, self.outputs,
                              pred=newpred, nonblocking=nb)

    def renamed(self, names: Dict = None, **kwargs) -> 'AbstractModule':
        """
        Rename input and output ports

        Parameters
        ----------
        names: dict, default = dict()
            Keys are str of old names, values are str of new names
        **kwargs:
            Same dictionary format as names.
        """

        names = dict([]) if names is None else names
        names.update(kwargs)

        newoutputs = self._out.copy()
        newinputs = self._in.copy()
        swapbits = dict()

        for oldname, newname in names.items():
            if oldname not in self.vars:
                raise ValueError("Cannot rename non-existent I/O " + oldname)
            if newname in self.vars:
                raise ValueError("Don't currently support renaming to an existing variable")

            if oldname in self.outputs:
                newoutputs[newname] = newoutputs.pop(oldname)
            elif oldname in self.inputs:
                newinputs[newname] = newinputs.pop(oldname)

            newbits = [newname + '_' + i.split('_')[1] for i in self.pred_bitvars[oldname]]
            self.mgr.declare(*newbits)
            swapbits.update({i: j for i, j in zip(self.pred_bitvars[oldname], newbits)})

        newpred = self.pred if swapbits == {} else self.mgr.let(swapbits, self.pred)

        return AbstractModule(self.mgr, newinputs, newoutputs, newpred)

    def composed_with(self, other: 'AbstractModule') -> 'AbstractModule':
        """
        Compose two modules.

        Automatically detects if the composition is parallel or series composition.

        mod1.composed_with(mod2) reduces to one of the following:\n
        1) mod1 | mod2\n
        2) mod1 >> mod2\n
        3) mod2 >> mod1\n

        Parameters
        ----------
        other: AbstractModule
            Module to compose with.
        
        Returns
        -------
        AbstractModule:
            Composed monolithic module

        """

        if self.mgr != other.mgr:
            raise ValueError("Module managers do not match")
        if not set(self._out).isdisjoint(other.outputs):
            raise ValueError("Outputs are not disjoint")
        
        inout = set(other._out).intersection(self._in)
        outin = set(other._in).intersection(self._out)
        if len(inout) > 0 and len(outin) > 0:
            raise ValueError("Feedback composition is disallowed")

        # Identify upstream >> downstream modules, or if parallel comp
        if len(inout) > 0:
            upstream = other
            downstream = self
        if len(outin) > 0:
            upstream = self
            downstream = other
        if len(outin) == 0 and len(inout) == 0:
            upstream = self
            downstream = other

        # Outputs are the union of both module outputs
        newoutputs = upstream._out.copy()
        newoutputs.update(downstream.outputs)

        # Compute inputs = (self.inputs union other.inputs) and check for differences
        newinputs = upstream._in.copy()
        for k in downstream.inputs:
            # Common existing inputs must have same grid type
            if k in newinputs and newinputs[k] != downstream.inputs[k]:
                raise TypeError("Mismatch between input spaces {0}, {1}".format(newinputs[k],
                                                                                downstream.inputs[k]))
            newinputs[k] = downstream.inputs[k]
        
        # Shared vars that are both inputs and outputs
        overlapping_vars = set(upstream._out) & set(downstream._in)
        for k in overlapping_vars:
            newinputs.pop(k)

        # Compute forall outputvars . (self.pred => other.nonblock())
        nonblocking = ~upstream.outspace() | ~upstream.pred | downstream.nonblock()
        elim_bits = set(flatten([upstream.pred_bitvars[k] for k in upstream._out]))
        elim_bits |= set(flatten([downstream.pred_bitvars[k] for k in downstream._out]))
        elim_bits &= nonblocking.support
        nonblocking = upstream.mgr.forall(elim_bits, nonblocking)
        return AbstractModule(upstream.mgr, newinputs, newoutputs,
                                upstream.pred & downstream.pred & nonblocking)

    def __rshift__(self, other: Union['AbstractModule', tuple]) -> 'AbstractModule':
        r"""
        Series composition or output renaming operator.

        Series composition reduces to parallel composition if no output variables
        feed into the other module's input.

        See Also
        --------
        __rrshift__:
            Input renaming operator
        composed_with:
            Generic composition operator 

        Parameters
        ----------
        other: Union['AbstractModule', tuple]
            If AbstractModule then computes the composition self >> other.
            If tuple(oldname: str, newname: str) then replaces output name.

        Returns
        -------
        AbstractModule:
            Either the series composition or 
        
        """
        if isinstance(other, tuple):
            # Renaming an output
            oldname, newname = other
            if oldname not in self._out:
                raise ValueError("Cannot rename non-existent output")

            return self.renamed({oldname: newname})

        elif isinstance(other, AbstractModule):
            if self.mgr != other.mgr:
                raise ValueError("Module managers do not match")
            if not set(self._out).isdisjoint(other.outputs):
                raise ValueError("Outputs are not disjoint")
            if not set(other._out).isdisjoint(self._in):
                raise ValueError("Downstream outputs feedback composed with upstream inputs")

            # Outputs are the union of both module outputs
            newoutputs = self._out.copy()
            newoutputs.update(other.outputs)

            # Compute inputs = (self.inputs | other.inputs) \ ()
            # Checks for type differences
            newinputs = self._in.copy()
            for k in other.inputs:
                # Common existing inputs must have same grid type
                if k in newinputs and newinputs[k] != other.inputs[k]:
                    raise TypeError("Mismatch between input spaces {0}, {1}".format(newinputs[k], 
                                                                                    other.inputs[k]))
                newinputs[k] = other.inputs[k]
            
            # Variables that are both inputs and outputs
            overlapping_vars = set(self._out) & set(other._in)
            for k in overlapping_vars:
                newinputs.pop(k)

            # Compute forall outputvars . (self.pred => other.nonblock())
            nonblocking = ~self.outspace() | ~self.pred | other.nonblock()
            elim_bits = set(flatten([self.pred_bitvars[k] for k in self._out]))
            elim_bits |= set(flatten([other.pred_bitvars[k] for k in other._out]))
            elim_bits &= nonblocking.support
            nonblocking = self.mgr.forall(elim_bits, nonblocking)
            return AbstractModule(self.mgr, newinputs, newoutputs,
                                  self.pred & other.pred & nonblocking)

        else:
            raise TypeError

    def __rrshift__(self, other) -> 'AbstractModule':
        r"""
        Input renaming operator via serial composition notation

        Example: module = ("a", "b") >> module
        """
        if isinstance(other, tuple):
            # Renaming an input
            newname, oldname = other
            if oldname not in self._in:
                raise ValueError("Cannot rename non-existent input")

            return self.renamed({oldname: newname})

        elif isinstance(other, AbstractModule):
            raise RuntimeError("__rshift__ should be called instead of __rrshift__")
        else:
            raise TypeError

    # Parallel composition
    def __or__(self, other: 'AbstractModule') -> 'AbstractModule':
        r"""
        Parallel composition of modules.

        Returns
        -------
        AbstractModule:
            Parallel composition of two modules

        """
        if self.mgr != other.mgr:
            raise ValueError("Module managers do not match")

        # Check for disjointness
        if not set(self._out).isdisjoint(other.outputs):
            raise ValueError("Outputs are not disjoint")
        if not set(self._out).isdisjoint(other.inputs):
            raise ValueError("Module output feeds into other module input")
        if not set(self._in).isdisjoint(other.outputs):
            raise ValueError("Module output feeds into other module input")

        # Take union of inputs and check for type differences
        newinputs = self._in.copy()
        for k in other.inputs:
            # Common existing inputs must have same grid type
            if k in newinputs and newinputs[k] != other.inputs[k]:
                raise TypeError("Mismatch between input spaces {0} and {1}".format(newinputs[k], other.inputs[k]))
            newinputs[k] = other.inputs[k]

        # Take union of disjoint output sets.
        newoutputs = self._out.copy()
        newoutputs.update(other.outputs)

        return AbstractModule(self.mgr, newinputs, newoutputs,
                              self.pred & other.pred)


class CompositeModule(object): 
    raise NotImplementedError