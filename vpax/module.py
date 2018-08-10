import itertools

import vpax.spaces as sp
from vpax.utils import flatten


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

    Methods
    -------

    """

    def __init__(self, mgr, inputs, outputs, pred=None, nonblocking=None):
        """
        Initialize the abstract module.

        The pred and nonblocking parameters should only be used for fast 
        initialization of the module.

        Parameters
        ----------
        mgr: bdd manager

        inputs: dict(str: vpax.spaces.SymbolicSet)
            Input variable name, SymbolicSet type
        outputs: dict(str: vpax.spaces.SymbolicSet)
            Output variable name, SymbolicSet type
        pred: bdd
            Predicate to initialize the input-output map
        nonblocking: bdd
            Predicate to initialize nonblocking inputs
        """

        self._in = inputs
        self._out = outputs
        self.mgr = mgr
        self._iospace = self.inspace() & self.outspace()

        if not set(self._in).isdisjoint(self._out):
            raise ValueError("A variable cannot be both an input and output")

        if any(var.isalnum() is False for var in self.vars):
            raise ValueError("Only alphanumeric strings are accepted as variable names")

        self._pred = self.mgr.false if pred is None else pred
        self._nb   = self.nonblock() if nonblocking is None else nonblocking

    def __repr__(self):
        maxvarlen = max(len(v) for v in self.vars)
        maxvarlen = max(maxvarlen, 20) + 4
        s = "{0:{1}}{2}\n".format("==Input Names==", maxvarlen, "==Input Spaces==")
        s += "\n".join(["{0:{1}}".format(k,maxvarlen) + v.__repr__() for k,v in self._in.items()]) + "\n"
        s += "{0:{1}}{2}\n".format("==Output Names==", maxvarlen, "==Output Spaces==")
        s += "\n".join(["{0:{1}}".format(k,maxvarlen) + v.__repr__() for k,v in self._out.items()]) + "\n"
        return s

    def __getitem__(self, var):
        """
        Access an input or output variable from its name.
        """
        if var not in self.vars:
            raise ValueError("Variable does not exist")
        return self._in[var] if var in self._in else self._out[var]

    def __eq__(self,other):
        if not isinstance(other, AbstractModule):
            return False
        if self.mgr != other.mgr:
            return False
        if self._in != other.inputs or self._out != other.outputs:
            return False
        if self.pred != other.pred:
            return False
        return True

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
    def pred_bitvars(self):
        r"""
        Dictionary with variable name keys and BDD bit names as values
        """
        s = self.pred.support
        allocbits = {v: [] for v in self.vars}
        for bitvar in s:
            prefix, _ = bitvar.split("_")
            allocbits[prefix].append(bitvar)
            allocbits[prefix].sort()
        return allocbits

    def inspace(self):
        r"""
        Input space predicate

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
        Output space predicate

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
        Compute a predicate of the inputs for which there exists an associated output. 
        Equivalent to the predicate from hiding all module outputs. 

        Returns
        -------
        bdd:
            Predicate for exists x'. (system /\ outspace(x'))

        """
        elim_bits = []
        for k in self._out:
            elim_bits += self.pred_bitvars[k]
        return self.mgr.exist(elim_bits, self.pred & self.outspace())

    def constrained_inputs(self):
        """
        Inputs with fully nondeterministic outputs

        Returns
        -------
        bdd: 
            Predicate for forall x'. (outspace(x') => system)
        """
        elim_bits = []
        for k in self._out:
            elim_bits += self.pred_bitvars[k]
        return self.mgr.forall(elim_bits, ~self.outspace() | self.pred)


    def _newinputs(self, inputpred):
        r"""

        Parameters
        ----------
        inputpred: bdd 
            Inputs

        Returns
        -------
        bdd:
            (~self.nonblock() & inputpred & self.inspace())

        Assumes that inputpred is actually a predicate over the input space alone.

        An input predicate can include inputs that are not currently covered by the current
        abstraction's nonblocking inputs. This method identifies the new region.
        """
        # TODO: Check that predicate is actually an input predicate!
        return ~self.nonblock() & inputpred & self.inspace()

    def apply_abstract_transitions(self, concrete, **kwargs):
        r"""
        Abstracts a concrete I/O transition and applies the abstract transitions to
        the current system abstraction.

        Proceeds in two steps:
            1. Adds fully nondeterministic outputs to any new inputs
                pred = pred | (~nonblocking & abstract_inputs)
            2. Constraints input/output pairs corresponding to the concrete transitions
                pred = pred & (abstract_inputs => abstract_outputs)

        Parameters
        ----------
        concrete: dict
        **kwargs:

        Side Effects
        ------------
        Mutates the pred attribute

        Returns
        -------
        bool:
            False if the transition was not applied due to an out of domain
            error.

        """

        assert set(concrete) == set(self.inputs) | set(self.outputs)

        try:
            inputs = {k: v for k, v in concrete.items() if k in self.inputs}
            outputs = {k: v for k, v in concrete.items() if k in self.outputs}
            inpred = self.concrete_input_to_abs(inputs, **kwargs)
            outpred = self.concrete_output_to_abs(outputs, **kwargs)
        except:  # TODO: Should catch a custom out of boundaries error
            return False

        self._pred = ((~self._nb & inpred) | self.pred) & (~inpred | outpred)
        self._nb = self._nb | inpred

        return True

    def check(self):
        r"""
        Check consistency of internal private attributes.

        Raises
        ------
        AssertionError
            Internal inconsistencies detected
        """
        assert self._nb == self.nonblock()

        def varname(bit):
            return bit.split("_")[0]

        # Variable support check
        assert {varname(bit) for bit in self._pred.support} <= self.vars
        # Check that there aren't any transitions with invalid inputs/outputs
        assert self._iospace() & self._pred == self.mgr.false

    def concrete_input_to_abs(self, concrete, **kwargs):
        r"""
        Parameters
        ----------
        concrete : dict
            Keys are input variable names, values are concrete instances of that variable
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

            in_bdd &= self.inputs[var].conc2pred(self.mgr, var, concrete[var], **custom_args)

        return in_bdd

    def concrete_output_to_abs(self, concrete, **kwargs):
        r"""

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

            out_bdd &= self.outputs[var].conc2pred(self.mgr, var, concrete[var], **custom_args)

        return out_bdd

    def ioimplies2pred(self, concrete, **kwargs):
        r"""
        Return the implication (input box => output box).

        Splits the hypberbox into input, output variables
        If the input/output boxes don't align with the grid then:
            - The input box is contracted
            - The output box is expanded

        If concrete is underspecified, then it generates a hyperinterval 
        embedded in a lower dimension

        Parameters
        ----------
        concrete: dict
            Keys are variable names, values are concrete instances of that variable
        **kwargs: 
            Arguments that are specific to each input/output space's conc2pred method

        Returns
        -------
        bdd: 
            Implication (input pred => output pred)

        """
        inputs = {k: v for k, v in concrete.items() if k in self.inputs}
        outputs = {k: v for k, v in concrete.items() if k in self.outputs}
        inpred = self.concrete_input_to_abs(inputs, **kwargs)
        outpred = self.concrete_output_to_abs(outputs, **kwargs)
        return (~inpred | outpred)

    def input_iter(self, precision: dict):
        r"""
        Generator for exhaustive search over the concrete input grid.

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

        #TODO: This solution is very adhoc!!! Need to find a better way to accommodate keyword arguments
        iters = [v.conc_iter(precision[k]) for k,v in self.inputs.items() if isinstance(v, sp.DynamicCover)]
        iters += [v.conc_iter() for k,v in self.inputs.items() if not isinstance(v, sp.DynamicCover)]

        for i in itertools.product(*iters):
            yield {names[j]: i[j] for j in range(numin)}

    def count_io(self, bits: int):
        r"""
        Number of input-output pairs in abstraction

        Parameters
        ----------
        bits: int
            Number of bits when counting

        Returns
        -------
        float:
            Number of transitions

        """
        return self.mgr.count(self.pred, bits)

    def count_io_space(self, bits: int):
        return self.mgr.count(self._iospace, bits)

    def hide(self, elim_vars):
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
                              self.mgr.exist(elim_bits, self.pred & self._iospace))

    def __le__(self, other):
        r"""
        Check for a feedback refinement relation between two modules.
        """
        raise NotImplementedError

    def coarsen(self, bits: dict):
        r"""
        Remove less significant bits and coarsen the system representation.

        Input bits are universally abstracted ("forall")
        Output bits are existentially abstracted ("exists")

        Parameters
        ---------
        bits: dict(str: int)
            Maps variable name to the number of bits to remove. All variables
            that are excluded aren't coarsened.
        """
        raise NotImplementedError

        # FIXME: Can only coarsen for continuous partitions

        def last_bits(var, num):
            return self.pred_bitvars[var][-num:]

        exist_bits = [last_bits(k, v) for k, v in bits.items() if k in self.outputs]
        exist_bits = flatten(exist_bits)
        forall_bits = [last_bits(k, v) for k, v in bits.items() if k in self.inputs]
        forall_bits = flatten(forall_bits)

        nb = self.mgr.forall(forall_bits, self.nonblock())

        return AbstractModule(self.mgr, self.inputs, self.outputs,
                             pred=self.mgr.exist(exist_bits, nb & self._pred),
                             nonblocking=nb)


    def __rshift__(self, other):
        r"""
        Serial composition self >> other by feeding self's output into other's
        input. Also an output renaming operator

        TODO: Break apart to renaming and series composition use cases
        """
        if isinstance(other, tuple):
            # Renaming an output
            oldname, newname = other
            if oldname not in self._out:
                raise ValueError("Cannot rename non-existent output")
            if newname in self.vars:
                raise ValueError("Don't currently support renaming to an existing variable")

            newoutputs = self._out.copy()
            newoutputs[newname] = newoutputs.pop(oldname)

            newbits = [newname + '_' + i.split('_')[1] for i in self.pred_bitvars[oldname]]
            self.mgr.declare(*newbits)
            newvars = {i: j for i, j in zip(self.pred_bitvars[oldname], newbits)}

            newpred = self.pred if newvars == {} else self.mgr.let(newvars, self.pred)

            return AbstractModule(self.mgr, self._in, newoutputs, newpred)

        elif isinstance(other, AbstractModule):
            if self.mgr != other.mgr:
                raise ValueError("Module managers do not match")
            if not set(self._out).isdisjoint(other.outputs):
                raise ValueError("Outputs are not disjoint")
            if not set(other._out).isdisjoint(self._in):
                raise ValueError("Downstream outputs feedback composed with upstream inputs")

            newoutputs = self._out.copy()
            newoutputs.update(other.outputs)

            # Compute inputs = (self.inputs | other.inputs) \ ()
            # Checks for type differences
            newinputs = self._in.copy()
            for k in other.inputs:
                # Common existing inputs must have same grid type
                if k in newinputs and newinputs[k] != other.inputs[k]:
                    raise TypeError("Mismatch between input spaces {0}, {1}".format(newinputs[k], other.inputs[k]))
                newinputs[k] = other.inputs[k]
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

    def __rrshift__(self, other):
        r"""
        Input renaming operator via serial composition notation

        Example: module = ("a", "b") >> module
        """
        if isinstance(other, tuple):
            # Renaming an input
            newname, oldname = other
            if oldname not in self._in:
                raise ValueError("Cannot rename non-existent input")
            if newname in self.vars:
                raise ValueError("Don't currently support renaming to an existing variable")

            newinputs = self._in.copy()
            newinputs[newname] = newinputs.pop(oldname)

            newbits = [newname + '_' + i.split('_')[1] for i in self.pred_bitvars[oldname]]
            self.mgr.declare(*newbits)
            newvars = {i: j for i, j in zip(self.pred_bitvars[oldname], newbits)}

            newpred = self.pred if newvars == {} else self.mgr.let(newvars, self.pred)

            return AbstractModule(self.mgr, newinputs, self._out, newpred)

        elif isinstance(other, AbstractModule):
            raise RuntimeError("__rshift__ should be called instead of __rrshift__")
        else:
            raise TypeError

    # Parallel composition
    def __or__(self, other):
        r"""
        Parallel composition of modules
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
