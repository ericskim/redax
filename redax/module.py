"""
.. module::module

Module container


"""

from __future__ import annotations

from typing import Dict, Generator, List, Union, Collection, Tuple, Set, Sequence 
import itertools

from toposort import toposort

import redax.spaces as sp
from .spaces import OutOfDomainError


class Interface(object):
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

        inputs: dict(str: redax.spaces.SymbolicSet)
            Input variable name, SymbolicSet type
        outputs: dict(str: redax.spaces.SymbolicSet)
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
            raise ValueError("Non-alphanumeric variable name")

        self._pred = self.mgr.false if pred is None else pred
        self._nb = self.nonblock() if nonblocking is None else nonblocking

    def __repr__(self):
        if len(self.vars) > 0:
            maxvarlen = max(len(v) for v in self.vars)
            maxvarlen = max(maxvarlen, 20) + 4
        s = "Interface(inputs={"
        s += ", ".join([k + ": " + v.__repr__() for k, v in self._in.items()])
        s += "}, outputs={"
        s += ", ".join([k + ": " + v.__repr__() for k, v in self._out.items()])
        s += "})"
        return s

    def __getitem__(self, var):
        r"""Access an input or output variable from its name."""
        if var not in self.vars:
            raise ValueError("Variable {} does not exist".format(var))
        return self._in[var] if var in self._in else self._out[var]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Interface):
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
        allocbits: Dict[str, List[str]] = {v: [] for v in self.vars}
        for bitvar in s:
            prefix, _ = bitvar.split("_")
            allocbits[prefix].append(bitvar)
            allocbits[prefix].sort()
        return allocbits

    def is_sink(self):
        return True if len(self.outputs) == 0 else False

    def is_source(self):
        return True if len(self.inputs) == 0 else False

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
        elim_bits : List[str] = []
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
            Custom arguments for conc2pred() method of spaces.

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
            inpred = self.input_to_abs(inputs, **kwargs)
            outpred = self.output_to_abs(outputs, **kwargs)
        except OutOfDomainError:
            return False
        except:
            raise

        self._pred = ((~self._nb & inpred & self.inspace()) | self.pred) & (~inpred | outpred)
        self._nb = self._nb | inpred

        return True

    def io_refined(self, concrete: dict, silent: bool=True, **kwargs) -> Interface:
        r"""
        Get a module refined with input-output data.

        Parameters
        ----------
        concrete: dict(Variable str: concrete values)

        silent: bool, optional
            If true, does not raise an error for out of bounds errors
            If false, raises an error.
        **kwargs:
            Custom arguments for conc2pred methods in spaces

        Returns
        -------
        Interface:
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
            inpred = self.input_to_abs(inputs, **kwargs)
            outpred = self.output_to_abs(outputs, **kwargs)
        except OutOfDomainError:
            if silent:
                return self
            raise
        except:
            raise

        return Interface(self.mgr,
                              self.inputs,
                              self.outputs,
                              (self.pred | ((~self._nb & inpred) & self.inspace())) & (~inpred | outpred),
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

    def input_to_abs(self, concrete: dict, **kwargs):
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

    def output_to_abs(self, concrete: dict, **kwargs):
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

        #FIXME: Implementation assumes dictionary ordering is stable.
        # This solution is very adhoc!!! Should instead use kwargs for space specific items

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

        iters = [v.conc_iter(precision[k]) for k, v in self.inputs.items()
                    if isinstance(v, sp.DynamicCover)]
        iters += [v.conc_iter() for k, v in self.inputs.items()
                    if not isinstance(v, sp.DynamicCover)]

        for i in itertools.product(*iters):
            yield {names[j]: i[j] for j in range(numin)}

    def count_nb(self, bits: int) -> float:
        return self.mgr.count(self.nonblock(), bits)

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

    def ohidden(self, elim_vars) -> Interface:
        r"""
        Hides an output variable and returns another module.

        Parameters
        ----------
        elim_vars: iterator
            Iterable of output variable names

        Returns
        -------
        Interface:
            Another abstract module with the removed outputs

        """
        from redax.ops import ohide
        return ohide(elim_vars, self)

    def ihidden(self, elim_vars: Sequence[str]) -> Interface:
        r"""
        Hides input variable for sink and returns another sink.
        
        """

        from redax.ops import ihide
        return ihide(elim_vars, self)

    def coarsened(self, bits=None, **kwargs) -> Interface:
        r"""Remove less significant bits and coarsen the system representation.

        Input bits are universally abstracted ("forall")
        Output bits are existentially abstracted ("exists")

        Parameters
        ----------
        bits: dict(str: int)
            Maps variable name to the maximum number of bits to keep. All
            excluded variables aren't coarsened.

        """

        from redax.ops import coarsen
        return coarsen(self, bits, **kwargs)

    def renamed(self, names: Dict = None, **kwargs) -> Interface:
        """
        Rename input and output ports.

        Parameters
        ----------
        names: dict, default = dict()
            Keys are str of old names, values are str of new names
        **kwargs:
            Same dictionary format as names.

        """

        from redax.ops import rename
        return rename(self, names, **kwargs)

    def composed_with(self, other: Interface) -> Interface:
        """
        Compose two modules.

        Automatically detects if the composition is parallel or series composition.

        mod1.composed_with(mod2) reduces to one of the following:\n
        1) mod1 | mod2\n
        2) mod1 >> mod2\n
        3) mod2 >> mod1\n

        Parameters
        ----------
        other: Interface
            Module to compose with.

        Returns
        -------
        Interface:
            Composed monolithic module

        """

        from redax.ops import compose
        return compose(self, other)

    def __rshift__(self, other: Union[Interface, tuple]) -> Interface:
        r"""
        Series composition or output renaming operator.

        Series composition reduces to parallel composition if no output
        variables feed into the other module's input.

        See Also
        --------
        __rrshift__:
            Input renaming operator
        composed_with:
            Generic composition operator

        Parameters
        ----------
        other: Union['Interface', tuple]
            If Interface then computes the composition self >> other.
            If tuple(oldname: str, newname: str) then replaces output name.

        Returns
        -------
        Interface:
            Either the series or parallel composition (if defined)

        """
        if isinstance(other, tuple):
            # Renaming an output
            oldname, newname = other
            if oldname not in self._out:
                raise ValueError("Cannot rename non-existent output")

            return self.renamed({oldname: newname})

        elif isinstance(other, Interface):
            return self.composed_with(other)

        else:
            raise TypeError

    def __rrshift__(self, other) -> Interface:
        r"""
        Input renaming operator via serial composition notation.

        Example: module = ("a", "b") >> module

        """
        if isinstance(other, tuple):
            # Renaming an input
            newname, oldname = other
            if oldname not in self._in:
                raise ValueError("Cannot rename non-existent input")

            return self.renamed({oldname: newname})

        elif isinstance(other, Interface):
            raise RuntimeError("__rshift__ should be called instead of __rrshift__")
        else:
            raise TypeError

    def __mul__(self, other: Interface) -> Interface:

        from redax.ops import parallelcompose
        return parallelcompose(self, other)

    def __le__(self, other: Interface) -> bool:
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

        # Incomparable
        if not isinstance(other, Interface):
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

    def _direct_io_refined(self, inpred, outpred) -> Interface:
        """
        Apply io refinement directly from abstract predicates.

        No conversion from concrete values.
        """
        return Interface(self.mgr,
                              self.inputs,
                              self.outputs,
                              ((~self._nb & inpred & self.inspace()) | self.pred) & (~inpred | outpred),
                              self._nb | inpred)



class CompositeModule(object):
    r"""Container for a collection of modules."""
    def __init__(self, modules: Collection[Interface], checktopo: bool=True) -> None:

        if len(modules) < 1:
            raise ValueError("Empty module collection")

        self.children : Tuple[Interface] = tuple(modules)

        if checktopo:
            self.check()

    @property
    def _inputs(self):
        """Variables that are inputs to some module."""
        inputs = dict()
        for mod in self.children:
            for var in mod.inputs:
                if var in inputs:
                    assert inputs[var] == mod.inputs[var]
                inputs[var] = mod.inputs[var]
        return inputs

    def nonblock(self):
        if len(self.children) == 0:
            raise RuntimeError("Should return false")

        nb = self.children[0].mgr.true
        for mod in self.children:
            nb &= mod._nb
        return nb
    
    @property
    def _outputs(self):
        """Variables that are outputs to some module."""
        outputs = dict()
        for mod in self.children:
            for var in mod.outputs:
                if var in outputs:
                    raise ValueError("Multiple modules own output {0}".format(var))
                if var in self._inputs:
                    assert self._inputs[var] == mod.outputs[var]
                outputs[var] = mod.outputs[var]
        return outputs

    @property
    def _var_io(self):
        vario = dict()
        for idx, mod in enumerate(self.children):
            for var in mod.inputs:
                if var not in vario:
                    vario[var] = {'i': [], 'o': []}
                vario[var]['i'].append(idx)

            for var in mod.outputs:
                if var not in vario:
                    vario[var] = {'i': [], 'o': []}
                vario[var]['o'].append(idx)

                # Cannot have output defined by multiple modules
                if len(vario[var]['o']) > 1:
                    raise ValueError
        return vario

    @property
    def inputs(self):
        r"""Variables that are inputs to the composite module."""
        invars = (var for var, io in self._var_io.items() if len(io['o']) == 0)
        return {var: self._inputs[var] for var in invars}

    @property
    def outputs(self):
        r"""Variables that are outputs to the composite module."""
        outvars = (var for var, io in self._var_io.items() if len(io['o']) > 0)
        return {var: self._outputs[var] for var in outvars}

    @property
    def latent(self):
        r"""
        Variables that are internal to the composite module.

        Note: All latent variables are also output variables from self.outputs
        """
        latentvars = set(self._inputs).intersection(self._outputs)
        return {var: self._outputs[var] for var in latentvars}

    def __getitem__(self, var):
        for mod in self.children:
            if var in mod.vars:
                return mod[var]

        raise ValueError("Variable does not exist")
    
    @property 
    def vars(self):
        vars = set([])
        for mod in self.children:
            vars.update(mod.vars)
        return vars

    def inspace(self):
        raise NotImplementedError

    def outspace(self):
        raise NotImplementedError

    def sorted_mods(self) -> Tuple[Tuple[Interface]]:

        # Pass to declare dependencies
        deps: Dict[int, Set[int]] = {idx: set() for idx, _ in enumerate(self.children)}
        for _, io in self._var_io.items():
            for inmods in io['i']:
                if len(io['o']) > 0:
                    deps[inmods].add(io['o'][0])

        # Toposort requires that elements are hashable.
        # self.children[i] converts from an index back to a module
        return tuple(
                        tuple(self.children[i] for i in modidx)
                        for modidx in toposort(deps)
                    )

    def hidden(self, var):
        raise NotImplementedError

    def check(self, verbose=False):
        # Check consistency of children
        for child in self.children:
            try:
                child.check()
            except:
                print(child)
                raise
            if verbose:
                print("Passed :", child)

        if len(self.children) > 0:
            mgr = self.children[0].mgr
        for child in self.children:
            assert mgr == child.mgr

        # Check validity of module topology
        self._inputs
        self._outputs
        self._var_io
        self.sorted_mods()

    def io_refined(self, concrete, silent: bool=True, **kwargs) -> CompositeModule:
        r"""
        Refine interior modules.

        Minimizes redundant computations of predicates by memoizing.

        Internal modules only updated when all inputs + outputs are provided
        as keys in 'concrete'.

        """
        newmods = []

        # Memoize input/output predicates
        pred_memo = dict()

        # Refine each module individually and append to newmods
        for mod in self.children:


            if not set(mod.vars).issubset(concrete):
                newmods.append(mod)  # don't refine
                continue

            outerror = False
            inerror = False
            inbox = mod.mgr.true
            for var, space in mod.inputs.items():
                # Input kwargs
                slice_kwargs = {k: v[var] for k, v in kwargs.items() if var in v}
                if isinstance(mod[var], sp.ContinuousCover):
                    slice_kwargs.update({'innerapprox': True})

                # Check for reusable predicates or compute it
                _hashable_args = [var] + list(slice_kwargs.items())
                hashable_args = tuple(_hashable_args)
                if hashable_args in pred_memo:
                    in_slice = pred_memo[hashable_args]
                else:
                    try:
                        in_slice = space.conc2pred(mod.mgr,
                                                   var,
                                                   concrete[var],
                                                   **slice_kwargs)
                        pred_memo[hashable_args] = in_slice
                    except OutOfDomainError:
                        if silent:
                            inerror = True
                            break
                        raise
                    except:
                        raise

                inbox &= in_slice

            outbox = mod.mgr.true
            for var, space in mod.outputs.items():
                # Output kwargs
                slice_kwargs = {k: v[var] for k, v in kwargs.items() if var in v}
                if isinstance(mod[var], sp.ContinuousCover):
                    slice_kwargs.update({'innerapprox': False})

                # Check for reusable predicates or compute it
                _hashable_args = [var] + list(slice_kwargs.items())
                hashable_args = tuple(_hashable_args)
                if hashable_args in pred_memo:
                    out_slice = pred_memo[hashable_args]
                else:
                    try:
                        out_slice = space.conc2pred(mod.mgr,
                                                    var,
                                                    concrete[var],
                                                    **slice_kwargs)
                        pred_memo[hashable_args] = out_slice
                    except OutOfDomainError:
                        if silent:
                            outerror = True
                            break
                        raise
                    except:
                        raise
                outbox &= out_slice

            # Refine and remember
            if (outerror or inerror) and silent:
                newmods.append(mod)  # don't refine
            else:
                newmods.append(mod._direct_io_refined(inbox, outbox))

        return CompositeModule(newmods, checktopo=False)

    def atomized(self) -> Interface:
        r"""Reduce composite module into an atomic one."""
        raise NotImplementedError


    @property
    def pred_bitvars(self) -> Dict[str, List[str]]:
        r"""Get dictionary with variable name keys and BDD bit names as values."""


        allocbits = {v: set([]) for v in self.vars}
        for mod in self.children:
            for k, v in mod.pred_bitvars.items():
                allocbits[k].update(v)

        list_allocbits = dict()
        for k, v in allocbits.items():
            list_allocbits[k] = list(v)
            list_allocbits[k].sort()

        return list_allocbits

