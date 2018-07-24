
import inspect 

from bidict import bidict
from collections import OrderedDict, namedtuple

import vpax.symbolicinterval as SymbolicInterval

class AbstractModule(object):
    """
    Function wrapper for translating between concrete and discrete I/O values
    """

    def __init__(self, mgr, inputs, outputs, pred = None):
        """
        func : Executable python function 
        inhook, outhook: Transforms variable names for concrete function evaluation 
        """

        self._in  = inputs 
        self._out = outputs

        if not set(self._in).isdisjoint(self._out):
            raise ValueError("A variable cannot be both an input and output")
        
        if any(var.isalnum() is False for var in self.vars):
            raise ValueError("Only alphanumeric strings are accepted as variable names")

        self.mgr = mgr
        self.pred = self.mgr.true if pred is None else pred

    def __repr__(self):
        s = "Input Grids:\n"
        s += "\n".join([str(k) + " -- " + v.__repr__() for k,v in self._in.items()]) + "\n"
        s += "Output Grids:\n"
        s += "\n".join([str(k) + " -- " + v.__repr__() for k,v in self._out.items()]) + "\n"
        return s

    @property
    def vars(self):
        return set(self._in).union(self._out)

    @property
    def input_bounds(self):
        return {k: v.bounds for k,v in self._in.items()}

    @property
    def output_bounds(self):
        return {k: v.bounds for k,v in self._out.items()}

    @property
    def inputs(self):
        return self._in
    
    @property
    def outputs(self):
        return self._out

    @property
    def pred_bitvars(self):
        s = self.pred.support
        allocbits = {v: [] for v in self.vars}
        for bitvar in s:
            prefix, index = bitvar.split("_")
            allocbits[prefix].append(bitvar)
        return allocbits

    @property
    def nonblock(self):
        elim_bits = []
        for k in self._out:
            elim_bits += self.pred_bitvars[k]
        return self.mgr.exist(elim_bits, self.pred)

    def ioimplies2pred(self, hyperbox, **kwargs):
        """
        Returns the implication (input box => output box) 

        Splits the hypberbox into input, output variables
        If the input/output boxes don't align with the grid then:
            - The input box is contracted 
            - The output box is expanded

        If the hyperbox is underspecified, then it generates a hyperinterval embedded in a lower dimension 
        """
        
        in_bdd = self.mgr.true # FIXME: Errors if using fixed intervals with non-power of two 
        out_bdd = self.mgr.true 
        for var in hyperbox.keys():
            if isinstance(self[var], SymbolicInterval.DynamicInterval):
                nbits = kwargs['precision'][var]
                assert isinstance(nbits, int) 
                if var in self._in:
                    in_bdd  &= self[var].box2pred(self.mgr, var, hyperbox[var],
                                                 nbits, innerapprox = True)
                else:
                    out_bdd &= self[var].box2pred(self.mgr, var, hyperbox[var],
                                                 nbits, innerapprox = False)
            else:
                raise NotImplementedError

        return (~in_bdd | out_bdd)

    def iopair2pred(self, hyperbox, **kwargs):
        """
        Returns the pair (input box AND output box) 

        Splits the hypberbox into input, output variables
        If the input/output boxes don't align with the grid then:
            - The input box is contracted 
            - The output box is expanded

        If the hyperbox is underspecified, then it generates a hyperinterval embedded in a lower dimension
        """
        
        io_bdd = self.mgr.true # FIXME: Errors if using fixed intervals with non-power of two 
        for var in hyperbox.keys():
            if isinstance(self[var], SymbolicInterval.DynamicInterval):
                nbits = kwargs['precision'][var]
                assert type(nbits) == int 
                if var in self._in:
                    io_bdd  &= self[var].box2pred(self.mgr, var, hyperbox[var],
                                                 nbits, innerapprox = True)
                else:
                    io_bdd &= self[var].box2pred(self.mgr, var, hyperbox[var],
                                                 nbits, innerapprox = False)
            else:
                raise NotImplementedError

        return io_bdd

    def count_io(self, bits):
        return self.mgr.count(self.pred, bits)

    def __getitem__(self, var):
        """
        Access an input or output variable from its name 
        """
        if var in self._in:
            return self._in[var]
        elif var in self._out:
            return self._out[var]
        else:
            raise ValueError("Variable does not exist")

    def __eq__(self,other):
        if not isinstance(other, AbstractModule): 
            return False 
        if self.mgr != other.mgr:
            return False
        if self._in != other.inputs: 
            return False
        if self._out != other.outputs:
            return False
        if self.pred != other.pred:
            return False 
        return True 

    def hide(self, vars):
        """
        Hides an output variable and returns another module 
        """

        if any(var not in self._out for var in vars):
            raise ValueError("Can only hide output variables")

        elim_bits = []
        for k in self._out:
            elim_bits += self.pred_bitvars[k]
        
        raise NotImplementedError
        return self.mgr.exist(elim_bits, self.pred)


    def __rshift__(self, other):
        """
        Serial composition self >> other by feeding self's output into other's input
        """
        if isinstance(other, tuple):
            # Renaming an output  
            oldname, newname  = other
            if oldname not in self._out:
                raise ValueError("Cannot rename non-existent output")
            if newname in self.vars:
                raise ValueError("Don't currently support renaming to an existing variable")

            newoutputs = self._out.copy() 
            newoutputs[newname] = newoutputs.pop(oldname)

            newbits = [newname + '_' + i.split('_')[1] for i in self.pred_bitvars[oldname]]
            self.mgr.declare(*newbits)
            newvars = {i:j for i,j in zip(self.pred_bitvars[oldname], newbits)}
            
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
                    raise TypeError("Mismatch between input grids {0} and {1}".format(newinputs[k], other.inputs[k]))
                newinputs[k] = other.inputs[k]
            overlapping_vars = set(self._out) & set(other._in)
            for k in overlapping_vars:
                newinputs.pop(k)

            # Flattens list of lists to a single list
            flatten = lambda l: [item for sublist in l for item in sublist]

            # Compute forall outputvars . (self.pred => other.nonblock)
            nonblocking = ~self.pred | other.nonblock 
            elim_bits   = set(flatten([self.pred_bitvars[k] for k in self._out]))
            elim_bits  |= set(flatten([other.pred_bitvars[k] for k in other._out]))
            elim_bits  &= nonblocking.support
            nonblocking = self.mgr.forall(list(elim_bits), nonblocking)
            return AbstractModule(self.mgr, newinputs, newoutputs, 
                                  self.pred & other.pred & nonblocking) 

        else:
            raise TypeError 

    # Serial composition being fed by other 
    def __rrshift__(self, other):
        """
        Serial composition by feeding other output into self input 
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
            newvars = {i:j for i,j in zip(self.pred_bitvars[oldname], newbits)}

            newpred = self.pred if newvars == {} else self.mgr.let(newvars, self.pred) 

            return AbstractModule(self.mgr, newinputs, self._out, newpred)

        elif isinstance(other, AbstractModule):
            raise RuntimeError("__rshift__ should be called instead of __rrshift__")
        else:
            raise TypeError

    # Parallel composition
    def __or__(self, other):
        """
        Returns the parallel composition of modules 
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
                raise TypeError("Mismatch between input grids {0} and {1}".format(newinputs[k], other.inputs[k]))
            newinputs[k] = other.inputs[k]

        # Take union of disjoint output sets. 
        newoutputs = self._out.copy()
        newoutputs.update(other.outputs)

        return AbstractModule(self.mgr, newinputs, newoutputs,
                                self.pred & other.pred)

# def input(mgr, 
#           index,
#           bounds, 
#           dynamic : bool = True, 
#           periodic : bool = False, 
#           discrete : bool = False,
#           precision: int = 0):
#     """
#     Function decorator for symbolic module inputs
#     """
#     assert type(index) in [str, int]
#     if not dynamic:
#         assert precision > 0, "Fixed intervals must have at least one bin. Set precision >= 1."
#     return ModuleAnnotator(mgr, True, index, bounds, discrete, dynamic, periodic, precision)

# def output(mgr,
#           index,
#           bounds, 
#           dynamic : bool = True,
#           periodic : bool = False, 
#           discrete : bool = False,
#           precision: int = 0): 
#     """
#     Function decorator for symbolic module outputs
#     """
#     assert isinstance(index[0], int) and isinstance(index[1], str) 
#     return ModuleAnnotator(mgr, False, index, bounds, discrete, dynamic, periodic, precision)

# class ModuleAnnotator(object):
#     """
#     Class to reason about annotations to a function module 
#     """

#     def __init__(self, mgr, isinput, index, bounds, discrete, dynamic, periodic, precision):
#         # Collect new annotations. Determine type of SymbolicInterval to instantiate 
#         self.isinput    = isinput
#         self.index      = index
#         self.bounds     = bounds
#         self.discrete   = discrete
#         self.periodic   = periodic 
#         self.dynamic    = dynamic
#         self.precision  = precision
#         self.mgr        = mgr

#         if self.discrete:
#             # raise NotImplementedError
#             self.porttype = SymbolicInterval.DiscreteInterval
#         else:
#             if self.dynamic:
#                 self.porttype = SymbolicInterval.DynamicInterval
#             else:
#                 self.porttype = SymbolicInterval.FixedInterval

#     def __call__(self, mod):
#         # aggregate previous annotations from previous modules
#         idx = self.index

#         if isinstance(mod, AbstractModule):
#             args = mod.args
#             inputs = mod.inputs
#             outputs = mod.outputs
#             assert self.mgr == mod.mgr, "BDD managers don't match"
#             name = mod.__name__
#             func = mod.concrete_func

#         else: # function type. Base case of recursive module construction
#             # assert hasattr(mod, '__call__'), "Must have callable object"
#             funcsig = inspect.signature(mod)
#             args = funcsig.parameters.keys()
#             inputs = OrderedDict({i: None for i in args})
#             name = mod.__name__
#             if isinstance(funcsig.return_annotation, tuple):
#                 outputs = [None] * len(funcsig.return_annotation)
#             elif type(funcsig.return_annotation) == type:
#                 outputs = [None]
#             else:
#                 raise ValueError("Type annotation must be either a tuple or type")
#             func = mod

#         if self.isinput:
#             nargs = len(args)
#             if isinstance(idx, str):
#                 assert idx in args, "Index " + str(idx) + " not in arguments " + str(args)
#             elif isinstance(idx, int): 
#                 assert idx < nargs 
#                 idx = args[idx]
#             else:
#                 raise TypeError("Index {0} needs to be a string or int".format(idx))

#             if self.discrete:
#                 assert isinstance(self.bounds, int)
#                 inputs[idx] = self.porttype(idx, self.mgr, self.bounds)
#             else:
#                 inputs[idx] = self.porttype(idx, self.mgr, self.bounds[0], self.bounds[1], self.precision, self.periodic)                
#         else:
#             varname = idx[1]
#             if self.discrete:
#                 assert isinstance(self.bounds, int)
#                 outputs[idx[0]] = self.porttype(varname, self.mgr, self.bounds)
#             else:
#                 outputs[idx[0]] = self.porttype(varname, self.mgr, self.bounds[0], self.bounds[1], self.precision, self.periodic)


#         return AbstractModule(self.mgr, inputs, outputs, func = func, name = name)



