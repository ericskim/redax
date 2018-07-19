
import inspect 

from bidict import bidict
from collections import OrderedDict, namedtuple

import vpax.symbolicinterval as SymbolicInterval

"""
# TODO: Declare for multiple input/outputs at a time 
# TODO: Support for positional arguments
# TODO: Change input/output decorators to kwargs and give a good docstring for the options 
# TODO: Figure out the best way to do parallel and serial composition...
# TODO: Learning with compositional data
    Support parallel composition first!
"""

# class Signature(namedtuple('Signature', ['Positions', 'Types', 'Names'])):
     

#     def __call__(self):
#         """
#         Accepts both numeric and string access 
#         """
#         pass 


def input(mgr, 
          index,
          bounds, 
          dynamic : bool = True, 
          periodic : bool = False, 
          discrete : bool = False,
          precision: int = 0):
    """
    Function decorator for symbolic module inputs
    """
    assert type(index) in [str, int]
    if not dynamic:
        assert precision > 0, "Fixed intervals must have at least one bin. Set precision >= 1."
    return ModuleAnnotator(mgr, True, index, bounds, discrete, dynamic, periodic, precision)

def output(mgr,
          index,
          bounds, 
          dynamic : bool = True,
          periodic : bool = False, 
          discrete : bool = False,
          precision: int = 0): 
    """
    Function decorator for symbolic module outputs
    """
    assert isinstance(index[0], int) and isinstance(index[1], str) 
    return ModuleAnnotator(mgr, False, index, bounds, discrete, dynamic, periodic, precision)

class ModuleAnnotator(object):
    """
    Class to reason about annotations to a function module 
    """

    def __init__(self, mgr, isinput, index, bounds, discrete, dynamic, periodic, precision):
        # Collect new annotations. Determine type of SymbolicInterval to instantiate 
        self.isinput    = isinput
        self.index      = index
        self.bounds     = bounds
        self.discrete   = discrete
        self.periodic   = periodic 
        self.dynamic    = dynamic
        self.precision  = precision
        self.mgr        = mgr

        if self.discrete:
            # raise NotImplementedError
            self.porttype = SymbolicInterval.DiscreteInterval
        else:
            if self.dynamic:
                self.porttype = SymbolicInterval.DynamicInterval
            else:
                self.porttype = SymbolicInterval.FixedInterval

    def __call__(self, mod):
        # aggregate previous annotations from previous modules
        idx = self.index

        if isinstance(mod, AbstractModule):
            args = mod.args
            inputs = mod.inputs
            outputs = mod.outputs
            assert self.mgr == mod.mgr, "BDD managers don't match"
            name = mod.__name__
            func = mod.concrete_func # FIXME: Should nest modules actually... 

        else: # function type. Base case of recursive module construction
            # assert hasattr(mod, '__call__'), "Must have callable object"
            funcsig = inspect.signature(mod)
            args = funcsig.parameters.keys()
            inputs = OrderedDict({i: None for i in args})
            name = mod.__name__
            if isinstance(funcsig.return_annotation, tuple):
                outputs = [None] * len(funcsig.return_annotation)
            elif type(funcsig.return_annotation) == type:
                outputs = [None]
            else:
                raise ValueError("Type annotation must be either a tuple or type")
            func = mod

        if self.isinput:
            nargs = len(args)
            if isinstance(idx, str):
                assert idx in args, "Index " + str(idx) + " not in arguments " + str(args)
            elif isinstance(idx, int): 
                assert idx < nargs 
                idx = args[idx]
            else:
                raise TypeError("Index {0} needs to be a string or int".format(idx))

            if self.discrete:
                assert isinstance(self.bounds, int)
                inputs[idx] = self.porttype(idx, self.mgr, self.bounds)
            else:
                inputs[idx] = self.porttype(idx, self.mgr, self.bounds[0], self.bounds[1], self.precision, self.periodic)                
        else:
            varname = idx[1]
            if self.discrete:
                assert isinstance(self.bounds, int)
                outputs[idx[0]] = self.porttype(varname, self.mgr, self.bounds)
            else:
                outputs[idx[0]] = self.porttype(varname, self.mgr, self.bounds[0], self.bounds[1], self.precision, self.periodic)


        return AbstractModule(self.mgr, inputs, outputs, func = func, name = name)

class AbstractModule(object):
    """
    Function wrapper for translating between concrete and discrete I/O values
    """

    def __init__(self, mgr, inputs, outputs, 
                 func = None, inhook = None, outhook = None, name= None, pred = None):
        """
        func : Executable python function 
        inhook, outhook: Transforms variable names for concrete function evaluation 
        """

        # TODO: Need to remember argument order, just for sanity?
        # Crosscheck inputs and outputs 
        self.__name__ = name
        self.inputs  = inputs 
        self.outputs = outputs

        # For calling a concrete executable function
        self.concrete_func = func
        if inhook is None:
            self.inhook   = None # FIXME: 
        else:
            self.inhook = inhook
            # TODO: Assert consistency with inputs
        if outhook is None:
            self.outhook  = None # FIXME: 
        else:
            self.outhook = outhook

        self.mgr = mgr
        if pred is None:
            self.pred = self.mgr.true
        else:
            self.pred = pred

    def __repr__(self):
        # s = "Abstract Module: {0}\n".format(self.__name__)
        s = "Number of Arguments: " + str(len(self.args)) + "\n"
        s += "Input Wrappers:\n"
        s += "\n".join([str(k) + " -- " + v.__repr__() for k,v in self.inputs.items()]) + "\n"
        s += "Output Wrappers:\n"
        s += "\n".join(["Out" + str(i) + " -- " + self.outputs[i].__repr__() for i in range(self.numout)]) + "\n"
        return s
    
    @property
    def args(self):
        return self.inputs.keys() 

    # @property
    # def argorder(self):
    #     return bidict({i : list(self.args)[i] for i in range(len(self.args))})

    @property
    def binaryvars(self):
        local = set([])
        for i in self.inputs:
            local.update(self.inputs[i].bits.keys())
        for o in range(self.numout):
            local.update(self.outputs[o].bits.keys())
        return local 

    @property
    def discreteinputs(self):
        return {k for k,v in self.inputs.items() if type(v) == SymbolicInterval.DiscreteInterval}        

    @property
    def numout(self):
        return len(self.outputs)

    @property
    def input_bounds(self):
        return {k: v.bounds for k,v in self.inputs.items()}

    def __getitem__(self, portname):
        return self.inputs[portname]

    def __call__(self, *args, **kwargs):
        # TODO: Variable renaming and stuff...
        if self.concrete_func is None:
            raise RuntimeError("Executable concrete function is not initialized") 
        return self.concrete_func(*args, **kwargs)

    def reset_abstraction(self):
        self.pred = self.mgr.true 

    def input_to_bdd(self,  **kwargs):
        """
        Takes an input and returns the BDD that corresponds with that input
        """

        assert kwargs.keys() == self.args, "Keyword arguments do not match"
        
        inbdd = self.mgr.true
        for k in self.args:
            inbdd &= self.inputs[k].pt2bdd(kwargs[k])
        return inbdd

    def input_box_to_bdd(self, inputboxes, granularity = None, verbose = False):
        """
        Input boxes must have an innerapproximation 
        """
        assert (inputboxes.keys() == self.args) # TODO: Shouldn't have this. Constraints reflect the dependencies

        input_bdd = self.mgr.true # FIXME: Errors if using fixed intervals with non-power of two 
        for k in self.args:
            if isinstance(self.inputs[k], SymbolicInterval.DiscreteInterval):
                input_bdd &= self.inputs[k].pt2bdd(inputboxes[k])
            else:
                input_bdd &= self.inputs[k].box2bdd(inputboxes[k],
                                                    innerapprox = True)
        return input_bdd

    def output_box_to_bdd(self, outputboxes, granulaity = None, verbose = False):
        """
        Output boxes should have an overapproximation 
        """
        assert len(outputboxes) == len(self.outputs) # TODO: Shouldn't need this. Outputs are uniquely defined. 

        output_bdd = self.mgr.true # FIXME: Errors if using fixed intervals with non-power of two 
        for k in range(self.numout):
            if isinstance(self.outputs[k], SymbolicInterval.DiscreteInterval):
                output_bdd &= self.outputs[k].pt2bdd(outputboxes[k])
            else:
                output_bdd &= self.outputs[k].box2bdd(outputboxes[k], 
                                                      innerapprox = False)
        return output_bdd

    def io_boxes_to_bdd(self, inputboxes, outputboxes, granularity = None, verbose = False):
        """
        Applies the implication (input box => output box) 
        If the input/output boxes don't align with the grid then:
            - The input box is contracted 
            - The output box is expanded
        """
        return (~self.input_box_to_bdd(inputboxes) | self.output_box_to_bdd(outputboxes))

    def apply_io_constraint(self, transition):
        # TODO: Add functionality for a desired granularity, perhaps higher than the existing one? 
        inbox, outbox = transition
        self.pred &= self.io_boxes_to_bdd(inbox, outbox)

    def check(self):
        if None in self.inputs.values():
            return False
        if None in self.outputs:
            return False
        return True

    def count_io(self):
        return self.mgr.count(self.pred, len(self.localvars))

    def hide(self, var):
        """
        Hides a variable and returns another module 
        """
        raise NotImplementedError

    def __rshift__(self, other):
        """
        Serial composition by feeding self output into other input
        """
        raise NotImplementedError
        if isinstance(other, tuple):
            # Renaming an output
            raise NotImplementedError
        elif isinstance(other, AbstractModule):
            raise NotImplementedError
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
            assert oldname in self.inputs, "Cannot rename non-existent input"
            newinputs = OrderedDict()
            for k,v in self.inputs.items():
                if k == oldname:
                    newinputs[newname] = v.renamed(newname)
                    newvars = {i:j for i,j in zip(v.bitorder, newinputs[newname].bitorder)}
                else: 
                    newinputs[k]=v

            # TODO: 1) Change input hooks
            return AbstractModule(self.mgr, newinputs, self.outputs, 
                                  self.concrete_func, self.inhook, self.outhook,
                                  self.__name__, 
                                  self.mgr.let(newvars, self.pred))

        elif isinstance(other, AbstractModule):
            raise RuntimeError("__rshift__ should be called instead of __rrshift__")
        else:
            raise TypeError


    # Parallel composition
    def __or__(self, other):
        """
        Returns the parallel composition of modules 
        """

        raise NotImplementedError

        if self.mgr != other.mgr:
            raise ValueError("Module managers do not match")
        
        # Take union of inputs. More granular input is precedent


        # Take union of disjoint output sets. 


        return AbstractModule(self.mgr, newinputs, self.outputs, 
                                self.concrete_func, self.inhook, self.outhook,
                                self.__name__, 
                                self.pred & other.pred)

    #bidict(enumerate(system.inputs.keys()))? 

class ModuleCompostion():
    """
    Composite modules 

    Common variable names? 
    """
    def __init__(self):
        raise NotImplementedError

