import itertools

import vpax.spaces as sp

class AbstractModule(object):
    """
    Wrapper for translating between concrete and discrete I/O values
       
    Attributes:
        mgr: dd manager
        pred: Predicate encoding the module I/O relation 
        inputs: Dictionary {str: symbolicintervals}
        outputs:
    
    """

    def __init__(self, mgr, inputs, outputs, pred = None):
        """

        """

        self._in  = inputs 
        self._out = outputs
        self.mgr = mgr
        self._iospace = self.inspace() & self.outspace()

        if not set(self._in).isdisjoint(self._out):
            raise ValueError("A variable cannot be both an input and output")
        
        if any(var.isalnum() is False for var in self.vars):
            raise ValueError("Only alphanumeric strings are accepted as variable names")


        self.pred = self._iospace if pred is None else pred

    def __repr__(self):
        s = "Input Grids:\n"
        s += "\n".join([str(k) + " -- " + v.__repr__() for k,v in self._in.items()]) + "\n"
        s += "Output Grids:\n"
        s += "\n".join([str(k) + " -- " + v.__repr__() for k,v in self._out.items()]) + "\n"
        return s

    def __getitem__(self, var):
        """
        Access an input or output variable from its name 
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
        s = self.pred.support
        allocbits = {v: [] for v in self.vars}
        for bitvar in s:
            prefix, index = bitvar.split("_")
            allocbits[prefix].append(bitvar)
        return allocbits

    def inspace(self):
        """
        Input space predicate

        Returns:
            bdd: Predicate corresponding to the Cartesian product of each input space. 
        """
        space = self.mgr.true
        for var in self.inputs:
            space &= self.inputs[var].abs_space(self.mgr, var)
        return space

    def outspace(self):
        """
        Output space predicate 

        Returns:
            bdd: Predicate corresponding to the Cartesian product of each output space. 
        """
        space = self.mgr.true
        for var in self.outputs:
            space &= self.outputs[var].abs_space(self.mgr, var)
        return space

    def nonblock(self):
        """
        Returns a predicate of the inputs for which there exists an associated output. 

        Returns:
            bdd: Predicate for exists x'. (system /\ outspace(x'))
        
        Equivalent to hiding all module outputs and obtaining the predicate 
        """
        elim_bits = []
        for k in self._out:
            elim_bits += self.pred_bitvars[k]
        return self.mgr.exist(elim_bits, self.pred & self.outspace()) 

    def constrained_inputs(self):
        """
        Inputs with fully nondeterministic outputs

        Returns:
            bdd: Predicate for forall x'. (outspace(x') => system)
        """
        elim_bits = []
        for k in self._out:
            elim_bits += self.pred_bitvars[k]
        return self.mgr.forall(elim_bits, ~self.outspace() | self.pred)


    def _newinputs(self, inputpred):
        """
        
        Args:
            inputpred (bdd): Inputs

        Returns:
            bdd: (~self.nonblock() & inputpred & self.inspace())

        Assumes that inputpred is actually a predicate over the input space alone.

        An input predicate can include inputs that are not currently covered by the current
        abstraction's nonblocking inputs. This method identifies the new region. 
        """
        # TODO: Check that predicate is actually an input predicate! 
        return ~self.nonblock() & inputpred & self.inspace()

    def ioimplies2pred(self, concrete, **kwargs):
        """
        Returns the implication (input box => output box)

        Args:
            - concrete (dict): Keys are variable names, values are concrete instances of that variable
            - kwargs: Arguments that are specific to each input/output space's conc2pred method

        Returns:
            bdd: Implication (input pred => output pred)

        Splits the hypberbox into input, output variables
        If the input/output boxes don't align with the grid then:
            - The input box is contracted 
            - The output box is expanded

        TODO: 
        Change this to get
            pred = (pred | (~nb & ibox)) & (ibox => obox)

        If concrete is underspecified, then it generates a hyperinterval embedded in a lower dimension
        """
        
        in_bdd  = self.inspace() 
        out_bdd = self.outspace()
        for var in concrete.keys():
            if isinstance(self[var], sp.DynamicPartition):
                nbits = kwargs['precision'][var]
                assert nbits >= 0 
                if var in self._in:
                    in_bdd  &= self[var].conc2pred(self.mgr, var, concrete[var],
                                                 nbits, innerapprox = True)
                else:
                    out_bdd &= self[var].conc2pred(self.mgr, var, concrete[var],
                                                 nbits, innerapprox = False)
            elif isinstance(self[var], sp.FixedPartition):
                if var in self._in:
                    in_bdd  &= self[var].conc2pred(self.mgr, var, concrete[var],
                                                    innerapprox = True)
                else:
                    out_bdd &= self[var].conc2pred(self.mgr, var, concrete[var],
                                                    innerapprox = False)
            elif isinstance(self[var], sp.EmbeddedGrid):
                if var in self._in:
                    in_bdd  &= self[var].conc2pred(self.mgr, var, concrete[var])
                else:
                    out_bdd &= self[var].conc2pred(self.mgr, var, concrete[var])
            else:
                raise TypeError  


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
        
        io_bdd = self._iospace
        for var in hyperbox.keys():
            if isinstance(self[var], sp.DynamicPartition):
                nbits = kwargs['precision'][var]
                assert type(nbits) == int 
                if var in self._in:
                    io_bdd  &= self[var].box2pred(self.mgr, var, hyperbox[var],
                                                 nbits, innerapprox = True)
                else:
                    io_bdd &= self[var].box2pred(self.mgr, var, hyperbox[var],
                                                 nbits, innerapprox = False)
            else:
                raise TypeError

        return io_bdd

    def input_iter(self, precision):
        """
        Exhaustively searches over the input grid

        Args:
            precision: A dictionary 

        Returns:

        Implementation assumes dictionary ordering is stable
        """
        numin = len(self.inputs)
        names = [k for k,v in self.inputs.items() if isinstance(v, sp.DynamicPartition)]
        names += [k for k,v in self.inputs.items() if not isinstance(v, sp.DynamicPartition)]
        iters = [v.conc_iter(precision[k]) for k,v in self.inputs.items() if isinstance(v, sp.DynamicPartition)]  #FIXME: not everything has a precision!!!
        iters += [v.conc_iter() for k,v in self.inputs.items() if not isinstance(v, sp.DynamicPartition)]
        for i in itertools.product(*iters):
            yield {names[j]: i[j] for j in range(numin)}

    def count_io(self, bits):
        return self.mgr.count(self.pred, bits)

    def hide(self, elim_vars):
        """
        Hides an output variable and returns another module

        Args:
            elim_vars: Iterable of output variable names
        
        """

        if any(var not in self._out for var in elim_vars):
            raise ValueError("Can only hide output variables")

        elim_bits = []
        for k in elim_vars:
            elim_bits += self.pred_bitvars[k]

        newoutputs = {k:v for k,v in self.outputs.items() if k not in elim_vars}

        elim_bits = set(elim_bits) & self.pred.support
        return AbstractModule(self.mgr, 
                              self.inputs.copy(),
                              newoutputs,
                              self.mgr.exist(elim_bits, self.pred & self._iospace))
    
    def __le__(self,other):
        """
        Checks for a feedback refinement relation between two modules
        """
        raise NotImplementedError
    
    def __rshift__(self, other):
        """
        Serial composition self >> other by feeding self's output into other's input
        Also an output renaming operator
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

            # Compute forall outputvars . (self.pred => other.nonblock())
            nonblocking = ~self.outspace() | ~self.pred | other.nonblock()
            elim_bits   = set(flatten([self.pred_bitvars[k] for k in self._out]))
            elim_bits  |= set(flatten([other.pred_bitvars[k] for k in other._out]))
            elim_bits  &= nonblocking.support
            nonblocking = self.mgr.forall(elim_bits, nonblocking)
            return AbstractModule(self.mgr, newinputs, newoutputs, 
                                  self.pred & other.pred & nonblocking) 

        else:
            raise TypeError 

    def __rrshift__(self, other):
        """
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

