import itertools

import vpax.symbolicinterval as si

class AbstractModule(object):
    """
    Function wrapper for translating between concrete and discrete I/O values
    """

    def __init__(self, mgr, inputs, outputs, pred = None):
        """

        """

        self._in  = inputs 
        self._out = outputs

        if not set(self._in).isdisjoint(self._out):
            raise ValueError("A variable cannot be both an input and output")
        
        if any(var.isalnum() is False for var in self.vars):
            raise ValueError("Only alphanumeric strings are accepted as variable names")

        self.mgr = mgr
        self.pred = self.mgr.true if pred is None else pred # FIXME: Change this to inspace => outspace 

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

    @property
    def nonblock(self):
        # TODO: Change so that it outputs a module and uses the output hiding operator instead. 
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
            if isinstance(self[var], si.DynamicInterval):
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
            if isinstance(self[var], si.DynamicInterval):
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

    def input_iter(self, precision):
        """
        Exhaustively searches over the input grid

        Args:
            precision 

        Implementation assumes dictionary ordering is stable
        """
        numin = len(self.inputs)
        names = tuple(self.inputs.keys())
        iters = [v.conc_iter(precision[k]) for k,v in self.inputs.values()]
        for i in itertools.product(*iters):
            yield {names[j]: i[j] for j in range(numin)}
        
    def input_space_pred(self):
        raise NotImplementedError
    
    def output_space_pred(self):
        raise NotImplementedError 

    def count_io(self, bits):
        return self.mgr.count(self.pred, bits)

    def hide(self, elim_vars):
        """
        Hides an output variable and returns another module 
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
                              self.mgr.exist(elim_bits, self.pred))
    
    def __le__(self,other):
        """
        Checks for a feedback refinement relation between two modules
        """
        raise NotImplementedError

    
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
            nonblocking = self.mgr.forall(elim_bits, nonblocking)
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

