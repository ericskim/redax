
from bidict import bidict
from vpax.module import AbstractModule

def to_control_module(mod, states):
    """
    Constructs a control module from a generic one. 
    """
    try: # Dictionary 
        prepost = [(k,v) for k,v in states.items()]
    except: # Tuple or list
        prepost = [(pre,post) for pre,post in states]
    finally:
        pass 

    # Check pre/post state domain equality
    if not {post for pre,post in prepost}.issubset(mod.outputs):
        raise ValueError("Unknown post state")
    if not {pre for pre,post in prepost}.issubset(mod.inputs):
        raise ValueError("Unknown pre state")
    if any(mod.inputs[pre] != mod.outputs[post] for pre,post in prepost):
        raise ValueError("Pre and post state domains do not match")    

    
    # Inputs that are not states are controllable 
    pres = [i for i in zip(*prepost)][0]
    control = {k:v for k,v in mod.inputs.items() if k not in pres}
    
    return ControlModule(mod.mgr, 
                         {i: mod.inputs[i[0]] for i in prepost}, 
                         control,
                         mod.pred)

class ControlModule(AbstractModule):

    def __init__(self, mgr, states, control, pred):
        r"""
        Args:
            mgr:
            states: dict. keys: (pre,post) tuples, values: symbolic type  
            control:
            pred: 
        """

        prestate = {k[0]: v for k,v in states.items()}
        poststate = {k[1]: v for k,v in states.items()}

        inputs = prestate.copy()
        inputs.update(control)

        AbstractModule.__init__(self, mgr, inputs, poststate.copy(), pred)
        self.control = control
        self.prestate = prestate
        self.poststate = poststate
        self.pre_to_post = bidict({k[0]: k[1] for k in states})

    def __repr__(self): 
        s = "Control Grids:\n"
        s += "\n".join([str(k) + " -- " + str(v) for k,v in self.control.items()]) + "\n"
        s += "State Grids:\n"
        s += "\n".join([str((str(k), str(self.pre_to_post[k]))) + " -- " + str(v)for k,v in self.prestate.items()]) + "\n"
        return s

    def __eq__(self, other):
        raise NotImplementedError 

    def controlspace(self):
        space = self.mgr.true
        for var in self.control:
            space &= self.control[var].abs_space(self.mgr, var)
        return space
        
    def statespace(self):
        space = self.mgr.true
        for var in self.prestate:
            space &= self.prestate[var].abs_space(self.mgr, var)
        return space

    def postspace(self):
        space = self.mgr.true
        for var in self.poststate:
            space &= self.poststate[var].abs_space(self.mgr, var)
        return space

    