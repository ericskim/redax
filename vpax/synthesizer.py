from functools import reduce

flatten = lambda l: [item for sublist in l for item in sublist]

def _name(i):
    return i.split('_')[0]

def _idx(i):
    return i.split('_')[1] 

class ControlPre():
    def __init__(self, controlsys):
        self.sys = controlsys
        self.nonblock = controlsys.nonblock

        self.elimcontrol = lambda bits, pred : self.sys.mgr.exist(bits, pred)
        self.elimpost = lambda bits, pred: self.sys.mgr.forall(bits, pred)

    def __call__(self, Z, no_inputs = False):
        """
        Controllable predecessor for target next state set Z(x')

        nonblock /\ forall x'. (sys(x,u,x') => Z(x'))
        """
        prebits = flatten([self.sys.pred_bitvars[state] for state in self.sys.prestate])
        postbits = [self.sys.pre_to_post[_name(i)] + '_' + _idx(i) for i in prebits]
        swapvars = {i:j for i,j in zip(prebits, postbits)}
        # Exchange Z's pre state variables for post state variables 
        Z = self.sys.mgr.let(swapvars, Z)
        # Compute implication 
        Z = (~self.sys.pred | Z)
        # Eliminate x' and return 
        if no_inputs:
            controlbits = flatten([self.sys.pred_bitvars[c] for c in self.sys.control])
            return self.elimcontrol(controlbits, (self.nonblock & self.elimpost(postbits, Z)))
        else:
            return self.nonblock & self.elimpost(postbits, Z)

class SafetyGame():
    def __init__(self, cpre, safeset):
        self.cpre = cpre
        self.safe = safeset # TODO: Check if a subset of the state space 
    def __call__(self, steps = None):
        if steps: 
            assert steps >= 0 

        z = self.cpre.sys.mgr.true
        zz = self.cpre.sys.mgr.false

        i = 0
        while (z != zz):
            if steps and i == steps:
                break
            zz = z 
            z = zz & self.cpre(zz, no_inputs = True) & self.safe 
            i += 1
        
        return z, i 

class ReachAvoidGame():
    def __init__(self):
        pass 

    def __call__(self):
        pass


class ReachGame():
    def __init__(self):
        pass 

    def __call__(self):
        pass