from functools import reduce
from vpax.controller import MemorylessController 

flatten = lambda l: [item for sublist in l for item in sublist]

def _name(i):
    return i.split('_')[0]

def _idx(i):
    return i.split('_')[1]

class ControlPre():
    def __init__(self, controlsys):
        self.sys = controlsys
        self.nonblock = controlsys.nonblock()

        self.elimcontrol = lambda bits, pred : self.sys.mgr.exist(bits, pred) # FIXME: exists
        self.elimpost = lambda bits, pred: self.sys.mgr.forall(bits, ~self.sys.outspace() | pred)

        prebits = flatten([self.sys.pred_bitvars[state] for state in self.sys.prestate])
        self.postbits = [self.sys.pre_to_post[_name(i)] + '_' + _idx(i) for i in prebits]
        self.swapvars = {i:j for i,j in zip(prebits, self.postbits)} 

    def __call__(self, Z, no_inputs = False):
        """
        Controllable predecessor for target next state set Z(x')

        Args:
            Z (bdd):

        nonblock /\ forall x'. (sys(x,u,x') => Z(x'))
        """

        # Exchange Z's pre state variables for post state variables 
        Z = self.sys.mgr.let(self.swapvars, Z)
        # Compute implication 
        Z = (~self.sys.pred | Z)
        # Eliminate x' and return 
        if no_inputs:
            controlbits = flatten([self.sys.pred_bitvars[c] for c in self.sys.control])
            return self.elimcontrol(controlbits, (self.nonblock & self.elimpost(self.postbits, Z)))
        else:
            return self.nonblock & self.elimpost(self.postbits, Z)


class SafetyGame():
    """
    Attributes: 
        cpre: 
        safe: 
    """
    def __init__(self, sys, safeset):
        self.cpre = ControlPre(sys)
        self.safe = safeset # TODO: Check if a subset of the state space

    def step(self, steps = None, winning = None):
        """
        Runs a safety game until reaching a fixed point or a maximum number of steps

        Args: 
            steps (int): Maximum number of game steps
            winning (dd BDD): Current winning set

        Returns:
            dd BDD: Safe invariant region
            int   : Number of game steps run 
        """
        if steps: 
            assert steps >= 0 

        z = self.cpre.sys.mgr.true if winning is None else winning
        zz = self.cpre.sys.mgr.false

        i = 0
        while (z != zz):
            if steps and i == steps:
                break 
            zz = z 
            z = zz & self.cpre(zz, no_inputs = True) & self.safe 
            i += 1
        
        return z, i

    def get_controller(self, winningset):
        """
        Takes winning set for invariance game and outputs a controller object

        Args:
            winningset (bdd): Winning region for invariance game 
        
        Returns: 
            MemorylessController TODO: finish this 
        """
        raise NotImplementedError

        c = MemorylessController(self.cpre.sys) 
        mgr = self.cpre.sys.mgr 
        def safecontrols(state):
            """


            """
            assert (state.keys() == self.cpre.sys.prestate.keys())

            pt_bdd = mgr.true
            elim_bits = []
            for k,v in state:
                elim_bits += self.cpre.sys.pred_bitvars[k]
                nbits = len(self.cpre.sys.pred_bitvars[k])
                pt_bdd &= self.cpre.sys.prestate[k].pt2bdd(mgr, k, state, nbits)

            # x AND forall x'.(sys(x,u) => winningset)
            # equivalent to forall x'( x & ~sys | x & winningset )
            xu = mgr.forall(elim_bits, (pt_bdd & ~self.cpre.sys.pred) | (pt_bdd & winningset))
            
            xu = mgr.exist(self.cpre.sys.prestate.keys(), xu)
            # Return generator for safe controls
            for ubdd in mgr.pick_iter(xu):
                # Translate BDD to an input box... 
                yield  

        # Override abstract call method 
        c.__call__ = lambda s: safecontrols(s)
        
        return c 


class ReachGame():
    def __init__(self, sys, target):
        self.cpre = ControlPre(sys)
        self.target = target # TODO: Check if a subset of the state space 

    def step(self, steps = None):
        """
        Runs a reachability game until reaching a fixed point or a maximum number of steps

        Args: 
            steps (int): Maximum number of game steps 

        Returns:
            dd BDD: Backward reachable set 
            int   : Number of game steps run
        """
        
        if steps: 
            assert steps >= 0

        z = self.cpre.sys.mgr.false
        zz = self.cpre.sys.mgr.true 

        i = 0
        while (z != zz):
            if steps and i == steps:
                break
            zz = z
            z = zz | self.cpre(zz, no_inputs = True) | self.target
            i += 1
        
        return z, i 

class ReachAvoidGame():
    def __init__(self):
        pass 

    def __call__(self):
        pass