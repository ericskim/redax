from functools import reduce

from vpax.controllers import MemorylessController
from vpax.utils import flatten

# flatten = lambda l: [item for sublist in l for item in sublist]

def _name(i):
    return i.split('_')[0]


def _idx(i):
    return i.split('_')[1]


class ControlPre():
    """Controlled Predecessor Computer."""

    def __init__(self, controlsys):
        self.sys = controlsys
        self.nonblock = controlsys.nonblock()

        prebits = flatten([self.sys.pred_bitvars[state] for state in self.sys.prestate])
        self.postbits = [self.sys.pre_to_post[_name(i)] + '_' + _idx(i) for i in prebits]
        self.swapstates = {i: j for i, j in zip(prebits, self.postbits)}

    def elimcontrol(self, bits, pred):
        r"""Existentially eliminate control bits from a predicate."""
        return self.sys.mgr.exist(bits, pred & self.sys.controlspace())

    def elimpost(self, bits, pred): 
        r"""Universally eliminate post state bits from a predicate."""
        return self.sys.mgr.forall(bits, ~self.sys.outspace() | pred)

    def __call__(self, Z, no_inputs = False):
        r"""
        Compute controllable predecessor for target next state set.

        Computes  nonblock /\ forall x'. (sys(x,u,x') => Z(x'))

        Parameters
        ----------
        Z: bdd
            Target set (expressed over pre states) at next time step
        no_inputs: bool
            If false then returns a (pre state,control) predicate. If true, returns a pre state predicate.

        """
        # Exchange Z's pre state variables for post state variables
        Z = self.sys.mgr.let(self.swapstates, Z)
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
    Safety game solver.

    Attributes
    ----------
    sys: ControlSystem
        Control system to synthesize for
    safe: bdd
        Safe region predicate

    """

    def __init__(self, sys, safeset):
        self.cpre = ControlPre(sys)
        self.sys  = sys
        self.safe = safeset # TODO: Check if a subset of the state space

    def step(self, steps=None, winning=None):
        """
        Run a safety game until reaching a fixed point or a maximum number of steps.

        Parameters
        ----------
        steps: int
            Maximum number of game steps to run 
        winning: int
            Currently winning region

        Returns
        -------
        bdd: 
            Safe invariant region
        int:
            Actualy number of game steps run.
        generator: 
            Controller that maps state dictionary to safe input dictionary

        """
        if steps is not None:
            assert steps >= 0

        C = self.sys.mgr.false

        z = self.sys.statespace() if winning is None else winning
        zz = self.sys.mgr.false

        i = 0
        while (z != zz):
            if steps and i == steps:
                break
            zz = z
            C = self.cpre(zz) & self.safe
            ubits = [k for k in C.support if _name(k) in self.sys.control.keys()]
            i += 1
            z = self.sys.mgr.exist(ubits , C)

        return z, i, MemorylessController(self.sys, C)


class ReachGame():
    """
    Reach game solver.

    Attributes
    ----------
    sys: ControlSystem
        Control system to synthesize for
    target: bdd
        Target region predicate

    """

    def __init__(self, sys, target):
        self.cpre = ControlPre(sys)
        self.target = target # TODO: Check if a subset of the state space
        self.sys = sys


    def step(self, steps = None, winning=None):
        """
        Run a reachability game until reaching a fixed point or a maximum number of steps.

        Parameters
        ----------
        steps: int
            Maximum number of game steps to run 
        winning: int
            Currently winning region

        Returns
        -------
        bdd: 
            Backward reachable set
        int:
            Number of game steps run
        MemorylessController:
                Controller for the reach game

        """
        if steps is not None:
            assert steps >= 0

        C = self.sys.mgr.false 

        z = self.sys.mgr.false if winning is None else winning
        zz = self.sys.mgr.true

        i = 0
        while (z != zz):
            if steps and i == steps:
                break

            zz = z
            z = self.cpre(zz) | self.target # state-input pairs
            ubits = [k for k in z.support if _name(k) in self.sys.control.keys()]
            C = C | (z & (~self.sys.mgr.exist(ubits , C))) # Add new state-input pairs to controller            
            z = self.sys.mgr.exist(ubits , z)
            i += 1

        return z, i, MemorylessController(self.sys, C)

class ReachAvoidGame():
    """
    Reach-avoid game solver.

    Solves for the temporal logic formula "safe UNTIL target"

    Attributes
    ----------
    sys: ControlSystem
        Control system to synthesize for
    safe: bdd
        Safety region predicate
    target: bdd
        Target region predicate

    """
    
    def __init__(self, sys, safe, target):
        self.cpre = ControlPre(sys)
        self.target = target  # TODO: Check if a subset of the state space
        self.safe = safe
        self.sys = sys

    def __call__(self, steps = None):
        """
        Run a reach-avoid game until reaching a fixed point or a maximum number of steps.

        Solves for the temporal logic formula "safe UNTIL target"

        Parameters
        ----------
        steps: int 
            Maximum number of game steps to run 

        Returns
        -------
        bdd:
            Safe backward reachable set
        int:
            Number of game steps run
        MemorylessController:
            Controller for the reach-avoid game
            
        """
        if steps:
            assert steps >= 0

        C = self.sys.mgr.false

        z = self.sys.mgr.false
        zz = self.sys.mgr.true

        i = 0
        while (z != zz):
            if steps and i == steps:
                break

            zz = z
            # z = (zz | self.cpre(zz, no_inputs = True) | self.target) & self.safe 
            z = (self.cpre(zz) & self.safe) | self.target # State input pairs
            ubits = [k for k in z.support if _name(k) in self.sys.control.keys()]
            C = C | (z & (~self.sys.mgr.exist(ubits , C))) # Add new state-input pairs to controller
            i += 1
            z = self.sys.mgr.exist(ubits , z)

        return z, i, MemorylessController(self.sys, C)