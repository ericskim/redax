from functools import reduce

from bidict import bidict

from redax.controllers import MemorylessController
from redax.bvutils import flatten
from redax.module import AbstractModule, CompositeModule

# flatten = lambda l: [item for sublist in l for item in sublist]


def _name(i):
    return i.split('_')[0]


def _idx(i):
    return i.split('_')[1]


class ControlPre():
    r"""Controlled predecessors calculator."""

    def __init__(self, mod, states, control) -> None:
        """
        Constructor

        Parameters
        ----------
        mod: redax.AbstractModule
            Module
        states: iterable of (str, str) tuples
            (pre state name, post state name)
        control: Collection of str

        """
        self.sys = mod

        prepost = [(prepost[0], prepost[1]) for prepost in states]

        if not {post for pre, post in prepost}.issubset(mod.outputs):
            raise ValueError("Unknown post state")
        if not {pre for pre, post in prepost}.issubset(mod.inputs):
            raise ValueError("Unknown pre state")
        if not {ctrl for ctrl in control}.issubset(mod.inputs):
            raise ValueError("Unknown control input")
        if not {ctrl for ctrl in control}.isdisjoint(flatten(prepost)):
            raise ValueError("Variable cannot be both state and control")
        if any(mod.inputs[pre] != mod.outputs[post] for pre, post in prepost):
            raise ValueError("Pre and post state domains do not match")

        self.prestate = {i[0]: mod[i[0]] for i in states}
        self.poststate = {i[1]: mod[i[1]] for i in states}
        self.control = {i: mod[i] for i in control}
        self.pre_to_post = bidict({k[0]: k[1] for k in states})

    @property
    def mgr(self):
        return self.sys.mgr

    def controlspace(self):
        """Predicate for the control space."""
        space = self.mgr.true
        for var in self.control:
            space &= self.control[var].abs_space(self.mgr, var)
        return space

    def prespace(self):
        """Predicate for the current state space."""
        space = self.mgr.true
        for var in self.prestate:
            space &= self.prestate[var].abs_space(self.mgr, var)
        return space

    def postspace(self):
        """Predicate for the successor/post state space."""
        space = self.mgr.true
        for var in self.poststate:
            space &= self.poststate[var].abs_space(self.mgr, var)
        return space

    def elimcontrol(self, pred):
        r"""Existentially eliminate control bits from a predicate."""
        pred_in_space = pred & self.controlspace()
        elimbits = tuple(i for i in pred_in_space.support if _name(i) in self.control)
        return self.mgr.exist(elimbits, pred_in_space)

    def elimprestate(self, pred):
        r"""Existentially eliminate state bits from a predicate."""
        pred_in_space = pred & self.prespace()
        elimbits = tuple(i for i in pred_in_space.support if _name(i) in self.prestate)
        return self.mgr.exist(elimbits, pred_in_space)

    def elimpost(self, pred):
        r"""Universally eliminate post state bits from a predicate."""
        implies_pred = ~self.postspace() | pred
        elimbits = tuple(i for i in implies_pred.support if _name(i) in self.poststate)
        return self.mgr.forall(elimbits, implies_pred)

    def swappedstates(self, Z):
        r"""Swaps bit variables between predecessor and post states."""
        bits = Z.support

        postbits = tuple(self.pre_to_post[_name(i)] + '_' + _idx(i) for i in bits)
        return {pre: post for pre, post in zip(bits, postbits)}

    def __call__(self, Z, no_inputs=False):
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
        swapvars = self.swappedstates(Z)
        if len(swapvars) > 0:
            Z = self.mgr.let(self.swappedstates(Z), Z)
        # Compute implication
        Z = ~self.sys.pred | Z
        # Eliminate post states and return
        if no_inputs:
            return self.elimcontrol(self.sys._nb & self.elimpost(Z))
        else:
            return self.sys._nb & self.elimpost(Z)

class DecompCPre(ControlPre):

    def __init__(self, mod: CompositeModule, states, control) -> None:
        
        if len(mod.sorted_mods()) > 2:
            raise NotImplementedError("Only implemented for parallel composed modules.")

        ControlPre.__init__(self, mod, states, control)
    
    @property
    def mgr(self):
        return self.sys.children[0].mgr

    def postspaceslice(self, postvar):
        """Predicate for the successor/post state space."""
        return self.poststate[postvar].abs_space(self.mgr, postvar)

    def elimpostslice(self, pred, postvar):
        r"""Universally eliminate post state bits from a predicate."""
        implies_pred = ~self.postspaceslice(postvar) | pred
        elimbits = tuple(i for i in implies_pred.support if _name(i) == postvar)
        return self.mgr.forall(elimbits, implies_pred)

    def __call__(self, Z, no_inputs=False):
        swapvars = self.swappedstates(Z)
        if len(swapvars) > 0:
            Z = self.mgr.let(self.swappedstates(Z), Z)

        to_elim_post = [i for i in self.poststate]

        while(len(to_elim_post) > 0):
            var = to_elim_post.pop()
            # elimbits = tuple(i for i in implies_pred.support if _name(i) == var)

            # Partition into modules that do/don't depend on var
            dep_mods = [mod for mod in self.sys.children if var in mod.outputs]
            # indep_mods = [mod for mod in self.sys.children if var not in mod.outputs]

            # Aggregate implication and construct nonblocking
            nb = self.mgr.true
            for mod in dep_mods:
                Z = ~mod.pred | Z
                nb = nb & mod._nb

            Z = nb & self.elimpostslice(Z, var)

        # Eliminate
        if no_inputs:
            return self.elimcontrol(Z)
        else:
            return Z

class SafetyGame():
    """
    Safety game solver.

    Attributes
    ----------
    sys: ControlPre
        Control predecessor of system that needs to satisfy safety/invariance property.
    safe: bdd
        Safe region predicate

    """

    def __init__(self, cpre, safeset):
        self.cpre = cpre
        self.safe = safeset  # TODO: Check if a subset of the state space

    def run(self, steps=None, winning=None, verbose=False):
        """
        Run a safety game until reaching a fixed point or a maximum number of steps.

        Parameters
        ----------
        steps: int
            Maximum number of game steps to run
        winning: int
            Currently winning region
        verbose: bool, False
            If True (not default), then print out intermediate statistics.

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

        C = self.cpre.mgr.false

        z = self.cpre.prespace() if winning is None else winning
        zz = self.cpre.mgr.false

        i = 0
        while (z != zz):
            if steps and i == steps:
                break
            zz = z
            C = self.cpre(zz) & self.safe
            z = self.cpre.elimcontrol(C)
            i = i + 1

            if verbose:
                print("Step: ", i)

        return z, i, MemorylessController(self.cpre, C)


class ReachGame():
    """
    Reach game solver.

    Attributes
    ----------
    sys: ControlPre
        Control predecessor of system that needs to satisfy reach property.
    target: bdd
        Target region predicate

    """

    def __init__(self, cpre, target):
        self.cpre = cpre
        self.target = target # TODO: Check if a subset of the state space

    def run(self, steps=None, winning=None):
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

        C = self.cpre.mgr.false

        z = self.cpre.mgr.false if winning is None else winning
        zz = self.cpre.mgr.true

        i = 0
        while (z != zz):
            if steps and i == steps:
                break

            zz = z
            z = self.cpre(zz) | self.target # state-input pairs
            C = C | (z & (~self.cpre.elimcontrol(C))) # Add new state-input pairs to controller
            z = self.cpre.elimcontrol(z)
            i += 1

        return z, i, MemorylessController(self.cpre, C)


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
        raise NotImplementedError("Needs to be refactored to take controllable predecessors")
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

        C = self.mgr.false

        z = self.mgr.false
        zz = self.mgr.true

        i = 0
        while (z != zz):
            if steps and i == steps:
                break

            zz = z
            # z = (zz | self.cpre(zz, no_inputs = True) | self.target) & self.safe 
            z = (self.cpre(zz) & self.safe) | self.target # State input pairs
            ubits = tuple(k for k in z.support if _name(k) in self.sys.control.keys())
            C = C | (z & (~self.cpre.mgr.exist(ubits , C))) # Add new state-input pairs to controller
            i += 1
            z = self.cpre.mgr.exist(ubits , z)

        return z, i, MemorylessController(self.sys, C)


"""
def fp_iter(operation, starting, steps):

"""