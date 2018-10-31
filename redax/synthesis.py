import time

from functools import reduce

from bidict import bidict

from redax.controllers import MemorylessController, SafetyController
from redax.utils.bv import flatten
from redax.module import Interface, CompositeModule
from redax.ops import ohide, compose, ihide

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
        mod: redax.Interface
            Module
        states: iterable of (str, str) tuples
            (pre state name, post state name)
        control: Collection of str

        """

        try:
            mod.check()
        except:
            import warnings
            warnings.warn("Module does not pass check")

        self.sys = mod

        prepost = [(prepost[0], prepost[1]) for prepost in states]

        if not {post for _, post in prepost}.issubset(mod.outputs):
            raise ValueError("Unknown post state")
        if not {pre for pre, _ in prepost}.issubset(mod.inputs):
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

    def __call__(self, Z, no_inputs=False, verbose=False):
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
            self.mgr.declare(*swapvars.values())
            Z = self.mgr.let(swapvars, Z)

        # Compute implication
        Z = ~self.sys.pred | Z
        # Eliminate post states and return
        if no_inputs:
            return self.elimcontrol(self.sys.assum & self.elimpost(Z))
        else:
            return self.sys.assum & self.elimpost(Z)

    def modulepre(self, Z: Interface, no_inputs:bool=False, collapser:bool=None):


        if len(Z.outputs) > 0:
            raise ValueError("Only accept sink modules as inputs.")

        # Rename inputs from pre variables to post.
        Z = Z.renamed(self.pre_to_post)

        # Compute robust state-input pairs
        if collapser is None:
            xu = ohide(self.sys.outputs.keys(), compose(self.sys, Z))
        else:
            xu = collapser(self.sys, Z)

        # Return state-input pairs or only states
        if no_inputs:
            return ihide(self.control, xu)
        else:
            return xu

class DecompCPre(ControlPre):

    def __init__(self, mod: CompositeModule, states, control, elim_order = None) -> None:
        
        # Check if all modules aren't just a parallel composition
        if len(mod.sorted_mods()) > 2:
            raise NotImplementedError("Only implemented for parallel composed modules.")

        ControlPre.__init__(self, mod, states, control)
    
        self.elimorder = elim_order

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

    def __call__(self, Z, no_inputs=False, verbose=False):
        swapvars = self.swappedstates(Z)
        if len(swapvars) > 0:
            self.mgr.declare(*swapvars.values())
            Z = self.mgr.let(swapvars, Z)

        if self.elimorder is not None:
            to_elim_post = list(self.elimorder)
        else:
            to_elim_post = list(self.poststate)

        while(len(to_elim_post) > 0):
            var = to_elim_post.pop()
            if verbose:
                print("Eliminating", var, "with", len(self.mgr), "nodes in manager")

            # Partition into modules that do/don't depend on var
            dep_mods = tuple(mod for mod in self.sys.children if var in mod.outputs)

            # Find Z bit precision. 
            Z_var_bits = len([b for b in Z.support if _name(b) == var])

            # Aggregate implication and construct nonblocking set
            nb = self.mgr.true
            for mod in dep_mods:
                Z = ~(mod.coarsened(**{var: Z_var_bits})).pred | Z
                nb = nb & mod.assum

            Z = nb & self.elimpostslice(Z, var)

        # TODO: Assert Z's support is in the composite systems input range. Line below hasn't been checked
        # assert Z.support <= set(flatten([self.sys.pred_bitvars[v] for v in self.sys.inputs]))

        # Eliminate control inputs
        if no_inputs:
            if verbose:
                print("Eliminating Control")
            return self.elimcontrol(Z)
        else:
            return Z

    def modulepre(self, Z: Interface, no_inputs=False, collapser=None):

        Z = Z.renamed(self.pre_to_post)

        if self.elimorder is not None:
            to_elim_post = list(self.elimorder)
        else:
            to_elim_post = list(self.poststate)

        while(len(to_elim_post) > 0):
            var = to_elim_post.pop()

            # FIXME: This code assumes that each module only has a single output. 
            # Should instead iterate over modules

            # Partition into modules that do/don't depend on var
            dep_mods = tuple(mod for mod in self.sys.children if var in mod.outputs)

            if len(dep_mods) == 0:
                continue

            # Find Z bit precision
            Z_var_bits = len([b for b in Z.pred.support if _name(b) == var])
            assert Z_var_bits == len(Z.pred_bitvars[var])

            for mod in dep_mods:
                if collapser is None:
                    Z = ohide([var], compose(mod.coarsened(**{var: Z_var_bits}), Z))
                else:
                    Z = collapser(mod.coarsened(**{var: Z_var_bits}), Z)


        if no_inputs:
            return ihide(self.control, Z)
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

    def run(self, steps=None, winning=None, verbose=False, winningonly=False):
        """
        Run a safety game until reaching a fixed point or a maximum number of steps.

        Parameters
        ----------
        steps: int
            Maximum number of game steps to run
        winning: BDD
            Currently winning region
        verbose: bool, False
            If True (not default), then print out intermediate statistics.
        winningonly: bool, False
            If true, output safety controller that only stores the invariant region.

        Returns
        -------
        bdd: 
            Safe invariant region
        int:
            Actualy number of game steps run.
        MemorylessController: 
            Controller that maps state dictionary to safe input dictionary

        """
        if steps is not None:
            assert steps >= 0

        z = self.cpre.prespace() & self.safe if winning is None else winning
        zz = self.cpre.mgr.false

        C = self.cpre.mgr.false

        i = 0
        synth_start = time.time()
        while (z != zz):
            if steps and i == steps:
                break
            step_start = time.time()
            zz = z

            if winningonly:
                z = self.cpre(zz, verbose=verbose, no_inputs=True) & self.safe
            else:
                z = self.cpre(zz, verbose=verbose) & self.safe

                C = z
                if verbose:
                    print("Eliminating control")
                z = self.cpre.elimcontrol(C)

            i = i + 1

            if verbose:
                print("Step #: ", i,
                      "Step Time (s): {0:.3f}".format(time.time() - step_start), 
                      "Winning Size:", self.cpre.mgr.count(z, len(z.support)),
                      "Winning nodes:", len(z))
        if winningonly: 
            return z, i, SafetyController(self.cpre, z)
        else:
            return z, i, MemorylessController(self.cpre, C)


class OptimisticSafetyGame(SafetyGame):
    """
    Just like a safety game but returns a "best effort controller" that should ideally work if the adversary isn't too strong.
    """

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

        

        z = self.cpre.prespace() & self.safe if winning is None else winning
        zz = self.cpre.mgr.false

        C = z

        i = 0
        synth_start = time.time()
        while (z != zz):
            if steps and i == steps:
                break
            step_start = time.time()
            zz = z
            z = self.cpre(zz, verbose=verbose) & self.safe

            C = C & (~self.cpre.elimcontrol(z) | z)
            if verbose:
                print("Eliminating control")
            z = self.cpre.elimcontrol(z)
            i = i + 1

            if verbose:
                print("Step #: ", i,
                      "Step Time (s): {0:.3f}".format(time.time() - step_start), 
                      "Winning Size:", self.cpre.mgr.count(z, len(z.support)),
                      "Winning nodes:", len(z))

        return z, i, MemorylessController(self.cpre, C)


class ReachGame():
    """
    Reach game solver.

    Attributes
    ----------
    sys: ControlPre
        Control predecessor of system that needs to satisfy reach property.
    target: Interface
        Target region predicate

    """

    def __init__(self, cpre, target:Interface):
        self.cpre = cpre
        self.target = target # TODO: Check if a subset of the state space

    def run(self, steps=None, winning:Interface=None, verbose=False, excludewinning=False):
        """
        Run a reachability game until reaching a fixed point or a maximum number of steps.

        Parameters
        ----------
        steps: int
            Maximum number of game steps to run
        winning: int
            Currently winning region
        verbose: bool
            If True (not default), then print out intermediate statistics.
        excludewinning: bool
            If True, controllable predecessor will avoid synthesizing for states that are already in winning region.

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

        # z = self.cpre.prespace() & self.target if winning is None else winning
        # zz = self.cpre.mgr.true

        z = self.target if winning is None else winning
        zz = Interface(self.cpre.mgr, {}, {})

        i = 0
        synth_start = time.time()
        while (z != zz):
            if steps and i == steps:
                break

            zz = z
            step_start = time.time()
            if excludewinning:
                z = self.cpre(zz, verbose=verbose, excludedstates=zz) + self.target # state-input pairs
            else:
                z = self.cpre(zz, verbose=verbose) + self.target # state-input pairs
            C = C | (z.assum * (~self.cpre.elimcontrol(C))) # Add new state-input pairs to controller
            if verbose:
                print("Eliminating control")
            z = self.cpre.elimcontrol(C)

            i += 1
            if verbose:
                print("Step #: ", i,
                      "Step Time (s): ", time.time() - step_start,
                      "Size: ", self.cpre.mgr.count(z.assum, len(z.assum.support)),
                      "Winning nodes:", len(z.assum))

        return z, i, MemorylessController(self.cpre, C)


class ReachAvoidGame():
    """
    Reach-avoid game solver.

    Solves for the temporal logic formula "safe UNTIL target"

    Attributes
    ----------
    sys: ControlPre
        Control system to synthesize for
    safe: bdd
        Safety region predicate
    target: bdd
        Target region predicate

    """
    
    def __init__(self, cpre, safe, target):
        raise NotImplementedError("Needs to be refactored to take controllable predecessors")
        self.cpre = cpre
        self.target = target  # TODO: Check if a subset of the state space
        self.safe = safe

    def run(self, steps = None, winning=None, verbose=False):
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
        if steps is not None:
            assert steps >= 0

        C = self.cpre.mgr.false

        z = self.cpre.prespace() & self.target if winning is None else winning
        zz = self.cpre.mgr.true

        i = 0
        while (z != zz):
            if steps and i == steps:
                break

            zz = z
            step_start = time.time()
            z = (self.cpre(zz, verbose=verbose) & self.safe) | self.target # State input pairs
            C = C | (z & (~self.cpre.elimcontrol(C))) # Add new state-input pairs to controller

            if verbose:
                print("Eliminating control")
            z = self.cpre.elimcontrol(C)

            i += 1
            if verbose:
                print("Step #: ", i,
                      "Step Time (s): ", time.time() - step_start,
                      "Size: ", self.cpre.mgr.count(z, len(z.support)),
                      "Winning nodes:", len(z))

        return z, i, MemorylessController(self.cpre, C)

