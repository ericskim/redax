import time
from functools import reduce
from typing import Optional, Union, Sequence, Callable

from bidict import bidict

from redax.controllers import MemorylessController
from redax.utils.bv import flatten, bv_var_name
from redax.module import Interface, CompositeInterface
from redax.ops import compose, ihide, sinkprepend, coarsen, rename


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


    def elimcontrol(self, pred):
        r"""Existentially eliminate control bits from a predicate."""
        pred_in_space = pred & self.controlspace()
        elimbits = tuple(i for i in pred_in_space.support if bv_var_name(i) in self.control)
        return self.mgr.exist(elimbits, pred_in_space)


    def __call__(self, Z: Interface, no_inputs:bool=False, verbose=False) -> Interface:
        """One step control predecessor"""
        assert Z.is_sink()

        if len(Z.outputs) > 0:
            raise ValueError("Only accept sink modules as inputs.")

        # Rename inputs from pre variables to post.
        Z = rename(Z, self.pre_to_post)

        # Compute robust state-input pairs
        xu = sinkprepend(self.sys, Z)

        # Return state-input pairs or only states
        if no_inputs:
            return ihide(xu, self.control)
        else:
            return xu

class DecompCPre(ControlPre):     # TODO: Get rid of inheritance??

    def __init__(self,
                 mod: CompositeInterface,
                 states,
                 control,
                 elim_order: Sequence = None) -> None:

        # Check if all modules aren't just a parallel composition
        if len(mod.sorted_mods()) > 2:
            raise NotImplementedError("Only implemented for parallel composed modules.")

        ControlPre.__init__(self, mod, states, control)

        self.elimorder = elim_order


    @property
    def mgr(self):
        return self.sys.children[0].mgr


    def __call__(self, Z: Interface, no_inputs=False, verbose=False) -> Interface:
        """One step control predecessor"""
        assert Z.is_sink()

        Z = rename(Z, self.pre_to_post)

        # See if the user has provided a pre-determined order to compose interfaces.
        if self.elimorder is not None:
            to_elim_post = list(self.elimorder)
        else:
            to_elim_post = list(self.poststate)

        # Eliminate each interface output
        # FIXME: This code assumes that each module only has a single output.
        # Should instead iterate over modules. Find a better way to specify the
        # module to be outputted. Can't hash them unfortunately.
        # Also doesn't take into account arbitrary DAG topologies and control inputs.
        while(len(to_elim_post) > 0):
            var = to_elim_post.pop()

            # Partition into modules that do/don't depend on var
            dep_mods = tuple(mod for mod in self.sys.children if var in mod.outputs)

            if len(dep_mods) == 0:
                continue

            # Find Z bit precision.
            Z_var_bits = len([b for b in Z.assum.support if bv_var_name(b) == var])
            assert Z_var_bits == len(Z.pred_bitvars[var])

            for mod in dep_mods:
                Z = sinkprepend(coarsen(mod, **{var: Z_var_bits}), Z)

        if no_inputs:
            return ihide(Z, self.control)
        else:
            return Z


class CompConstrainedPre(DecompCPre):
    """
    Identical to decomppre but also coarsens the sink until a specific condition is met.
    """
    def __init__(self,
                 mod: CompositeInterface,
                 states,
                 control,
                 condition:Optional[Callable] = None,
                 heuristic:Optional[Callable] = None,
                 elim_order: Sequence = None,) -> None:
        ControlPre.__init__(self, mod, states, control)

        # Default condition never triggers heuristic
        self.condition = condition if condition else lambda x: False

        # Default heuristic does nothing and is identity function
        self.heuristic = heuristic if heuristic else lambda x: x

        self.elimorder = elim_order


    def __call__(self, Z: Interface, no_inputs=False, verbose=False) -> Interface:
        """
        One step decomposed control predecessor
        Coarsens Z interface whenever its complexity grows too large.
        """
        assert Z.is_sink()

        Z = rename(Z, self.pre_to_post)

        # See if the user has provided a pre-determined order to compose interfaces.
        if self.elimorder is not None:
            to_elim_post = list(self.elimorder)
        else:
            to_elim_post = list(self.poststate)

        if self.condition(Z):
            Z = self.heuristic(Z)

        # Eliminate each interface output
        # FIXME: This code assumes that each module only has a single output.
        # Should instead iterate over modules. Find a better way to specify the
        # module to be outputted. Can't hash them unfortunately.
        # Also doesn't take into account arbitrary DAG topologies and control inputs.
        while(len(to_elim_post) > 0):
            var = to_elim_post.pop()

            # Partition into modules that do/don't depend on var
            dep_mods = tuple(mod for mod in self.sys.children if var in mod.outputs)

            if len(dep_mods) == 0:
                continue

            # Find Z bit precision.
            Z_var_bits = len([b for b in Z.assum.support if bv_var_name(b) == var])
            assert Z_var_bits == len(Z.pred_bitvars[var])

            for mod in dep_mods:
                Z = sinkprepend(coarsen(mod, **{var: Z_var_bits}), Z)

        if no_inputs:
            return ihide(Z, self.control)
        else:
            return Z


class SafetyGame():
    """
    Safety game solver.

    Attributes
    ----------
    sys: ControlPre
        Control predecessor of system that needs to satisfy safety/invariance property.
    safe: Interface
        Safe region predicate

    """

    def __init__(self,
                 cpre: Union[ControlPre, DecompCPre, CompConstrainedPre],
                 safeset: Interface) -> None:
        self.cpre = cpre
        self.safe = safeset  # TODO: Check if a subset of the state space

    def run(self, steps: Optional[int]=None, winning: Optional[Interface]=None, verbose=False, winningonly=False):
        """
        Run a safety game until reaching a fixed point or a maximum number of steps.

        Parameters
        ----------
        steps: int
            Maximum number of game steps to run
        winning: Interface or None
            Currently winning region
        verbose: bool, False
            If True (not default), then print out intermediate statistics.
        winningonly: bool, False
            If true, output safety controller that only stores the invariant region.

        Returns
        -------
        Interface:
            Safe invariant region
        int:
            Actualy number of game steps run.
        MemorylessController:
            Controller that maps state dictionary to safe input dictionary

        """
        if steps is not None:
            assert steps >= 0

        z = self.safe if winning is None else winning
        zz = Interface(self.cpre.mgr, {}, {}) # Defaults to false interface.

        # C = self.cpre.mgr.false
        state_control = self.cpre.prestate.copy()
        state_control.update(self.cpre.control)
        # C = Interface(self.cpre.mgr, state_control, {})

        i = 0
        while (z != zz):
            if steps and i == steps:
                break
            step_start = time.time()
            zz = z

            z = self.cpre(zz, verbose=verbose)

            C = z
            if verbose:
                print("Eliminating control")

            z = ihide(z, self.cpre.control)
            z = z * self.safe

            i = i + 1

            if verbose:
                print("Step #: ", i,
                      "Step Time (s): {0:.3f}".format(time.time() - step_start),
                      "Winning Size:", self.cpre.mgr.count(z.assum, len(z.assum.support)),
                      "Winning nodes:", len(z.assum))

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

    def __init__(self,
                 cpre: Union[ControlPre, DecompCPre, CompConstrainedPre],
                 target: Interface) -> None:
        self.cpre: Interface = cpre
        self.target: Interface = target  # TODO: Check if a subset of the state space

    def run(self,
            steps: Optional[int]=None,
            winning: Optional[Interface]=None,
            verbose:bool=False):
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

        Returns
        -------
        Interface:
            Backward reachable set
        int:
            Number of game steps run
        MemorylessController:
                Controller for the reach game

        """
        if steps:
            assert steps >= 0

        state_control = self.cpre.prestate.copy()
        state_control.update(self.cpre.control)
        C = Interface(self.cpre.mgr, state_control, {})

        z = self.target if winning is None else winning
        zz = Interface(self.cpre.mgr, {}, {}) # Defaults to false interface.

        i = 0
        while (z != zz):
            if steps and i == steps:
                break

            zz = z
            step_start = time.time()
            z = self.cpre(zz, verbose=verbose) # state-input pairs
            # TODO: Change this to shared refinement?
            C._assum = C.assum | (z.assum & ~self.cpre.elimcontrol(C.assum))  # Add new state-input pairs to controller
            if verbose:
                print("Eliminating control")
            z = ihide(z, self.cpre.control)
            z = z + self.target

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
    safe: Interface
        Safety region predicate
    target: Interface
        Target region predicate

    """

    def __init__(self,
                 cpre: Union[ControlPre, DecompCPre, CompConstrainedPre],
                 safe: Interface,
                 target: Interface) -> None:

        self.cpre = cpre
        self.target = target  # TODO: Check if a subset of the state space
        self.safe = safe

        assert set(cpre.prestate.keys()) == set(target.inputs.keys())
        assert set(cpre.prestate.keys()) == set(safe.inputs.keys())

    def run(self,
            steps: Optional[int]=None,
            winning: Optional[Interface]=None,
            verbose:bool=False):
        """
        Run a reach-avoid game until reaching a fixed point or a maximum number of steps.

        Solves for the temporal logic formula "safe UNTIL target"

        Parameters
        ----------
        steps: int
            Maximum number of game steps to run

        Returns
        -------
        Interface:
            Safe backward reachable set
        int:
            Number of game steps run
        MemorylessController:
            Controller for the reach-avoid game

        """

        if steps:
            assert steps >= 0

        state_control_vars = self.cpre.prestate.copy()
        state_control_vars.update(self.cpre.control)
        C = Interface(self.cpre.mgr, state_control_vars, {})

        z = self.target if winning is None else winning
        zz = Interface(self.cpre.mgr, {}, {}) # Defaults to false interface.

        i = 0
        while (z != zz):
            if steps and i == steps:
                print("Reached step limit")
                break

            zz = z
            step_start = time.time()
            z = self.cpre(zz, verbose=verbose) # state-input pairs
            # z = (self.cpre(zz, verbose=verbose) * self.safe) + self.target  # state-input pairs
            # C = C | (z.assum & ~self.cpre.elimcontrol(C))  # Add new state-input pairs to controller
            C._assum = C.assum | (z.assum & ~self.cpre.elimcontrol(C.assum))  # Add new state-input pairs to controller
            C._assum &= self.safe.assum

            if verbose:
                print("Eliminating control")
            z = ihide(z, self.cpre.control)

            z = (z * self.safe) + self.target

            i += 1
            if verbose:
                print("Step #: ", i,
                      "Step Time (s): ", time.time() - step_start,
                      "Size: ", self.cpre.mgr.count(z.assum, len(z.assum.support)),
                      "Winning nodes:", len(z.assum))


        return z, i, MemorylessController(self.cpre, C)
