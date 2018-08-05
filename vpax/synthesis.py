from functools import reduce

flatten = lambda l: [item for sublist in l for item in sublist]

def _name(i):
    return i.split('_')[0]


def _idx(i):
    return i.split('_')[1]


class ControlPre():
    def __init__(self, controlsys):
        self.sys = controlsys
        self.nonblock = controlsys.nonblock()

        self.elimcontrol = lambda bits, pred : self.sys.mgr.exist(bits, pred)  #FIXME: exists
        self.elimpost = lambda bits, pred: self.sys.mgr.forall(bits, ~self.sys.outspace() | pred)

        prebits = flatten([self.sys.pred_bitvars[state] for state in self.sys.prestate])
        self.postbits = [self.sys.pre_to_post[_name(i)] + '_' + _idx(i) for i in prebits]
        self.swapvars = {i: j for i, j in zip(prebits, self.postbits)}

    def __call__(self, Z, no_inputs = False):
        r"""
        Controllable predecessor for target next state set Z(x')

        Args:
            Z (bdd):
            no_inputs (bool): If false then returns a (pre state,control) predicate. If true, returns a pre state predicate.

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
    Safety game solver.

    Attributes:
        sys (ControlModule): Control system module
        safe (bdd): Safe set
    """
    def __init__(self, sys, safeset):
        self.cpre = ControlPre(sys)
        self.sys  = sys
        self.safe = safeset # TODO: Check if a subset of the state space

    def step(self, steps=None, winning=None):
        """
        Run a safety game until reaching a fixed point or a maximum number of steps.

        Args:
            steps (int): Maximum number of game steps
            winning (dd BDD): Intermediate winning set

        Returns:
            dd BDD    : Safe invariant region
            int       : Actualy number of game steps run.
            generator : Controller that maps state dictionary to safe input dictionary
        """
        if steps is not None:
            assert steps >= 0

        z = self.sys.statespace() if winning is None else winning
        zz = self.sys.mgr.false

        i = 0
        while (z != zz):
            if steps and i == steps:
                break
            zz = z
            z = self.cpre(zz, no_inputs=True) & self.safe
            i += 1

        def safecontrols(state):
            r"""

            """
            assert (state.keys() == self.sys.prestate.keys())

            # Convert concrete state to BDD
            pt_bdd = self.sys.mgr.true
            forall_bits = []
            exists_bits = []
            for k, v in state.items():
                poststate = self.sys.pre_to_post[k]
                forall_bits += self.sys.pred_bitvars[poststate]
                exists_bits += self.sys.pred_bitvars[k]
                nbits = len(self.sys.pred_bitvars[k])
                pt_bdd &= self.sys.prestate[k].pt2bdd(self.sys.mgr, k, v, nbits)

            # Safe state-input pairs
            xu = pt_bdd & z & self.cpre(z, no_inputs = False) & self.safe

            # Safe control inputs
            u = self.sys.mgr.exist(exists_bits, xu)

            # Return generator for safe controls
            for u_assignment in self.sys.mgr.pick_iter(u):
                # Translate BDD assignment into concrete counterpart
                uval = dict()
                for uvar in self.sys.control.keys():
                    ubits = [k for k in u_assignment if _name(k) == uvar]
                    ubits.sort()
                    bv = [u_assignment[bit] for bit in ubits]
                    uval[uvar] = self.sys.control[uvar].bv2conc(bv)
                yield uval

        return z, i, safecontrols


class ReachGame():
    """
    Reach game solver.

    Attributes:
        sys (ControlModule): Control system module
        target (bdd): Target set
    """
    def __init__(self, sys, target):
        self.cpre = ControlPre(sys)
        self.target = target # TODO: Check if a subset of the state space
        self.sys = sys

    def step(self, steps = None, winning=None):
        """
        Run a reachability game until reaching a fixed point or a maximum number of steps.

        Args:
            steps (int): Maximum number of game steps
            winning(bdd): Currently winning region

        Returns:
            dd BDD: Backward reachable set
            int   : Number of game steps run
        """

        if steps is not None:
            assert steps >= 0

        z = self.sys.mgr.false if winning is None else winning
        zz = self.sys.mgr.true

        i = 0
        while (z != zz):
            if steps and i == steps:
                break
            zz = z
            z = self.cpre(zz, no_inputs = True) | self.target
            i += 1

        return z, i

class ReachAvoidGame():
    def __init__(self, sys, safe, target):
        self.cpre = ControlPre(sys)
        self.target = target  # TODO: Check if a subset of the state space
        self.safe = safe
        self.sys = sys

    def __call__(self, steps = None):
        """
        Run a reach-avoid game until reaching a fixed point or a maximum number of steps.

        Minimum fixed point

        Args:
            steps (int): Maximum number of game steps

        Returns:
            dd BDD: Safe backward reachable set
            int   : Number of game steps run
        """

        if steps:
            assert steps >= 0

        z = self.sys.mgr.false
        zz = self.sys.mgr.true

        i = 0
        while (z != zz):
            if steps and i == steps:
                break
            zz = z
            # z = (zz | self.cpre(zz, no_inputs = True) | self.target) & self.safe
            z = (self.cpre(zz, no_inputs=True) & self.safe) | self.target
            i += 1

        return z, i