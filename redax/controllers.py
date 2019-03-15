"""
Controller interface classes


"""

# from redax.synthesis import ControlPre, DecompCPre

from redax.spaces import OutOfDomainError
from redax.module import Interface
from redax.utils.bv import bv_var_name, bv_var_idx
from redax.ops import ihide


class MemorylessController():
    """
    MemorylessController.

    Parameters
    ----------
    sys: ControlPre
        The system to be controlled
    allowed_controls: bdd
        BDD encoding state-input pairs

    Methods
    -------
    allows(state: dict) -> generator:
        Maps state dict to a generator outputing dicts of allowed inputs

    """

    def __init__(self, cpre, allowed_controls: Interface):
        self.cpre = cpre
        self.C = allowed_controls

    def isempty(self):
        return self.C.count_nb == 0

    def winning_set(self):
        return ihide(self.cpre.control.keys(), self.C)


    def winning_states(self, exclude=None):
        r"""
        Generator for states from the winning set.

        Parameters
        ----------
        exclude: bdd
            Set of states to exclude from generation.

        Returns
        -------
        generator
            Yields dictionaries with state var keys and concrete values
        """
        winning = ihide(self.cpre.control.keys(), self.C)

        # assert exclude.support.issubset(winning.support)

        exclude = self.cpre.mgr.false if exclude is None else exclude

        # Generate a winning point
        for x_assignment in self.cpre.mgr.pick_iter(winning.assum & ~exclude):
            # Translate BDD assignment into concrete counterpart
            xval = dict()
            for xvar in self.cpre.prestate:
                xbits = [k for k in x_assignment if bv_var_name(k) == xvar]
                xbits.sort()
                bv = [x_assignment[bit] for bit in xbits]
                xval[xvar] = self.cpre.prestate[xvar].bv2conc(bv)
            yield xval


    def allows(self, state):
        """
        Compute the set of allowed inputs associated with a state.

        Parameters
        ----------
        state: dict
            Keys are module variables and values are concrete values

        Returns
        ----------
        generator
            Yields dictionaries with control variable keys and allowed concrete input values

        """

        assert (state.keys() == self.cpre.prestate.keys())

        # Convert concrete state to BDD
        pt_bdd = self.cpre.mgr.true
        for k, v in state.items():
            nbits = len(self.cpre.sys.pred_bitvars[k])
            try:
                pt_bdd &= self.cpre.prestate[k].pt2bdd(self.cpre.mgr, k, v, nbits)
            except OutOfDomainError:
                print(k, v)
                raise

        # Get collection of admissible control inputs
        for i, assignment in enumerate(self.cpre.mgr.pick_iter(pt_bdd)):
            if i > 0:
                raise RuntimeError("Multiple discrete states assocaited with argument state.")
            u = self.cpre.mgr.let(assignment, self.C.assum)

        # Generate allowed controls
        for u_assignment in self.cpre.mgr.pick_iter(u):
            # Translate BDD assignment into concrete counterpart
            uval = dict()
            for uvar in self.cpre.control:
                ubits = [k for k in u_assignment if bv_var_name(k) == uvar]
                ubits.sort()
                bv = [u_assignment[bit] for bit in ubits]
                uval[uvar] = self.cpre.control[uvar].bv2conc(bv)
            yield uval

