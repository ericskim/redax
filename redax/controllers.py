"""
Controller interface classes


"""

# from redax.synthesis import ControlPre, DecompCPre

def _name(i):
    return i.split('_')[0]


def _idx(i):
    return i.split('_')[1]


class SupervisoryController(object):
    pass


class MemorylessController(SupervisoryController):
    """
    MemorylessController.

    Parameters
    ----------
    sys: ControlSystem
        The system to be controlled
    allowed_controls: bdd
        BDD encoding state-input pairs

    Methods
    -------
    allows(state: dict) -> generator: 
        Maps state dict to a generator outputing dicts of allowed inputs

    """

    def __init__(self, cpre, allowed_controls):
        SupervisoryController.__init__(self)
        self.cpre = cpre
        self.C = allowed_controls

    def isempty(self):
        return self.C == self.cpre.mgr.false

    def winning_states(self):
        r"""
        Generates a state from the winning set

        Returns
        -------
        generator
            Yields dictionaries with state var keys and concrete values
        """
        winning = self.cpre.elimcontrol(self.C)
        
        # Generate a winning point
        for x_assignment in self.cpre.mgr.pick_iter(winning):
            # Translate BDD assignment into concrete counterpart
            xval = dict()
            for xvar in self.cpre.prestate:
                xbits = [k for k in x_assignment if _name(k) == xvar]
                xbits.sort()
                bv = [x_assignment[bit] for bit in xbits]
                xval[xvar] = self.cpre.prestate[xvar].bv2conc(bv)
            yield xval


    def allows(self, state): # -> Generator[Dict[str, concretetype], None, None]
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
            pt_bdd &= self.cpre.prestate[k].pt2bdd(self.cpre.mgr, k, v, nbits)

        # TODO: To compute u should assign x variable... but this works
        xu = pt_bdd & self.C  # Safe state-input pairs
        u = self.cpre.elimprestate(xu)  # Safe control inputs

        # Generate allowed controls
        for u_assignment in self.cpre.mgr.pick_iter(u):
            # Translate BDD assignment into concrete counterpart
            uval = dict()
            for uvar in self.cpre.control:
                ubits = [k for k in u_assignment if _name(k) == uvar]
                ubits.sort()
                bv = [u_assignment[bit] for bit in ubits]
                uval[uvar] = self.cpre.control[uvar].bv2conc(bv)
            yield uval

