"""
Controller interface classes


"""

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
        pt_bdd = self.cpre.sys.mgr.true
        forall_bits = []  # Post state bits
        exists_bits = []  # Pre state bits
        for k, v in state.items():
            nbits = len(self.cpre.sys.pred_bitvars[k])
            pt_bdd &= self.cpre.prestate[k].pt2bdd(self.cpre.sys.mgr, k, v, nbits)

        # Safe state-input pairs
        xu = pt_bdd & self.C

        # Safe control inputs
        u = self.cpre.elimprestate(xu)

        # Generate allowed controls
        for u_assignment in self.cpre.sys.mgr.pick_iter(u):
            # Translate BDD assignment into concrete counterpart
            uval = dict()
            for uvar in self.cpre.control:
                ubits = [k for k in u_assignment if _name(k) == uvar]
                ubits.sort()
                bv = [u_assignment[bit] for bit in ubits]
                uval[uvar] = self.cpre.control[uvar].bv2conc(bv)
            yield uval


