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

    def __init__(self, sys, allowed_controls):
        SupervisoryController.__init__(self)
        self.sys = sys
        self.C = allowed_controls

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

        assert (state.keys() == self.sys.prestate.keys())

        # Convert concrete state to BDD
        pt_bdd = self.sys.mgr.true
        forall_bits = []  # Post state bits
        exists_bits = []  # Pre state bits
        for k, v in state.items():
            poststate = self.sys.pre_to_post[k]
            forall_bits += self.sys.pred_bitvars[poststate]
            exists_bits += self.sys.pred_bitvars[k]
            nbits = len(self.sys.pred_bitvars[k])
            pt_bdd &= self.sys.prestate[k].pt2bdd(self.sys.mgr, k, v, nbits)

        # Safe state-input pairs
        xu = pt_bdd & self.C

        # Safe control inputs
        u = self.sys.mgr.exist(exists_bits, xu)

        # Generate allowed controls
        for u_assignment in self.sys.mgr.pick_iter(u):
            # Translate BDD assignment into concrete counterpart
            uval = dict()
            for uvar in self.sys.control.keys():
                ubits = [k for k in u_assignment if _name(k) == uvar]
                ubits.sort()
                bv = [u_assignment[bit] for bit in ubits]
                uval[uvar] = self.sys.control[uvar].bv2conc(bv)
            yield uval


