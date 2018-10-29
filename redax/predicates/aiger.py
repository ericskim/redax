
import funcy as fn

import aiger
from aiger_analysis.bdd import from_bdd, to_bdd

try:
    from dd.cudd import BDD
except ImportError:
    try:
        from dd.autoref import BDD
    except ImportError:
        raise ImportError(
            "Cannot import dd.cudd or dd.autoref." +
            "Reinstall with BDD support."
)

aa = None
bddmgr = BDD()


to_bdd = fn.partial(to_bdd, manager=bddmgr)
from_bdd = fn.partial(from_bdd, manager=bddmgr)


class AAG():
    """
    Wrapper around py-aiger's BoolExpr class.

    Used to support an interface that's analogous to dd.

    """
    def __init__(self, aag):
        self.aag = aag

    @property
    def support(self):
        return self.aag.inputs

    def __invert__(self):
        return AAG(~self.aag)

    def __or__(self, other): 
        return AAG(self.aag | other.aag)

    def __and__(self, other):
        return AAG(self.aag & other.aag)

    def __eq__(self, other):
        # return aa.is_equal(self.aag, other.aag) # sat-based interface

        ### BDD based interface for now
        bdd, _, vartable = to_bdd(self.aag)
        bddmgr.declare(*vartable.inv.values())
        bdd = bddmgr.let(vartable.inv, bdd)
        return to_bdd(self.aag)[0] == to_bdd(other.aag)[0]

    def __iand__(self, other):
        return AAG(self.aag & other.aag) 

    def __ior__(self, other):
        return AAG(self.aag | other.aag)

class aigerwrapper(object): 
    """
    Wrapper around py-aiger.

    Mimics the interface of dd.
    """
    def __init__(self):
        pass

    def declare(self, name: str):
        pass

    def var(self, name: str):
        return AAG(aiger.atom(name))

    def exist(self, vars, pred):
        # return AAG(aa.eliminate(aag, vars))
        aagbdd, _, vartable = to_bdd(pred.aag)
        bddmgr.declare(*vartable.inv.values())
        aagbdd = bddmgr.let(vartable.inv, aagbdd)
        return AAG(from_bdd( bddmgr.exist(vars, aagbdd) ))

    def forall(self, vars, pred):
        # return AAG(~aa.eliminate(~aag, vars))
        aagbdd, _, vartable = to_bdd(pred.aag)
        bddmgr.declare(*vartable.inv.values())
        aagbdd = bddmgr.let(vartable.inv, aagbdd)
        return AAG(from_bdd( bddmgr.forall(vars, aagbdd)))

    @property
    def true(self):
        return AAG(aiger.atom(True))
    
    @property
    def false(self):
        return AAG(aiger.atom(False))

    def count(self, aag, bits): 
        raise NotImplementedError

    def pick_iter(self, aag):
        raise NotImplementedError


