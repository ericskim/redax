from redax.predicates.aiger import aigerwrapper
from redax.spaces import DynamicCover, EmbeddedGrid, FixedCover, DiscreteSet, OutOfDomainError


def test_aiger_space():

    mgr = aigerwrapper()

    x0 = mgr.var('x_0')
    negx0 = ~x0

    x = DynamicCover(0, 16)

    xpred = x.conc2pred(mgr, 'x', (.5,1.5) , 4, innerapprox=False)

    assert xpred.support == {'x_0', 'x_1', 'x_2'}

    assert xpred == xpred 

    assert mgr.exist({'x_0', 'x_1', 'x_2'}, xpred) == mgr.true
    assert mgr.forall({'x_0', 'x_1', 'x_2'}, xpred) == mgr.false
    assert ~mgr.exist({'x_0', 'x_1', 'x_2'}, ~xpred) == mgr.false
    assert ~mgr.forall({'x_0', 'x_1', 'x_2'}, ~xpred) == mgr.true
