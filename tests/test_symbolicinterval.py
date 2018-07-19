from vpax.symbolicinterval import *
from dd.cudd import BDD 

from pytest import approx, raises 

def test_helpers():
    assert len(index_interval(4,5, nbits = 4, graycode = True)) == 16
    assert index_interval(5,4, nbits = 4, graycode = True) == [5,4]
    
    for i in range(4):
        assert index_interval(i,i, 2, True) == [i]
        assert index_interval(i,i, 2, False) == [i]

def test_dynamic_regular():
    mgr = BDD() 
    x = DynamicInterval('x', mgr, -2, 2)
    assert x.num_bits == 0 
    x = x.withbits(3)
    assert x.num_bits == 3
    assert x.pt2bv(-2) == [False] * 3 
    assert x.pt2bv(2) == [True] * 3
    assert x.pt2box(-.1) == (approx(-.5), approx(0))
    assert x.pt2box(.6) == (approx(.5), approx(1.0))
    assert x.pt2box(.6, nbits = 4) == (approx(.5), approx(.75))
    assert x.pt2box(.6, nbits = 2) == (approx(0), approx(1.0))
    assert x.pt2box(.6, nbits = 1) == (approx(0), approx(2.0))

    with raises(AssertionError):
        x.pt2bv(3)
        
    # Identical names 
    xcopy = DynamicInterval('x', mgr, -3, 3, num_bits=3)
    assert list(xcopy.bits.keys()) == ['x_0', 'x_1', 'x_2']
    assert len(mgr.vars) == 3

    # Coarsening maintains the manager variables
    x = x.withbits(1)
    xcopy = xcopy.withbits(1)
    assert (xcopy.bits.keys() == x.bits.keys())
    assert (xcopy == x) is False
    assert len(mgr.vars) == 3

    # Manager keeps finer variables 
    x.withbits(8)
    assert len(x.bits) == 1
    assert len(mgr.vars) == 8 

def test_dynamic_periodic():
    mgr = BDD()
    x = DynamicInterval('x', mgr, 0, 20, periodic = True)
    x = x.withbits(4) 
    assert x.pt2bv(11) == [True, True, False, False]
    assert x.pt2bv(19+20) == [True, False, False, False]

def test_fixed_regular():
    mgr = BDD()
    x = FixedInterval('x', mgr, -3, 7, 13)
    assert x.num_bits == 4
    assert x.pt2index(-3) == 0
    assert x.pt2index(7) == 12

    bv_box = list(x.box2bvs((.1, 3.7)))
    bv_innerbox = list(x.box2bvs((.1,3.7), innerapprox=True))
    assert bv_box == [[False, True, False, False], 
                        [False, True, False, True],
                        [False, True, True, False],
                        [False, True, True, True],
                        [True, False, False, False]]
    assert len(bv_box) == len(bv_innerbox) + 2

def test_fixed_periodic():
    assert True

def test_discrete():
    assert True

# def conc(x):
#     return list(mgr.pick_iter(x))

# print(conc(x.box2bdd((-.5,.5), False)))
# print(conc(x.box2bdd((-.5,.5), True)))

# print(conc(x.box2bdd((-.6,.1), False)))
# print(conc(x.box2bdd((-.6,.1), True)))



# y = SymbolicInterval.DynamicInterval('y', mgr, -2, 2,periodic=True)
# y.add_bits(3)
# print(conc(y.box2bdd((-.5,.5), False)))
# print(conc(y.box2bdd((-.5,.5), True)))

# print(conc(y.box2bdd((-.6,.1), False)))
# print(conc(y.box2bdd((-.6,.1), True)))
