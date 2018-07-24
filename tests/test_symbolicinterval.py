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
    x = DynamicInterval(-2, 2)
    assert x.pt2bv(-2,3) == [False] * 3 
    assert x.pt2bv(2,3) == [True] * 3
    assert x.pt2box(-.1, 3) == (approx(-.5), approx(0))
    assert x.pt2box(.6, 3) == (approx(.5), approx(1.0))
    assert x.pt2box(.6, nbits = 4) == (approx(.5), approx(.75))
    assert x.pt2box(.6, nbits = 2) == (approx(0), approx(1.0))
    assert x.pt2box(.6, nbits = 1) == (approx(0), approx(2.0))

    # No bits yields the entire interval 
    assert x.pt2box(1, 0) == (approx(-2), approx(2))

    with raises(AssertionError):
        x.pt2bv(3, 1)
        x.pt2bv(-1, 1)


    # BDD creation
    x.box2pred(mgr, "x", (.4,.6), 3)

    # TODO: Inner and outer approximation tests

def test_dynamic_periodic():
    x = DynamicInterval(0, 20, periodic = True)
    assert x.pt2bv(11,4) == [True, True, False, False]
    assert x.pt2bv(19+20,4) == [True, False, False, False]


def test_fixed_regular():
    x = FixedInterval( -3, 7, 13)
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
    pass 

def test_discrete():
    pass

def test_utils():
    pass

# def conc(x):
#     return list(mgr.pick_iter(x))

# print(conc(x.box2pred((-.5,.5), False)))
# print(conc(x.box2pred((-.5,.5), True)))

# print(conc(x.box2pred((-.6,.1), False)))
# print(conc(x.box2pred((-.6,.1), True)))



# y = SymbolicInterval.DynamicInterval('y', mgr, -2, 2,periodic=True)
# y.add_bits(3)
# print(conc(y.box2pred((-.5,.5), False)))
# print(conc(y.box2pred((-.5,.5), True)))

# print(conc(y.box2pred((-.6,.1), False)))
# print(conc(y.box2pred((-.6,.1), True)))
