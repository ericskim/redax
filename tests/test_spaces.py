import numpy as np
from pytest import approx, raises

from sydra.spaces import DynamicCover, EmbeddedGrid, FixedCover
from sydra.utils import bv_interval, bvwindow, index_interval

try:
    from dd.cudd import BDD
except ImportError:
    from dd.autoref import BDD


def test_helpers():
    assert len(index_interval(4, 5, nbits=4, graycode=True)) == 16
    assert index_interval(5, 4, nbits=4, graycode=True) == [5, 4]
    
    for i in range(4):
        assert index_interval(i, i, 2, True) == [i]
        assert index_interval(i, i, 2, False) == [i]


def test_dynamic_regular():
    mgr = BDD() 
    x = DynamicCover(-2.0, 2.0)
    assert x.pt2index(-2.0, 3) == 0
    assert x.pt2index(2.0, 3) == 8 # TODO: 3 bits = 0-7 but need to return 8 b/c of the right/left align detection
    assert x.pt2index(1.99, 3) == 7
    assert x.pt2index(-1, 1) == 0

    assert x.pt2box(-.1, 3) == (approx(-.5), approx(0))
    assert x.pt2box(.6, 3) == (approx(.5), approx(1.0))
    assert x.pt2box(.6, nbits=4) == (approx(.5), approx(.75))
    assert x.pt2box(.6, nbits=2) == (approx(0), approx(1.0))
    assert x.pt2box(.6, nbits=1) == (approx(0), approx(2.0))

    # No bits yields the entire interval
    assert x.pt2box(1, 0) == (approx(-2), approx(2))

    with raises(AssertionError):
        x.pt2bv(3, 1)

    # Inner approximation tests
    assert set(x.box2bvs((.4, .6), 2, innerapprox=True)) == set([])
    assert set(x.box2bvs((-.4, 1.6), 2, innerapprox=True)) == {(True, False)}
    assert set(x.box2bvs((-.3, .3), 2, innerapprox=True)) == set([])

    # Outer approximations tests
    assert set(x.box2bvs((.4, .6), 2, innerapprox=False)) == {(True, False)}
    assert set(x.box2bvs((.4, .6), 2, innerapprox=False)) == {(True, False)}
    assert set(x.box2bvs((.99, 1.01), 2, innerapprox=False)) == {(True, False),
                                                                 (True, True)}

    # Some numerical sensitivity tests 
    assert x.box2indexwindow((-1.0, 1.0), 2, innerapprox=True) == (1, 2)
    assert x.box2indexwindow((-1.0, 1.0-.00001), 2, innerapprox=True) == (1, 1)
    assert x.box2indexwindow((-1.0, 1.0+.00001), 2, innerapprox=True) == (1, 2)
    assert x.box2indexwindow((-1.0 - .00001, 1.0+.00001), 2, innerapprox=False) == (0, 3)

    # BDD creation
    assert x.conc2pred(mgr, "x", (.4, .6), 1, True) == mgr.false
    args = [mgr, "x", (.4, .6), 1, True]
    assert x.conc2predold(*args) == x.conc2pred(*args)
    args = [mgr, "x", (-.34, .65), 4, True]
    assert x.conc2predold(*args) == x.conc2pred(*args), x.box2indexwindow(*args[2:])
    assert x.box2indexwindow(*args[2:]) == (7, 9)
    
    for i in range(50):
        bits = np.random.randint(0,4)
        left = np.random.rand() * 4 - 2
        right = np.random.rand() * (2-left) + left
        inner = False  # if np.random.randint(2) == 1 else False
        args = [mgr, "x", (left, right), bits, inner]
        assert x.conc2predold(*args) == x.conc2pred(*args), [i, args, x.box2indexwindow(*args[2:])]

    pspace = DynamicCover(-2, 2)
    assert pspace.box2indexwindow((0, .8), 6, False) == (32, 44)

    # Over and under approximations of boxes that align exactly with the grid are the same
    assert pspace.conc2pred(mgr, 'x', (-1,1) , 4, innerapprox=False) == pspace.conc2pred(mgr, 'x', (-1,1) , 4, innerapprox=True) 
    assert pspace.conc2pred(mgr, 'x', (-1.5,1.5) , 4, innerapprox=False) == pspace.conc2pred(mgr, 'x', (-1.5,1.5) , 4, innerapprox=True) 


def test_dynamic_periodic():
    mgr = BDD()
    x = DynamicCover(0, 20, periodic=True)
    # assert x.pt2bv(11, 4) == (True, True, False, False)
    # assert x.pt2bv(19+20, 4) == (True, False, False, False)

    # Wrap around
    assert set(x.box2bvs((17, 7), 2, innerapprox=True)) == {(False, False)}
    assert set(x.box2bvs((17, 7), 2, innerapprox=False)) == {(True, False),
                                                             (False, False),
                                                             (False, True)}

    assert x.conc2pred(mgr, 'x', (1, 19), 3) == mgr.true
    # assert x.conc2pred(mgr, 'x', (19,1), 3, True) == mgr.false
    assert x.conc2pred(mgr, 'x', (0, 9.9), 3, False) == mgr.add_expr('~x_0')
    assert x.conc2pred(mgr, 'x', (0, 9.9), 3, True) == mgr.add_expr('~x_0 & ~x_1') | mgr.add_expr('~x_0 & x_1 & x_2')

    assert x.box2indexwindow((.1, 19), 3, False) == (0, 7)
    assert x.box2indexwindow((.1, 19), 3, True) == (1, 6)
    assert x.box2indexwindow((19, 1), 3, True) is None
    assert x.box2indexwindow((19, 1), 3, False) == (7, 0)
    assert x.box2indexwindow((.1, 1), 3, False) == (0, 0)
    assert x.box2indexwindow((.1, 1), 3, True) is None
    assert x.box2indexwindow((1, .1), 3, True) == (1, 7)
    assert x.box2indexwindow((9.7, 29.5), 3, True) == (4, 2)
    assert x.box2indexwindow((1.80, 21.1), 1, True) == (1, 1)
    assert x.box2indexwindow((16.2, 31), 3, True) == (7, 3)


    assert x.box2indexwindow((19.9, .1), 3, innerapprox=True) is None
    assert x.box2indexwindow((2.4, 2.6), 3, innerapprox=True) is None

    # wrap around with overapproximation
    assert x.box2indexwindow((19.9, .1), 3, innerapprox=False) == (7, 0)
    assert x.box2indexwindow((39.9, 20.1), 3, innerapprox=False) == (7, 0)
    assert x.box2indexwindow((9.9, 9.8), 3, innerapprox=False) == (4, 3)
    assert x.box2indexwindow((29.9, 29.8), 3, innerapprox=False) == (4, 3)
    assert x.box2indexwindow((29.9, 9.8), 3, innerapprox=False) == (4, 3)
    assert x.box2indexwindow((19.9, 19.8), 3, innerapprox=False) == (0, 7)  # total cover
    assert x.box2indexwindow((39.9, 39.8), 3, innerapprox=False) == (0, 7)  # total cover
    

    # Over and under approximations of boxes that align exactly with the grid are the same
    assert x.conc2pred(mgr, 'x', (5,15), 4, innerapprox=True) == x.conc2pred(mgr, 'x', (5,15), 4, innerapprox=False)
    assert x.conc2pred(mgr, 'x', (0,5), 4, innerapprox=True) == x.conc2pred(mgr, 'x', (0,5), 4, innerapprox=False)
    assert x.conc2pred(mgr, 'x', (15,5), 4, innerapprox=True) == x.conc2pred(mgr, 'x', (15,5), 4, innerapprox=False)
    assert x.conc2pred(mgr, 'x', (0,20), 4, innerapprox=True) == x.conc2pred(mgr, 'x', (20,40), 4, innerapprox=False)

    # print("======= Random tests =======")
    # for i in range(100):
    #     bits = np.random.randint(0,4)
    #     left = np.random.rand() * 20
    #     right = np.random.rand() * 20 + left
    #     inner = True # if np.random.randint(2) == 1 else False
    #     args = [mgr, "x", (left,right), bits, inner]
    #     print(i, args)
    #     print(x.box2indexwindow(*args[2:]))
    #     print(list(x.box2bvs(*args[2:])), "\n\n")
    #     assert x.conc2predold(*args) == x.conc2pred(*args)


def test_fixed_regular():

    x = FixedCover(-3, 7, 13)
    assert x.pt2index(-3) == 0

    bv_box = set(x.box2bvs((.1, 3.7)))
    bv_innerbox = set(x.box2bvs((.1, 3.7), innerapprox=True))
    assert bv_box == {(False, True, False, False),
                      (False, True, False, True),
                      (False, True, True, False),
                      (False, True, True, True),
                      (True, False, False, False)}
    assert len(bv_box) == len(bv_innerbox) + 2

    y = FixedCover(0, 10, 5)
    assert set(y.box2bvs((3, 7), False)) == {(False, False, True),
                                             (False, True, False),
                                             (False, True, True)}
    assert set(y.box2bvs((3, 7), True)) == {(False, True, False)}

    # Inner-outer tests
    assert set(y.box2bvs((3, 3.5), True)) == set([])
    assert set(y.box2bvs((3, 3.5), False)) == {(False, False, True)}
    assert set(y.box2bvs((3, 5), True)) == set([])
    assert y.box2indexwindow((0.1, 6.9), innerapprox=True) == (1, 2)


def test_fixed_periodic():

    mgr = BDD()

    y = FixedCover(0, 10, 5, periodic=True)  # 5 bins
    assert set(y.box2bvs((3, 7), False)) == {(False, False, True),
                                             (False, True, False),
                                             (False, True, True)}
    assert set(y.box2bvs((3, 7), True)) == {(False, True, False)}

    # Inner-outer tests
    assert set(y.box2bvs((3, 3.5), True)) == set([]) 
    assert set(y.box2bvs((3, 3.5), False)) == {(False, False, True)}
    assert set(y.box2bvs((3, 5), True)) == set([]) 

    # Wrap around tests 
    assert set(y.box2bvs((9, 1), True)) == set([])
    assert set(y.box2bvs((9, 2.1), True)) == {(False, False, False)}
    assert set(y.box2bvs((9, 2.1), False)) == {(True, False, False),
                                               (False, False, False),
                                               (False, False, True)}

    z = FixedCover(0, 10, 5, periodic=True)
    assert y == z
    assert z.box2indexwindow((9.9, .1), innerapprox=True) is None
    assert z.box2indexwindow((1.9, 2.1), innerapprox=True) is None
    assert z.box2indexwindow((9.9, .1), innerapprox=False) == (4, 0)
    assert z.box2indexwindow((4.4, 4.3), innerapprox=False) == (3, 2)
    assert z.box2indexwindow((9.9, 9.8), innerapprox=False) == (0, 4)
    assert z.conc2pred(mgr, 'z', (4.4, 4.3), innerapprox=False) == mgr.add_expr('~z_0 | (z_0 & ~z_1 & ~z_2)')
    assert z.box2indexwindow((19.9, 19.8), innerapprox=False) == (0, 4)

    # Over and under approximations of boxes that align exactly with the grid are the same
    assert z.conc2pred(mgr, 'z', (0,4), innerapprox=True) == z.conc2pred(mgr, 'z', (0,4), innerapprox=False)


def test_discrete():
    pass


def test_embedded_grid():
    x = EmbeddedGrid(10, 50, 21)
    assert x.num_bits == 5
    assert x.pt2index(24) == 7  # (24-10)/2
    assert x.pt2index(22.9, snap=True) == 6
    assert x.pt2index(25.1, snap=True) == 8

    mgr = BDD() 
    assert(mgr.count(x.abs_space(mgr, 'x'), 5)) == 21

    with raises(ValueError):
        x.pt2index(23)
    with raises(ValueError):
        EmbeddedGrid(40, 50, 0)
    with raises(ValueError):
        EmbeddedGrid(50, 40, 3)
    with raises(ValueError):
        EmbeddedGrid(40, 50, 1)

    # Snapping to nearest from out of range
    assert x.pt2index(9, snap=True) == 0
    assert x.pt2index(60, snap=True) == 20

    EmbeddedGrid(10, 10, 1)


def test_utils():
    # Regular intervals 
    assert set(bv_interval([False, True], [True, True])) == {(False, True), 
                                                             (True, False), 
                                                             (True, True)}
    assert set(bv_interval([True, True], [False, True])) == set([])
