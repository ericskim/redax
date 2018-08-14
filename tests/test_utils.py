from pytest import approx, raises

from  sydra.utils import bvwindow, bvwindowgray

def test_bvwindow():
    assert set(bvwindow(0,3,3)) == set([(False,)])
    assert set(bvwindow(4,7,3)) == set([(True,)])
    assert set(bvwindow(0,31, 5)) == set([(True,), (False,)])
    assert set(bvwindow(3,4,3)) == set([(True,False,False), (False, True, True)])

    assert set(bvwindow(0,1,3)) == set([(False,False)])

def test_bvwindowgray():
    assert set(bvwindowgray(0,3,3)) == set([(False,)])
    assert set(bvwindowgray(3,12,4)) == set([(False, True), # 4-7
                                         (True, True), #8-11
                                         (False, False, True, False), #3
                                         (True, False, True, False) ] ) #12

    # Wrap arounds
    assert set(bvwindowgray(14, 1, 4)) == set([ (True, False, False), # 14-15
                                                (False, False, False)]) # 0-2
    assert set(bvwindowgray(4, 3, 4)) == set([(True,), # 8-15
                                              (False, True), # 4-7
                                              (False, False)]) # 0-3

    assert set(bvwindowgray(7, 0, 3)) == set([(True, False, False), # 7
                                              (False, False, False)]) # 0
    assert set(bvwindowgray(0, 7, 3)) == set([(True,), (False,)])
