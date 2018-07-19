from pytest import approx, raises 

from dd.cudd import BDD

import numpy as np 

from vpax.module import input, output, AbstractModule



def test_input_decorator():
    pass 

def test_output_decorator():
    pass

def test_module():
    mgr = BDD()

    @input(mgr, 'x', [0,20], dynamic = False, precision = 3)
    @input(mgr, 'y', [0,4], precision = 3)
    @input(mgr, 'z', [0,1], precision = 2, periodic = True) 
    @input(mgr, 'w', [0,1], dynamic = False, precision = 5, periodic = True)
    @output(mgr, (0, 'r'), [0,4])
    def f(x: float, y, z:float, w:float) -> (float):
        return x + y + z + w + 3

    @input(mgr, 'x', [0,16], dynamic = False, precision = 16)
    @input(mgr, 'y', [0,4], precision = 3)
    @output(mgr, (0, 's'), [0,4], precision = 3)
    def g(x: float, y) -> (float):
        return x + y  + 3
    
    def conc(x):
        return list(mgr.pick_iter(x)) 

    f_in = {'x': np.random.rand()*20, 'y':np.random.rand()*4, 'z':np.random.rand(), 'w': np.random.rand()}
    f_left = {'x': np.random.rand() * (20 - f_in['x']), 
            'y': np.random.rand() * (4 - f_in['y']), 
            'z': np.random.rand() * (1 - f_in['z']),
            'w': np.random.rand() * (1 - f_in['w']) }
    f_box = {k: (f_left[k], f_in[k] + f_left[k]) for k in f_in}

    inorder = g['x'].bitorder + g['y'].bitorder
    for i in conc(g.input_box_to_bdd({'x': (3,10), 'y': (2.5,3.8)})):
        print("Input: ", [(k, i[k])for k in inorder])
    for i in conc(g.output_box_to_bdd([(2.1,3.1)])):
        print("Output: ", i)

    assert g.count_io() == approx(1024)
    g.apply_io_constraint( ({'x': (3,10), 'y': (2.5,3.8)}, {0: (2.1,3.1)}) )
    assert g.count_io() == approx(954)

    assert f.check() == True
    assert g.check() == True 

def test_renaming():
    pass 

def test_composition():
    pass