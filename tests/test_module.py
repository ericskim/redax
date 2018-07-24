from pytest import approx, raises 

from dd.cudd import BDD

import numpy as np 

from vpax.module import AbstractModule
from vpax.symbolicinterval import *

"""
pstate = DynamicInterval(-10, 10)
vstate = DynamicInterval(-20,20)
inputs = {'p': pstate,
          'v': vstate,
          'a': DynamicInterval(-20,20)}
outputs = {'pnext': pstate,
           'vnext': vstate}

system = AbstractModule(mgr, inputs, outputs)
grid_iterator = system.inputiterator(precision = {'p': 3, 'v': 3, 'a': 3})
system.set_precision(precision) <--- assigns a default precision to be remembered for the dynamic intervals 
for i in grid_iterator:
    out_OA = OA(i) 
    system.apply_io_constraint(i,out_OA, precision)



for i in system.inputiterator(precision = {'p': 4, 'v': 2, 'a': 3}):
    out_OA = OA(i) 
    system.apply_io_constraint(i,out_OA, precision)

# Hides an output 
system.hide('pnext') 

compositesys = sys1 | sys2
compsys = sys1 >> ('x', 'y') >> sys2

#pupdate.register_concrete(func, hooks)?

"""

def test_module():
    mgr = BDD() 
    
    inputs = {'x': DynamicInterval(0, 16),
              'y': DynamicInterval(0, 4),
              }
    output = {'z': DynamicInterval(0, 4)
             }

    h = AbstractModule(mgr, inputs, output)

    # Rename inputs 
    assert set(h.inputs) == {'x','y'} 
    assert set(h.outputs) == {'z'}
    g = ('j', 'x') >> h >> ('z', 'r')
    assert set(g.inputs) == {'j','y'}
    assert set(g.outputs) == {'r'}

    precision = {'j': 4, 'y': 3, 'r': 3}
    bittotal = sum(precision.values()) 
    
    assert g.count_io(bittotal) == approx(1024)
    g.pred &= g.ioimplies2pred( {'j': (3.,10.), 'y': (2.5,3.8), 'r': (2.1,3.1)}, precision = precision)
    assert g.count_io(bittotal) == approx(954) 

    # Adding same transitions twice does nothing 
    oldpred = g.pred
    g.pred &= g.ioimplies2pred( {'j': (3.,10.), 'y': (2.5,3.8), 'r': (2.1,3.1)}, precision = precision)
    assert g.pred == oldpred 


    assert (g).pred.support == {'j_0', 'j_1', 'j_2', 'j_3',
                                    'y_0', 'y_1', 'y_2',
                                    'r_0', 'r_1', 'r_2'}
    assert (g  >> ('r', 'z') ).pred.support == {'j_0', 'j_1', 'j_2', 'j_3',
                                                     'y_0', 'y_1', 'y_2',
                                                     'z_0', 'z_1', 'z_2'}

    assert g.nonblock == mgr.true # No inputs block

    # Identity test for input and output renaming 
    assert ((g  >> ('r', 'z') ) >> ('z','r')) == g
    assert (('j','w') >> (('w','j') >> g) ) == g

    # Parallel composition 
    assert set((g | h).outputs) == {'z','r'}
    assert set((g | h).inputs) == {'x','y','j'}

    # Series composition with disjoint I/O yields parallel composition 
    assert (g >> h) == (g | h) 

    
# def test_module():
#     mgr = BDD()

#     @input(mgr, 'x', [0,20], dynamic = False, precision = 3)
#     @input(mgr, 'y', [0,4], precision = 3)
#     @input(mgr, 'z', [0,1], precision = 2, periodic = True) 
#     @input(mgr, 'w', [0,1], dynamic = False, precision = 5, periodic = True)
#     @output(mgr, (0, 'r'), [0,4])
#     def f(x: float, y, z:float, w:float) -> (float):
#         return x + y + z + w + 3

#     @input(mgr, 'x', [0,16], dynamic = False, precision = 16)
#     @input(mgr, 'y', [0,4], precision = 3)
#     @output(mgr, (0, 's'), [0,4], precision = 3)
#     def g(x: float, y) -> (float):
#         return x + y  + 3
    
#     def conc(x):
#         return list(mgr.pick_iter(x)) 

#     f_in = {'x': np.random.rand()*20, 'y':np.random.rand()*4, 'z':np.random.rand(), 'w': np.random.rand()}
#     f_left = {'x': np.random.rand() * (20 - f_in['x']), 
#             'y': np.random.rand() * (4 - f_in['y']), 
#             'z': np.random.rand() * (1 - f_in['z']),
#             'w': np.random.rand() * (1 - f_in['w']) }
#     f_box = {k: (f_left[k], f_in[k] + f_left[k]) for k in f_in}

#     inorder = g['x'].bitorder + g['y'].bitorder
#     for i in conc(g.input_box_to_bdd({'x': (3,10), 'y': (2.5,3.8)})):
#         print("Input: ", [(k, i[k])for k in inorder])
#     for i in conc(g.output_box_to_bdd([(2.1,3.1)])):
#         print("Output: ", i)

#     assert g.count_io() == approx(1024)
#     g.apply_io_constraint( ({'x': (3,10), 'y': (2.5,3.8)}, {0: (2.1,3.1)}) )
#     assert g.count_io() == approx(954)

#     assert f.check() == True
#     assert g.check() == True 