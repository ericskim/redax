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
    assert g.nonblock == (g.hide(g.outputs).pred)

    # Identity test for input and output renaming 
    assert ((g  >> ('r', 'z') ) >> ('z','r')) == g
    assert (('j','w') >> (('w','j') >> g) ) == g

    # Parallel composition 
    assert set((g | h).outputs) == {'z','r'}
    assert set((g | h).inputs) == {'x','y','j'}

    # Series composition with disjoint I/O yields parallel composition 
    assert (g >> h) == (g | h) 

    