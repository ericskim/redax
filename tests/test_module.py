from pytest import approx, raises 

from dd.cudd import BDD

import numpy as np 

from vpax.module import AbstractModule
from vpax.symbolicinterval import *
 

def test_dynamic_module():
    mgr = BDD() 
    
    inputs = {'x': DynamicPartition(0, 16),
              'y': DynamicPartition(0, 4),
              }
    output = {'z': DynamicPartition(0, 4)
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


def test_mixed_module():

    from dd.cudd import BDD

    import numpy as np 

    from vpax.module import AbstractModule
    from vpax.symbolicinterval import DynamicPartition, FixedPartition

    mgr = BDD() 
    inputs = {'x': DynamicPartition(0, 16),
              'y': FixedPartition(-10,10,10),
              'theta': DynamicPartition(-np.pi, np.pi, periodic=True),
              'v': FixedPartition(0,5, 5),
              'omega': FixedPartition(-2,2,4)
              }
    outputs = {'xnext': DynamicPartition(0, 4),
             'ynext': FixedPartition(-10,10,10),
             'thetanext': DynamicPartition(-np.pi, np.pi, periodic=True)
             }

    dubins = AbstractModule(mgr, inputs, outputs) 

    dubins.ioimplies2pred( {'v': (3.6,3.7), 'theta': (6,-6), 'y': (2,3), 'ynext': (2.1,3.1)}, 
                            precision = {'theta': 3}) 

    
