import numpy as np
from dd.cudd import BDD
from pytest import approx, raises

from vpax.module import AbstractModule
from vpax.spaces import DynamicPartition, EmbeddedGrid, FixedPartition


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
    
    assert g.count_io_space(bittotal) == approx(1024)
    assert g.count_io(bittotal) == approx(0)
    g.apply_abstract_transitions( {'j': (2.9,10.1), 'y': (2.4,3.8), 'r': (2.1,3.1)}, nbits = precision)
    assert g.count_io(bittotal) == approx(42) # = 7 * 2 * 3
    
    # Adding approximately the same transitions twice does nothing due to quantization 
    oldpred = g.pred
    g.apply_abstract_transitions( {'j': (3.,10.), 'y': (2.5,3.8), 'r': (2.1,3.1)}, nbits = precision)
    assert g.pred == oldpred 

    assert (g).pred.support == {'j_0', 'j_1', 'j_2', 'j_3',
                                    'y_0', 'y_1', 'y_2',
                                    'r_0', 'r_1', 'r_2'}
    assert (g  >> ('r', 'z') ).pred.support == {'j_0', 'j_1', 'j_2', 'j_3',
                                                     'y_0', 'y_1', 'y_2',
                                                     'z_0', 'z_1', 'z_2'}

    assert g.nonblock() == g.concrete_input_to_abs({'j': (3.,10.), 'y': (2.5,3.8)}, nbits = precision) # No inputs block
    assert g.nonblock() == (g.hide(g.outputs).pred) 

    # Identity test for input and output renaming 
    assert ((g  >> ('r', 'z') ) >> ('z','r')) == g
    assert (('j','w') >> (('w','j') >> g) ) == g

    # Parallel composition 
    assert set((g | h).outputs) == {'z','r'}
    assert set((g | h).inputs) == {'x','y','j'}

    # Series composition with disjoint I/O yields parallel composition 
    assert (g >> h) == (g | h) 

    # Out of bounds errors 
    with raises(AssertionError):
        g.pred &= g.ioimplies2pred( {'j': (3.,10.), 'y': (2.5,3.8), 'r': (2.1,4.6)}, nbits = precision)
    
def test_mixed_module():

    from dd.cudd import BDD

    from vpax.module import AbstractModule
    from vpax.spaces import DynamicPartition, FixedPartition

    mgr = BDD() 
    inputs = {'x': DynamicPartition(0, 16),
              'y': FixedPartition(-10, 10, 10),
              'theta': DynamicPartition(-np.pi, np.pi, periodic=True),
              'v': FixedPartition(0, 5, 5),
              'omega': FixedPartition(-2, 2, 4)
              }
    outputs = {'xnext': DynamicPartition(0, 4),
             'ynext': FixedPartition(-10,10,10),
             'thetanext': DynamicPartition(-np.pi, np.pi, periodic=True)
             }
    
    dubins = AbstractModule(mgr, inputs, outputs) 

    dubins.ioimplies2pred( {'v': (3.6,3.7), 'theta': (6,-6), 'y': (2,3), 'ynext': (2.1,3.1)}, 
                            nbits = {'theta': 3}) 

    # Test that fixed partitions yield correct space cardinality
    assert mgr.count(dubins.inspace(), 4+4+4+3+2) == 16 * 10 * 16 * 5 * 4
    assert mgr.count(dubins.outspace(), 2 + 4 + 4) == 4 * 10 * 16

    
def test_embeddedgrid_module():
    from dd.cudd import BDD

    from vpax.module import AbstractModule
    from vpax.spaces import EmbeddedGrid

    mgr = BDD() 
    inputs = {'x': EmbeddedGrid(0,3,4)}
    outputs = {'y': EmbeddedGrid(4,11,8)}

    m = AbstractModule(mgr, inputs, outputs)

    assert m.ioimplies2pred({'x': 2, 'y':4}) == mgr.add_expr(" ~( x_0 /\ ~x_1) | (~y_0 /\ ~y_1 /\ ~y_2)")
    
    assert len(mgr.vars) > 0 
