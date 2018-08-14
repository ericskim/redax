

import numpy as np
try:
    from dd.cudd import BDD
except ImportError:
    from dd.autoref import BDD
from pytest import approx, raises

from sydra.module import AbstractModule
from sydra.spaces import DynamicCover, EmbeddedGrid, FixedCover


def test_dynamic_module():
    mgr = BDD() 
    
    inputs = {'x': DynamicCover(0, 16),
              'y': DynamicCover(0, 4),
              }
    output = {'z': DynamicCover(0, 4)
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
    g = g.io_refined( {'j': (2.9,10.1), 'y': (2.4,3.8), 'r': (2.1,3.1)}, nbits = precision)
    assert g.count_io(bittotal) == approx(42) # = 7 * 2 * 3
    
    # Adding approximately the same transitions twice does nothing due to quantization 
    oldpred = g.pred
    g = g.io_refined( {'j': (3.,10.), 'y': (2.5,3.8), 'r': (2.1,3.1)}, nbits = precision)
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
        g = g.io_refined( {'j': (3.,10.), 'y': (2.5,3.8), 'r': (2.1,4.6)}, silent=False, nbits = precision)
    
def test_mixed_module():

    from sydra.module import AbstractModule
    from sydra.spaces import DynamicCover, FixedCover

    mgr = BDD() 
    inputs = {'x': DynamicCover(0, 16),
              'y': FixedCover(-10, 10, 10),
              'theta': DynamicCover(-np.pi, np.pi, periodic=True),
              'v': FixedCover(0, 5, 5),
              'omega': FixedCover(-2, 2, 4)
              }
    outputs = {'xnext': DynamicCover(0, 4),
             'ynext': FixedCover(-10,10,10),
             'thetanext': DynamicCover(-np.pi, np.pi, periodic=True)
             }
    
    dubins = AbstractModule(mgr, inputs, outputs) 

    # Underspecified input-output
    with raises(AssertionError):
        dubins.io_refined( {'v': (3.6,3.7), 'theta': (6,-6), 'y': (2,3), 'ynext': (2.1,3.1)}, 
                                nbits = {'theta': 3}) 

    # Test that fixed covers yield correct space cardinality
    assert mgr.count(dubins.inspace(), 4+4+4+3+2) == 16 * 10 * 16 * 5 * 4
    assert mgr.count(dubins.outspace(), 2 + 4 + 4) == 4 * 10 * 16

    
def test_embeddedgrid_module():

    from sydra.module import AbstractModule
    from sydra.spaces import EmbeddedGrid

    mgr = BDD() 
    inputs = {'x': EmbeddedGrid(0,3,4)}
    outputs = {'y': EmbeddedGrid(4,11,8)}

    m = AbstractModule(mgr, inputs, outputs)

    assert m.io_refined({'x': 2, 'y':4}).pred == mgr.add_expr(r"( x_0 /\ ~x_1)") & mgr.add_expr(r" ~( x_0 /\ ~x_1) | (~y_0 /\ ~y_1 /\ ~y_2)")
    
    assert len(mgr.vars) > 0 

def test_refinement_and_coarsening(): 

    mgr = BDD()
    from sydra.module import AbstractModule
    from sydra.spaces import DynamicCover

    def conc(x):
        return -3*x

    x = DynamicCover(-10, 10)
    y = DynamicCover(20, 20)

    mod = AbstractModule(mgr, {'x': x}, {'y':y})

    width = 15

    for _ in range(50):
        # Generate random input windows
        f_width = {'x': np.random.rand()*width}
        f_left  = {'x': -10 +  np.random.rand() * (20 - f_width['x'])}
        f_right = {k: f_width[k] + f_left[k] for k in f_width}
        iobox   = {k: (f_left[k], f_right[k]) for k in f_width}

        # Generate output overapproximation
        ur = conc(**f_left)
        ll = conc(**f_right)
        iobox['y'] = (ll, ur)

        # Refine and check abstract relation
        newmod = mod.io_refined(iobox, nbits = {'x': 8, 'y':8})
        assert mod <= newmod
        mod = newmod

        # Check abstract relation relative to coarsened module
        assert mod.coarsen(x=5,y=5) <= mod
         # Coarsen should do nothing because it keeps many bits
        assert mod.coarsen({'x':10},y=10) == mod


# def test_identity_module():
#     from sydra.module import AbstractModule
