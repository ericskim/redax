import math

import numpy as np
try:
    from dd.cudd import BDD
except ImportError:
    from dd.autoref import BDD
from pytest import approx, raises

from redax.module import AbstractModule, CompositeModule
from redax.spaces import DynamicCover, EmbeddedGrid, FixedCover, OutOfDomainError, ContinuousCover


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
    assert g == h.renamed(x='j', z='r')

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

    assert g.nonblock() == g.input_to_abs({'j': (3.,10.), 'y': (2.5,3.8)}, nbits = precision) # No inputs block
    assert g.nonblock() == (g.hidden(g.outputs).pred) 

    # Identity test for input and output renaming 
    assert ((g  >> ('r', 'z') ) >> ('z','r')) == g
    assert (('j','w') >> (('w','j') >> g) ) == g

    # Parallel composition 
    assert set((g | h).outputs) == {'z','r'}
    assert set((g | h).inputs) == {'x','y','j'}

    # Series composition with disjoint I/O yields parallel composition
    assert (g >> h) == (g | h)
    assert (g >> h) == g.composed_with(h)
    assert (g >> h) == h.composed_with(g)
    assert (g | h) == h.composed_with(g)

    # Out of bounds errors 
    with raises(OutOfDomainError):
        g = g.io_refined( {'j': (3.,10.), 'y': (2.5,3.8), 'r': (2.1,4.6)}, silent=False, nbits = precision)

def test_series_comp():

    mgr = BDD() 
    
    inputs = {'x': DynamicCover(0, 4),
              'y': DynamicCover(0, 4),
              }
    output = {'z': DynamicCover(0, 4)
             }

    h = AbstractModule(mgr, inputs, output)
    g = ('j', 'y') >> h >> ('z', 'r')
    precision = {'j': 4, 'x': 3, 'r': 3}
    g = g.io_refined( {'j': (.75,2.5), 'x': (2.5,3.8), 'r': (2.1,3.1)}, nbits = precision)

    h = ('r', 'x') >> h
    h = h.io_refined({'r': (1.3, 3.8), 'y': (1.0, 2.0), 'z': (.9, 3.1)}, nbits = {'r': 3, 'y': 3, 'z': 4})
    assert (g >> h) == g.composed_with(h)
    assert (g >> h) == h.composed_with(g)
    assert (g >> h).nonblock() != mgr.false
    assert (g >> h).count_nb(bits=10) == approx(28)  # 7 * 2 * 2

def test_mixed_module():

    from redax.module import AbstractModule
    from redax.spaces import DynamicCover, FixedCover

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

    mgr = BDD() 
    inputs = {'x': EmbeddedGrid(4, 0, 3)}
    outputs = {'y': EmbeddedGrid(8, 4, 11)}

    m = AbstractModule(mgr, inputs, outputs)

    assert m.io_refined({'x': 2, 'y':4}).pred == mgr.add_expr(r"( x_0 /\ ~x_1)") & mgr.add_expr(r" ~( x_0 /\ ~x_1) | (~y_0 /\ ~y_1 /\ ~y_2)")
    
    assert len(mgr.vars) > 0 

def test_module_composition():
    mgr = BDD()

    x = DynamicCover(0,10)

    m1 = AbstractModule(mgr, {'a': x}, {'b': x, 'c': x})
    m2 = AbstractModule(mgr, {'i': x, 'j': x}, {'k': x})

    m12 = (m1 >> m2.renamed(i = 'c'))
    assert set(m12.inputs) == set(['a', 'j'])
    assert set(m12.outputs) == set(['c', 'b', 'k'])
    assert m12 == m2.renamed(i = 'c').composed_with(m1)
    assert m12 == (('c', 'i') >> m2 ).composed_with(m1)

    # Renaming is left associative
    assert m12 == m1 >> (('c', 'i') >> m2 )
    assert m12 != (m1 >> ('c', 'i')) >> m2
    assert m12 == ((m1.renamed(c = 'i') >> m2)).renamed(i = 'c')
    assert m12 == ((m1 >> ('c', 'i') >> m2)).renamed(i = 'c')
    assert m12 == m1.renamed(c = 'i').composed_with(m2).renamed(i = 'c')
    assert m12 == m1.renamed(c = 'i').composed_with(m2) >> ('i', 'c')

def test_refinement_and_coarsening(): 

    mgr = BDD()

    def conc(x):
        return -3*x

    x = DynamicCover(-10, 10)
    y = DynamicCover(20, 20)

    linmod = AbstractModule(mgr, {'x': x}, {'y':y})

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
        newmod = linmod.io_refined(iobox, nbits = {'x': 8, 'y':8})
        assert linmod <= newmod
        linmod = newmod

        # Check abstract relation relative to coarsened module
        assert linmod.coarsened(x=5,y=5) <= linmod
        assert linmod.coarsened(x=5) <= linmod
        assert linmod.coarsened(y=5) <= linmod        
         # Coarsen should do nothing because it keeps many bits
        assert linmod.coarsened({'x':10},y=10) == linmod

def test_sin_sqrt_comp():

    def containszero(left, right):
        """Determine if 0 is contained in a periodic interval [left,right]. Possible that right < left."""
        # Map to interval [-pi,pi]
        left = ((left + np.pi) % (np.pi*2) ) - np.pi
        left = left - 2*np.pi if left > np.pi else left
        right = ((right + np.pi) % (np.pi*2) ) - np.pi
        right = right - 2*np.pi if right > np.pi else right

        if right < left:
            if left <= 0:
                return True
        else:
            if left <= 0 and right >= 0:
                return True

        return False

    def maxmincos(left, right):
        """Compute the maximum and minimum values of cos in an interval."""
        if containszero(left, right) is True:
            maxval = 1
        else:
            maxval = max([np.cos(left), np.cos(right)])

        if containszero(left + np.pi, right + np.pi) is True:
            minval = -1
        else:
            minval = min([np.cos(left), np.cos(right)])

        return (minval, maxval)

    def maxminsin(left, right):
        """Compute the maximum and minimum values of sin in an interval."""
        return maxmincos(left - np.pi/2, right - np.pi/2)

    mgr = BDD()

    sinout = DynamicCover(-1.2,1.2)
    sinin = DynamicCover(-2*np.pi, 2*np.pi, periodic=True)

    # Sin module 
    sinmod = AbstractModule(mgr, {'sin': sinin}, {'sout': sinout})

    # Sqrt module
    sqrtout = DynamicCover(0, 1.2)
    sqrtmod = AbstractModule(mgr, {'sout': sinout}, {'sqrt': sqrtout})

    comp = CompositeModule([sinmod, sqrtmod])

    def random_input_gen(module: AbstractModule, scale: float) -> dict:
        iobox = dict()
        for invar, space in module.inputs.items():
            if isinstance(space, ContinuousCover):
                width = np.random.rand() * scale * space.width()
                if space.periodic:
                    left = space.lb + np.random.rand() * space.width() 
                else:
                    left = space.lb + np.random.rand() * (space.width() - width)
                right = left + width
                iobox.update({invar: (left, right)})
        return iobox

    precision = {'sin': 8, 'sout': 8, 'sqrt': 8}

    # Learn sin module
    for _ in range(200):
        iobox = random_input_gen(sinmod, scale = .1)
        out = maxminsin(iobox['sin'][0], iobox['sin'][1])
        iobox.update({'sout': (out[0], out[1])})

        # No errors should be raised
        sinmod = sinmod.io_refined(iobox, silent=False, nbits=precision)
        comp = comp.io_refined(iobox, nbits=precision)
        assert sinmod == comp.children[0]

    # Learn sqrt module
    for _ in range(200):
        iobox = random_input_gen(sqrtmod, scale = .2)
        if iobox['sout'][0] < 0 or iobox['sout'][1] < 0:
            continue
        out = (math.sqrt(iobox['sout'][0]), math.sqrt(iobox['sout'][1]))
        iobox.update({'sqrt': out})
        sqrtmod = sqrtmod.io_refined(iobox, silent=True, nbits=precision)
        comp = comp.io_refined(iobox, nbits=precision)
        assert sqrtmod == comp.children[1]

    # sinroot = (sinmod >> sqrtmod).hidden(['sout'])
    sinroot = (comp.children[0] >> comp.children[1]).hidden(['sout'])
    assert set(sinroot.vars) == {'sin', 'sqrt'}
    assert sinroot.pred != mgr.false
    sinroot.check() 

    # from redax.visualizer import plot2D
    # plot2D(mgr, ('sin', sinin), ('sout', sqrtout), comp.children[0].pred)
    # plot2D(mgr, ('sin', sinin), ('sout', sqrtout), comp.children[0].coarsened(sout = 6, sin = 6) .pred)
    # plot2D(mgr, ('sout', sqrtout), ('sqrt', sqrtout), comp.children[1].pred)
    # plot2D(mgr, ('sin', sinin), ('sqrt', sqrtout), sinroot.pred)

def test_composite_module_topology():

    mgr = BDD()

    x = DynamicCover(0,10)

    m1 = AbstractModule(mgr, {'a': x}, {'b': x, 'i': x})
    m2 = AbstractModule(mgr, {'i': x, 'j': x}, {'k': x})


    m12 = CompositeModule([m1, m2])
    assert m12.sorted_mods() == ((m1,), (m2,))
    assert set(m12.outputs) == {'k', 'b', 'i'}
    assert set(m12.inputs) == {'a', 'j'}
    assert set(m12.latent) == {'i'}

    m3 = AbstractModule(mgr, {'k': x, 'b': x}, {})

    m123 = CompositeModule([m1,m2,m3])
    assert m123.sorted_mods() == ((m1,), (m2,), (m3,))
    assert set(m123.outputs) == {'b','i','k'}

