from pytest import approx

import numpy as np
import funcy as fn

from redax.module import Interface, CompositeInterface
from redax.spaces import DynamicCover
from redax.synthesis import SafetyGame, ControlPre, DecompCPre, ReachGame
from redax.visualizer import scatter2D, plot3D, plot3D_QT, pixel2D
from redax.utils.overapprox import bloatbox

from redax.predicates.dd import BDD
mgr = BDD()
mgr.configure(reordering=True)

ts = .2
k = .1
g = 9.8

def dynamics(p, v, a):
    vsign = 1 if v > 0 else -1
    return p + v * ts, v + a*ts - vsign*k*(v**2)*ts - g*ts

pspace = DynamicCover(-10, 10)
vspace = DynamicCover(-16, 16)
aspace = DynamicCover(0, 20)

# Smaller component modules
pcomp = Interface(mgr,
                 {'p': pspace,
                  'v': vspace},
                 {'pnext': pspace}
        )
vcomp = Interface(mgr,
                 {'v': vspace,
                  'a': aspace},
                 {'vnext': vspace}
        )

# Declare grid precision
p_precision = 7
v_precision = 7
precision = {'p': p_precision, 'v': v_precision, 'a': 7, 'pnext': p_precision, 'vnext': v_precision}
bittotal = sum(precision.values())
outorder = {0: 'pnext', 1: 'vnext'}
possible_transitions = (pcomp * vcomp).count_io_space(bittotal)

np.random.seed(1337)

for numapplied in range(600):

    # Shrink window widths over time
    width = 20 * 1/np.log10(2*numapplied+10)

    # Generate random input windows
    f_width = {'p': np.random.rand()*.5*width,
               'v': np.random.rand()*width,
               'a': np.random.rand()*.5*width}
    f_left  = {'p': -10 +  np.random.rand() * (20 - f_width['p']),
               'v': -16 + np.random.rand() * (32 - f_width['v']),
               'a': 0 + np.random.rand() * (20 - f_width['a'])}
    f_right = {k: f_width[k] + f_left[k] for k in f_width}
    iobox   = {k: (f_left[k], f_right[k]) for k in f_width}

    # Generate output overapproximation
    ll = dynamics(**f_left)
    ur = dynamics(**f_right)
    outbox = {outorder[i]: (ll[i], ur[i]) for i in range(2)}
    iobox.update(outbox)

    # Apply constraint to parallel updates
    pcomp = pcomp.io_refined({k: v for k, v in iobox.items() if k in pcomp.vars}, nbits=precision)
    vcomp = vcomp.io_refined({k: v for k, v in iobox.items() if k in vcomp.vars}, nbits=precision)


def test_safe_control():
    composite = CompositeInterface((pcomp, vcomp))
    dcpre = DecompCPre(composite, (('p', 'pnext'), ('v', 'vnext')), ('a'))

    safe = pspace.conc2pred(mgr, 'p', [-8,8], 6, innerapprox=True)
    safesink = Interface(mgr, {'p': pspace, 'v': vspace}, {},  guar = mgr.true, assum=safe)

    # Solve game and plot 2D invariant region
    game = SafetyGame(dcpre, safesink)
    dinv, _, controller = game.run()

    system = pcomp * vcomp
    cpre = ControlPre(system, (('p', 'pnext'), ('v', 'vnext')), ('a'))
    game = SafetyGame(cpre, safesink)
    inv, _, _ = game.run()

    assert dinv == inv

    assert dinv.count_nb(p_precision + v_precision) == approx(5988)

    # Simulate for initial states
    state_box = fn.first(controller.winning_states())
    assert state_box is not None
    state = {k: .5*(v[0] + v[1]) for k, v in state_box.items()}
    for step in range(30):
        u = fn.first(controller.allows(state))
        assert u is not None
        picked_u = {'a': u['a'][0]} # Pick lower bound of first allowed control voxel

        state.update(picked_u)
        nextstate = dynamics(**state)
        state = {'p': nextstate[0], 'v': nextstate[1]}



def test_reach_control():
    composite = CompositeInterface((pcomp, vcomp))
    dcpre = DecompCPre(composite, (('p', 'pnext'), ('v', 'vnext')), ('a'))

    target = pspace.conc2pred(mgr, 'p', [-2,2], 6, innerapprox=True)
    targetint = Interface(mgr, {'p': pspace, 'v': vspace}, {},  guar = mgr.true, assum=target)

    # Solve game and plot 2D invariant region
    game = ReachGame(dcpre, targetint)
    dbasin, _, _ = game.run()

    system = pcomp * vcomp
    cpre = ControlPre(system, (('p', 'pnext'), ('v', 'vnext')), ('a'))
    game = ReachGame(cpre, targetint)
    basin, _, _ = game.run()

    assert dbasin == basin

    assert dbasin.count_nb(p_precision + v_precision) == approx(5964)
