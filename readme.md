[![Build Status](https://travis-ci.org/ericskim/redax.svg?branch=master)](https://travis-ci.org/ericskim/redax)

# **REDAX: Robust Control Synthesis with Finite Abstractions**

REDAX is a controller synthesis tool that constructs finite state machines that mimic continuous dynamics. We call the finite state machines *abstractions* because they abstract out low level state information.

## Table of Contents

- [About](#About)
- [Installation](#installation)
- [Distinguishing Features](#distinguishing-features)
- [Development Notes](#development-notes)
- [References](#references)

## About

Supports:

- Nonlinear Discrete-Time Control Systems
- Safety, Reachability, and Reach-Avoid Objectives
- Requires a method to overapproximate the set of open-loop reachable states from a hyperrectangle of initial states to ensure the outputted controller is sound and correct, but the tool can still yield useful results without it.

Ideally for:

- Systems with complex constraints and nonlinear dynamics.
- Objectives and systems that are of interest in a bounded set.
- Controller synthesis problems approached by gridding a continuous space such as in dynamic programming.
- Systems of moderate state dimensions (4-6).

## Installation

Clone this repo and run the following commands:

```shellscript
cd /location/of/redax/here/
pip install -r requirements
pip install .
```

### Dependencies

- Primary
  - python 3.6
  - cudd with [dd](https://github.com/tulip-control/dd) wrapper
  - numpy
  - bidict
  - funcy
  - toposort
  - dataclasses
- Secondary
  - Visualization:
    - matplotlib
    - pyqtgraph
  - Testing:
    - pytest

## Distinguishing Features

- **Extensible Abstraction Spaces**

  Abstracting a set means grouping together points and constructing a lower cardinality set. We support a number of methods such as:

  - Fixed vs. dynamic precision covers: Union of rectangles where each one is uniquely identified by a bitvector. Throwing away lower precision bits implicitly makes the cover more coarse. Adding additional bits implies higher precision.
  - Linear vs. periodic covers: Use the traditional binary encoding to encode hyperrectangles in the grid. Periodic grids use [gray code](https://en.wikipedia.org/wiki/Gray_code).
  - Embedded grids: A grid of discrete points embedded in continuous space, like the integers.

  ```python
  # Declare continuous state spaces for a Dubins vehicle
  pspace      = DynamicCover(-2,2)
  anglespace  = DynamicCover(-np.pi, np.pi, periodic=True)
  # Declare discrete values for control inputs
  vspace      = EmbeddedGrid(vmax/2, vmax, 2)
  angaccspace = EmbeddedGrid(-1.5, 1.5, 3)
  ```

- **Small Reusable Modules**

  Interfaces are like functions, but have named inputs/outputs and may exhibit non-deterministic outputs or block for some inputs.

  ```python
  # Declare modules. mgr is a BDD manager from dd.
  dubins_x      = Interface(mgr,
                            inputs={'x': pspace, 'theta': anglespace, 'v': vspace},
                            outputs={'xnext': pspace}
                  )

  dubins_y      = Interface(mgr,
                            inputs={'y': pspace, 'theta': anglespace, 'v': vspace},
                            outputs={'ynext': pspace}
                  )

  dubins_theta  = Interface(mgr,
                            inputs={'theta': anglespace, 'v': vspace, 'omega': angaccspace},
                            outputs={'thetanext': anglespace}
                  )
  ```

- **Manipulate and Compose Interfaces**

  Abstract systems can be manipulated with a collection of operations:

  ```python
  from redax.ops import rename, ohide, coarsen, rename

  # Parallel composition is useful for constructing larger dimensional systems.
  dubins = dubins_x * dubins_y * dubins_theta # Parallel composition shorthand
  assert dubins == compose(dubins_x, compose(dubins_y, dubins_theta))

  # Series composition is a robust version of function composition m2(m1(...)).
  # Composition accounts m1's output non-determinism and m2's blocking inputs.
  m1 = Interface(mgr, {'x': DummySpace1}, {'y': DummySpace2})
  m2 = Interface(mgr, {'y': DummySpace2}, {'z': DummySpace3})
  m12 = compose(m1, m2)
  assert m12 == m1 >> m2 # Series composition shorthand

  # Input and output variable renaming.
  dubins_renamed = rename(dubins_x, {'xnext': 'xposnext', 'x': 'xpos'})

  # Hide outputs from interfaces.
  x = ohide(dubins_x, {'xnext'})

  # Future feature: Compute lower complexity abstractions keeping only the most significant bits.
  coarser_model = coarsen(model, x=3, y=4)  # Retain 3 bits for 'x' variable and 4 bits for 'y' variable.
  ```

- **Refinement checks**

  A built in notion of one interface refining another or vice versa where an interface is an abstraction of another.

  ```python
  # Coarser model abstracts the original model
  coarser_model = coarsen(model, x=3, y=4)
  assert coarser_model <= model

  # Variable hiding, renaming, and interface compose operations preserve refinement order
  assert ohide('y', coarser_model) <= ohide('y', model)
  assert rename(coarser_model, {'y': 'z'}) <= rename(model, {'y': 'z'})
  assert coarsen(dubins_x, ynext=4) * dubins_y * dubins_theta <= dubins_x * dubins_y * dubins_theta
  ```

- **Flexible Abstraction Construction**

  Abstractions of modules are constructed by providing the following characterization of input-output pairs

  - [Input set] -> [Overapproximation of the set of possible outputs]

  The samples do NOT need disjoint input sets or have input-output sets align with a grid. The controller synthesis results get better as sample quantity increases and the overapproximation tightens.

  ```python
  # Collections of input-output pairs can be generated any number of ways!
  collection = database/file or iterator or odesolver/simulator or random input generator

  # Anytime algorithm for abstraction construction
  refined_interface = initial_interface
  for io_pair in collection:
    refined_interface = refined_interface.io_refined(io_pair)
    assert initial_interface <= refined_interface

  # Refinement operations can be chained together
  abstraction.io_refined(io_pair_1).io_refined(io_pair_2)
  ```

- **Extensible control synthesizers**

  Use multiple built-in control predecessors constructed using the above operators to solve safety, reach, and reach-avoid games.

  ```python
  from redax.synthesis import ReachGame, DecompCPre, ControlPre

  target = get_target_interface()

  # Monolithic Control Predecessor
  dubins = dubins_x * dubins_y * dubins_theta # Construct monolithic representation
  cpre = ControlPre(dubins,
                    states=(('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')),
                    control=('v', 'omega')
                   )
  game = ReachGame(cpre, target)
  reach_basin, steps, controller = game.run()

  # Decomposed Control Predecessor exploits decomposed structure and is faster
  composite = CompositeInterface([dubins_x, dubins_y, dubins_theta])
  dcpre = DecompCPre(composite,
                     states=(('x', 'xnext'), ('y', 'ynext'), ('theta', 'thetanext')),
                     control=('v', 'omega')
                    )
  dgame = ReachGame(dcpre, target)
  dreach_basin, steps, controller = dgame.run()

  assert reach_basin == dreach_basin
  ```

- **Swappable Symbolic Backend**

  We build on top of libraries for symbolically representing finite sets and relations. The current implementation uses binary decision diagrams via the dd package. Support for aiger circuits is possible with the [py-aiger](https://github.com/mvcisback/py-aiger) library.

## Development Notes

### Status

The core foundations is working and providing results on some meaningful test examples but the APIs are fluctuating rapidly.

### TODOs

- Scale up lunar lander
  - Fix issues with unnecessary blocking with position modules dependencies on angle.
  - Iterator for shifted but bounded space.
- Make a variable = name + types object
- Controlled predecessor with multiple time scale models and control inputs.
- Different iterators of input space
  - Iterators for the $2^{N-1}$ reduced grid traversal
- Rewrite continuous cover grid to have an overlap parameter
- Make the BDD manager a class attribute instead of instance attribute?

### Future Features

- Support for disjoint union and product operations for spaces (thus adding support for hybrid spaces)
  - Requires a predicate bit name generator
  - Each space should generate a higher order function quantizer.

## References

### Literature

- [*Abstractions for Symbolic Controller Synthesis are Composable*, Eric S. Kim, Murat Arcak](https://arxiv.org/abs/1807.09973)

### Similar tools for abstraction-based controller synthesis

- [SCOTS](https://gitlab.lrz.de/hcs/scots)
  - [MASCOT](https://github.com/hsukyle/mascot)
- [PESSOA](https://sites.google.com/a/cyphylab.ee.ucla.edu/pessoa/)
- [ARCS](https://github.com/pettni/abstr-refinement/)
- [ROCS](https://git.uwaterloo.ca/hybrid-systems-lab/rocs)
