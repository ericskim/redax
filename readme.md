[![Build Status](https://travis-ci.org/ericskim/redax.svg?branch=master)](https://travis-ci.org/ericskim/redax)

# REDAX: Control Synthesizer in Python with (Dynamic|Declarative|Robust) Abstractions

REDAX is a controller synthesis tool that constructs finite state machines that mimic continuous dynamics. We call the finite state machines *abstractions* because they abstract out low level state information.

## Table of Contents

- [About](#About)
- [Installation](#installation)
- [Features](#distinguishing-features)
- [Notes](#notes)
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

- Core
  - python 3.6
  - cudd with dd wrapper
  - numpy
  - bidict
- Secondary
  - Visualization:
    - matplotlib
    - pyqtgraph
  - Testing and Documentation:
    - pytest
    - sphinx

## Distinguishing Features

- **Extensible Abstraction Spaces**

  Abstracting a set means grouping together points and constructing a lower cardinality set. We support a number of methods such as:

  - Fixed vs. dynamic precision covers: Union of rectangles where each one is uniquely identified by a bitvector. Throwing away lower precision bits implicitly makes the cover more coarse. Adding addtional bits implies higher precision.
  - Linear vs. periodic covers: Use the traditional binary encoding to identify hyperrectangles in the grid. Periodic grids use [gray code](https://en.wikipedia.org/wiki/Gray_code).
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

  Modules represent stateless input-output maps but can have non-deterministic outputs and may block for some inputs.

  ```python
  # Declare modules. mgr is a BDD manager from dd.
  dubins_x        = Interface(mgr, {'x': pspace, 'theta': anglespace, 'v': vspace},
                                        {'xnext': pspace})
  dubins_y        = Interface(mgr, {'y': pspace, 'theta': anglespace, 'v': vspace},
                                        {'ynext': pspace})
  dubins_theta    = Interface(mgr, {'theta': anglespace, 'v': vspace, 'omega': angaccspace},
                                        {'thetanext': anglespace})
  ```

- **Flexible Abstraction Construction**

  Abstractions of modules are constructed by providing the following characterization of input-output pairs

  - [Input set] -> [Overapproximation of the set of possible outputs]

  The samples do NOT need disjoint input sets or have input-output sets align with a grid. The controller synthesis results get better as sample quality and quantity increase.

  ```python
  # Collections of input-output pairs can be generated any number of ways!
  collection = database/file or iterator or odesolver/simulator or random input generator

  # Anytime algorithm for abstraction construction
  for io_pair in collection:
    abstraction = abstraction.io_refined(io_pair)

  # Refinement operations can be chained together
  abstraction.io_refined(io_pair_1).io_refined(io_pair_2)
  ```

- **Manipulate and Compose Interfaces**

  Abstract systems can be manipulated with a collection of operations:

  ```python
  # Parallel composition is useful for constructing larger dimensional systems.
  dubins = dubins_x * dubins_y * dubins_theta

  # Variable Renaming.
  dubins = dubins >> ('xnext', 'newname')
  dubins = ('z', 'x') >> dubins

  # Series composition generalizes function composition m2(m1(.)), except with named variables.
  m12 = m1 >> m2

  # Hide outputs from interfaces.
  x = interface.ohidden('x')

  # Future feature: Compute lower complexity abstractions keeping only the most significant bits.
  coarser_model = model.coarsened(x=3, y=4)  # Retain 3 bits for 'x' variable and 4 bits for 'y' variable.
  ```

  Operations can also be chained together:

  ```python
  simpler_dubins = dubins.ohidden('xnext')\
                         .coarsened(x=5,y=4)\
                         .renamed(ynext = 'nextposition')
  ```

- **Extensible control synthesizers**

  Use built-in algorithms for computing controllers enforcing safety and reachability specifications, or customize them using the operators above.

- **Extensible Symbolic Backend**

  We build on top of libraries for symbolically representing sets and relations. The current implementation uses binary decision diagrams via the dd package. Support for aiger circuits is possible with the py-aiger library.

## Development Notes

### Status

The core foundations are working and providing results on some meaningful test examples but the APIs are fluctuating rapidly.

### TODOs

- Organize examples folder
- Scale up lunar lander
  - Fix issues with unnecessary blocking with position modules dependencies on angle.
  - Iterator for shifted but bounded space.
- Make a variable = name + types object
- Controlled predecessor with multiple time scale models and control inputs.
- Different iterators of input space
  - Iterators for the $2^{N-1}$ reduced grid traversal
- Rewrite continuous cover grid to have an overlap parameter

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
