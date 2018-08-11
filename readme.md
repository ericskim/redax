
PYDRA (A Python Implementation of Dynamic and Reusable Abstractions)
=====

PYDRA is a controller synthesis tool that constructs finite state machines that mimic continuous dynamics. We call the finite state machines *abstractions* because they abstract out low level state information.

Supports:

- Nonlinear Discrete-Time Control Systems
- Safety, Reachability, and Reach-Avoid Objectives

Requires:

- A method to safety overapproximate the set of open-loop reachable states from a hyperrectangle of initial states (this is necessary to ensure the outputted controller is sound and correct, but the tool may still yield useful results without it).

Ideally for:

- Systems with complex constraints and nonlinear dynamics.
- Objectives and systems that are of interest in a bounded set.
- Controller synthesis problems approached by gridding a continuous space such as in dynamic programming.
- Systems of moderate state dimensions (4-6).
- Problems tackled by the following tools:
  - [Level Set Toolbox](https://www.cs.ubc.ca/~mitchell/ToolboxLS/)
  - [SCOTS control synthesis toolbox](https://gitlab.lrz.de/hcs/scots)

Installation
======

Clone this repo and run the following commands:

```shellscript
cd /location/of/pydra/here/
pip install .
```

Requirements
------

- python 3.6 or above
- cudd with dd wrapper
- numpy
- bidict

Distinguishing Features
=======

- **Extensible Abstraction Spaces**

  Abstracting a set means grouping together points and constructing a lower cardinality set. We support a number of methods such as:

  - Fixed vs. dynamic precision covers: Union of rectangles where each one is uniquely identified by a bitvector. Throwing away lower precision bits implicitly makes the cover more coarse.
  - Linear vs. periodic covers: use the traditional binary encoding to identify hyperrectangles in the grid. Periodic grids use [gray code](https://en.wikipedia.org/wiki/Gray_code).
  - Embedded grids: A grid of discrete points embedded in continuous space, like the integers.
  - Your own space here!

  ```python
  # Declare continuous state spaces for a Dubins vehicle
  pspace      = DynamicCover(-2,2)
  anglespace  = DynamicCover(-np.pi, np.pi, periodic=True)
  # Declare discrete control spaces for a Dubins vehicle
  vspace      = EmbeddedGrid(vmax/2, vmax, 2)
  angaccspace = EmbeddedGrid(-1.5, 1.5, 3)
  ```

- **Small Reusable Modules**

  Modules represent stateless input-output maps but can have non-deterministic outputs and may block for some inputs.

  ```python
  # Declare modules
  dubins_x        = AbstractModule(mgr, {'x': pspace, 'theta': anglespace, 'v': vspace},
                                        {'xnext': pspace})
  dubins_y        = AbstractModule(mgr, {'y': pspace, 'theta': anglespace, 'v': vspace},
                                        {'ynext': pspace})
  dubins_theta    = AbstractModule(mgr, {'theta': anglespace, 'v': vspace, 'omega': angaccspace},
                                        {'thetanext': anglespace})
  ```

- **Flexible Abstraction Construction**

  Abstractions of modules are constructed by providing the following input-output data

  - [Input set] -> [Overapproximation of the set of possible outputs]

  There are no further assumptions on the input-output data. The samples do NOT need disjoint input sets or have sets align with a grid. The controller synthesis results get better as sample quality and quantity increase.

  ```python
  # Collections of input-output pairs can be generated any number of ways!
  collection = database/file or iterator or odesolver/simulator or random input generator

  # Anytime algorithm for abstraction construction
  for in_out_pair in collection:
    abstraction.apply_abstract_transitions(in_out_pair)
  ```

- **Manipulate and Compose Modules**

  Abstract systems can be manipulated with a collection of operations:

  ```python

  # Variable Renaming
  dubins >> ('xnext', 'xnext')

  # Parallel composition is useful for constructing larger dimensional systems
  dubins = dubins_x | dubins_y | dubins_theta

  # Series composition resembles function composition m2(m1(.))
  m1 >> m2

  # Hide outputs
  Need a simpler example...

  # Future feature: Compute lower complexity abstractions keeping only the most significant bits
  simple_model = model.coarsen(x=1, y=2)
  ```

  Operations can also be chained together:

  ```python
  simpler_dubins = dubins.hide('xnext')\
                         .coarsen(x=5,y=4)\
                         .rename(ynext = 'nextposition')
  ```

- **Extensible Symbolic Backend**

  We build on top of libraries for binary decision diagrams.

Notes and References
======

References
-------

- [*Abstractions for Symbolic Controller Synthesis are Composable*, Eric S. Kim, Murat Arcak](https://arxiv.org/abs/1807.09973)

TODOs
------

- Tests for floating point inequalities for conc2abs method in continuous covers
  - Iterators for the 2^n reduced grid traversal
- Document more (especially class attributes)
- FRR check across modules for dynamic grids
- Custom errors for out of domain
- Example
  - 3DOF ship
  - OpenAI lunar lander
  - Pair of dubins vehicles
- Reinstall from scratch in a virtualenv
  - Upload to github and add code coverage, travis-ci banners
- Test performance of an immutable AbstractModule and abstraction via a .io_refine() method.
- Rewrite continuous cover grid to have an overlap parameter.
- Rewrite safe/target predicates as source and sink modules and use series composition operator for synthesis.

Future Features
------

- Concrete executable functions associated with module dynamics
- Support for disjoint union and product operations for spaces (thus adding support for switched and hybrid spaces)
- Control synthesizer that is aware of the control system structure. Options include:
  1. Non-eager composition like in neural network packages. This is useful if we have concrete executables inside the module.
  2. Provide a collection of modules and have the synthesizer construct a DAG internally.
- Generic wrapper for py-aiger and dd manipulation of predicates

Design Choices
========
