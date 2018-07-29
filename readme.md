
VPAX (Variable Precision Abstract Controller Synthesis)
=====

VPAX is a tool for controller synthesis by first constructing finite state machines that mimic continuous dynamics.

Supports:

- Nonlinear Discrete-Time Systems
- Safety, Reachability, and Reach-Avoid Objectives

Requires:

- A method to safety overapproximate the set of open-loop reachable states from a hyperrectangle of initial states (this is necessary to ensure the outputted controller is sound and correct, but the tool can still run without it).

Ideally for:

- Systems with complex constraints
- Controller synthesis problems that are ideally tackled with the [level set toolbox](https://www.cs.ubc.ca/~mitchell/ToolboxLS/index.html)
- Objectives that can be encoded in a bounded set

<!-- - Systems of moderate dimension (4-6).  -->

Notable Features
=======

1. **Variable Precision Grids**

    The grid is a union of hyperrectangles and each hyperrectangle is uniquely identified by a bitvector. Throwing away lower precision bits implicitly makes the grid more coarse, which is useful if
    - Sample efficiency and more succinct encodings of the abstract finite state machine are desired
    - One wants to experience the benefits of both speed (coarse grids) and precision (fine grids) for controller synthesis

2. **Data-driven Abstraction**

    Many other controller synthesis tools require an executable function that yields an overapproximations of reachable sets. What if only a collection of input-output pairs is available? We can accommodate a data-driven abstraction procedure where a user provides only the following information about trajectories.

    [Initial State hyper-rectangle] x  [Control Input] -> [Hyperrectangle Overapproximation of successor states]

    The initial state hyperrectangles do *not* need to be disjoint, have identical sizes, or align with the grid.

3. **Linear, Periodic and Discrete Grids**

    Linear grids use the traditional binary encoding to identify hyperrectangles in the grid. Periodic grids use gray code. Different encodings can be used in tandem.
    ```python
    Code here
    ```

4. **Composable Modules**

    Abstractions of high dimensional systems can be difficult to construct because the underlying state space is a N-dimensional grid. One may instead construct them via abstracting smaller components and combining them via series and parallel composition.
    ```python
    #Series composition
    a >> b
    # Parallel composition
    a | b
    ```

5. **Symbolic Reasoning Backends**

    All of the above are compiled down to operations on boolean circuits.



Installation
======

Clone this repo and run the following commands:

```shellscript
cd /location/of/vpax/here/
pip install .
```

Dependencies
------

- dd with cudd
- bidict

TODOs before release
------

- Controller synthesis
  - Controller refinement class
- Spaces
  - Grids in continuous space 
  - More tests for numerical errors 
  - generic and polymorphic conc2abs function
  - Ones that don't align with boolean cube
- Move to py-aiger backend
- FRR check across modules for dynamic grids
- Change abstraction granularity (exists outputs, forall inputs)
- Non-eager composition so the structure is available to control synthesizer

TODOs
------

- Support for sum and product operations over algebraic data types