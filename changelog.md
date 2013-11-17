#PyAutoDiff Changelog

## 0.4 - November 2013

Total rewrite of low-level backend to parse and manipulate function AST's, thereby avoiding having to call functions prior to compilation. Refactored mid-level interface to take advantage of new features, but mid/high-level API's remain largely the same.

### Features

- Total rewrite of backend
    - Deprecated bytecode backend (that required calling the function) in favor of static AST analysis and transformation.
    - New TheanoTransformer class for changing NumPy AST's into Theano AST's with runtime checks.
    - New classes for shadowing arbitrary objects
- Added `escape()` function and updated `tag()` capabilities
- Added `@theanify` decorator to encourage adoption of NumPy-to-Theano transformations without compilation



##0.3 - May 2013

Total rewrite of mid-level interface around class-based symbolic tracers. More work to "do the right thing" when Python constructs don't have Theano analogues (i.e. complex signatures).

### Features

- Total rewrite of mid-level interface
    - New `Symbolic` class with much greater functionality (and generaltiy!)
    - New `Function`/`Gradient`/`HessianVector`/`VectorArg` classes that are subclasses of `Symbolic`
- Support for complex Python function signatures
    - Support for container arguments (lists/tuples/dicts) and nested containers
    - Automatic conversion from complex Python signatures to flat Theano signatures for compilation

##0.2 - May 2013

Expanded support beyond simple functions to classes and tracing through multiple functions.

### Features

- Added `Symbolic` general tracing/compiling mechanism
- Added `tag` mechanism
- Support for decorating bound methods, `@staticmethod`, `@classmethod`
- Preliminary support for wrapping docstrings of traced functions



##0.1 - May 2013

Initial release. Updated prototype for Theano 0.6 and added mid/high level interfaces.

### Features

- Enhanced low-level interface
    - Updated `FrameVM` for Theano 0.6
        - Added all shared NumPy/Theano functions
        - Support for advanced indexing and inplace updates
- Added mid-level interface
    - `Function`, `Gradient`, `HessianVector`, `VectorArg` (for SciPy optimization) classes
    - `@function`, `@gradient`, `@hessian_vector` decorators
- Added high-level interface for SciPy optimizers
    - L-BFGS-B, nonlinear conjugate gradient, Newton-CG
- Added helper functions
    - `constant`
- Added unit tests
- Added compatibility module (`OrderedDict`, `getcallargs`)

### Fixes

- Wrapped small `int` variables to solve tracing issues



##0.0.1 (prototype) - June 2012
- Introduced low-level `Context` and `FrameVM` tracing objects
- L-BFGS-B minimization routine
- stochastic gradient descent routine
