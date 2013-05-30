#PyAutoDiff Changelog

##0.3 - May 2013

### Features

- Total rewrite of mid-level interface
    - New `Symbolic` class with much greater functionality (and generaltiy!)
    - New `Function`/`Gradient`/`HessianVector`/`VectorArg` classes that are subclasses of `Symbolic`
- Support for complex Python function signatures
    - Support for container arguments (lists/tuples/dicts) and nested containers
    - Automatic conversion from complex Python signatures to flat Theano signatures for compilation

##0.2 - May 2013

### Features

- Added `Symbolic` general tracing/compiling mechanism
- Added `tag` mechanism
- Support for decorating bound methods, `@staticmethod`, `@classmethod`
- Preliminary support for wrapping docstrings of traced functions



##0.1 - May 2013

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
