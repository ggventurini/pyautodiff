#PyAutoDiff Changelog

##0.1 - May 2013

### Features

- Enhanced low-level interface
    - Updated `FrameVM` for Theano 0.6
        - Added all shared NumPy/Theano functions
        - Support for advanced indexing
    - Add `Constant` transform
- Added mid-level interface:
    - `Symbolic`, `Function`, `Gradient`, `HessianVector`, `VectorArg` (for SciPy optimization) classes
    - `@function`, `@gradient`, `@hessian-vector` decorators
- Added high-level interface for SciPy optimizers
    - L-BFGS-B, nonlinear conjugate gradient, Newton-CG
- Added unit tests
- Added compatibility module (`OrderedDict`, `getcallargs`)

### Fixes

- Wrapped small `int` variables to solve tracing issues

 

##0.0.1 (prototype) - June 2012
- Introduced low-level `Context` and `FrameVM` tracing objects
- L-BFGS-B minimization routine
- stochastic gradient descent routine
