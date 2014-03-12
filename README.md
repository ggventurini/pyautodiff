#PyAutoDiff


#### Automatic differentiation for NumPy

AutoDiff automatically compiles NumPy code with [Theano](http://deeplearning.net/software/theano/)'s powerful symbolic engine, allowing users to take advantage of features like mathematical optimization, GPU acceleration, and automatic  differentiation.

Please note that development has shifted to Python 3, though this branch maintains support for Python 2. Python 3 users should check out the [python3](https://github.com/LowinData/pyautodiff/tree/python3) branch.

## Quickstart

### Decorators
AutoDiff decorators are the simplest way to leverage the library and will be the primary interface for most users. The `@function` and `@gradient` decorators allow Theano to be leveraged invisibly; the `@symbolic` decorator runs code through Theano but does not compile and execute it. The latter functionality can also be accessed via the `@theanify` alias.

```python
from autodiff import function, gradient

# -- use Theano to compile a function

@function
def f(x):
    return x ** 2

f(5.0) # returns 25.0; not surprising but executed in Theano!

# -- automatically differentiate a function with respect to its inputs

@gradient
def f(x):
    return x ** 2

f(5.0) # returns 10.0 because f'(5.0) = 2 * 5.0 = 10.0

# -- symbolically differentiate a function with respect to a specifc input

@gradient(wrt='y')
def f(x, y):
    return x * y

f(x=3.0, y=5.0) # returns 3.0 because df/dy(3.0, 5.0) = x = 3.0

@symbolic
def f(x):
    return x ** 2

f(5.0) # returns a Theano scalar object
f(np.ones((3, 4))) # returns a Theano matrix object
f(T.scalar()) # returns a Theano scalar object

```

### General Symbolic Tracing
The `Symbolic` class allows more general tracing of NumPy objects through (potentially) multiple functions. Users should call its `trace` method on any functions and arguments, followed by either the `compile_function` or `compile_gradient` method in order to get a compiled Theano function.

Critically, `Symbolic` can compile functions not only from existing arguments and results, but of any NumPy object
referenced while tracing. The following example traces objects through three different functions and ultimately compiles a function of an existing argument, a global variable, and a local variable via AutoDiff's `tag` mechanism:

```python
import numpy as np
import theano.tensor
from autodiff import Symbolic, tag

# -- a vanilla function
def f1(x):
    return x + 2

# -- a function referencing a global variable
y = np.random.random(10)
def f2(x):
    return x * y

# -- a function with a local variable
def f3(x):
    z = tag(np.ones(10), 'local_var')
    return (x + z) ** 2

# -- create a general symbolic tracer and apply it to the three functions
x = np.random.random(10)
tracer = Symbolic()

out1 = tracer.trace(f1, x)
out2 = tracer.trace(f2, out1)
out3 = tracer.trace(f3, out2)

# -- compile a function representing f(x, y, z) = out3
new_fn = tracer.compile_function(inputs=[x, y, 'local_var'],
                                 outputs=out3)

# -- compile the gradient of f(x) = out3, with respect to y
fn_grad = tracer.compile_gradient(inputs=x,
                                  outputs=out3,
                                  wrt=y,
                                  reduction=theano.tensor.sum)

assert np.allclose(new_fn(x, y, np.ones(10)), f3(f2(f1(x))))
```

### Classes

AutoDiff classes are also available (the decorators are simply convenient ways of automatically wrapping functions in classes). In addition to the function` and gradient decorators/classes shown here, a Hessian-vector product class and decorator are also available.

```python
from autodiff import Function, Gradient

def fn(x):
    return x ** 2

f = Function(fn) # compile the function
g = Gradient(fn) # compile the gradient of the function

print f(5.0) # 25.0
print g(5.0) # 10.0

```

## Important notes

### Code transformation
AutoDiff takes NumPy functions and attempts to make them compatible with Theano objects by analyzing their code and transforming it as necessary. This takes a few forms:

 * NumPy arrays are intercepted and replaced by equivalent Theano variables
 * NumPy functions are replaced by Theano versions (for example, `np.dot` is replaced by `T.dot`)
 * NumPy syntax is replaced by its Theano analogue (for example, inplace array assignment is replaced by a call to `T.set_subtensor`)

 ### Caveats

 **As a rule of thumb, if the code you're writing doesn't operate directly on a NumPy array, then there's a good chance the Theano version won't behave as you expect.**

 There are important differences between NumPy and Theano that make some transformations impossible (or, at least, yield unexpected results). A NumPy function can be called multiple times, and each time the code in the function is executed on its arguemnts. A Theano function is called only once, after which its output is analyzed and compiled into a static Theano function.

 This means that control flow and loops don't work the same way in NumPy and Theano and (often) can't be transformed properly. For example, an `if-else` statement in a NumPy function will examine its conditional argument and select the appropriate branch every time. The same `if-else` branch in a Theano function will be executed one time, and the selected branch will be compiled as the sole path in the resulting Theano function.

 In AutoDiff, there **is** a way to sometimes avoid this problem, but at the cost of significantly more expensive calculations. If an `autodiff` class is instantiated with keyword `use_cache=False`, then it will not cache its compiled functions. Therefore, it will reevaluate all control flow statements at every call. However, compile and call a Theano function every time it is called -- meaning functions will take significantly longer to run. This should only be used as a last resort if more clever designs are simply not possible -- note that it will not solve all problems, as Theano and NumPy have certain irreconcilable differences.

In addition, other small details may be important. For example, `dtypes` matter! In particular, Theano considers the gradient of an integer argument to be undefined, and also only supports `float32` dtypes on the GPU.

## Concepts

### Symbolic

The `Symbolic` class is used for general tracing of NumPy objects through Theano. Following tracing, its `compile` method can be used to compile Theano functions returning any combination of the function, gradient, and Hessian-Vector product corresponding to the provided inputs and outputs. The `compile_function`, `compile_gradient`, and `compile_function_gradient` methods are convenient shortcuts.

### Functions

The `Function` class and `@function` decorator use Theano to compile the target function. AutoDiff has support for all NumPy operations with Theano equivalents and limited support for many Python behaviors (see caveats).

### Gradients

The `Gradient` class and `@gradient` decorator compile functions which return the gradient of the the target function. The target function must be scalar-valued. A `wrt` keyword may be passed to the class or decorator to indicate which variables should be differentiated; otherwise all arguments are used.

### Hessian-vector products

The `HessianVector` class and `@hessian_vector` decorator compile functions that return the product of an argument's Hessian and an arbitrary vector (or tensor). The vectors must be provided to the resulting function with the `_tensors` keyword argument.

### Optimization

The `autodiff.optimize` module wraps some SciPy minimizers, automatically compiling functions to compute derivatives and Hessian-vector products that the minimizers require in order to optimize an arbitrary function.

### Special Functions

#### Escape

AutoDiff provides an `escape()` function, which signals that the library shouldn't attempt to transform any of the code inside the function. Additionally, if a variable has been replaced with a Theano object, calling escape() on it will restore the underlying NumPy array. This provides a limited mechanism for using constructs that aren't compatible with Theano. Also see the `escaped_call()` function for calling functions without attempting to analyze or transform their code.

#### Tag
AutoDiff tacing makes it relatively easy to access a function's symbolic inputs and outputs, allowing Theano to compile the function with ease. However, advanced users may wish to access the symbolic representations of other variables, including variables local to the function. To that end, users can manually tag symbolic variables with arbitrary keys, as the following example demonstrates:

```python
from autodiff import tag

@function
def local_fn(x):
    y = tag(x + 2, 'y_var')
    z = y * 3
    return z

local_fn(10.0)                   # call local_fn to trace and compile it
y_sym = local_fn.s_vars['y_var'] # access the symbolic version of the function's
                                 # local variable 'y', tagged as 'y_var'
```

Tagging is especially useful in combination with AutoDiff's `Symbolic` class, as it allows tracing and compiling functions of purely local variables. An example of this behavior can be found in the Symbolic section of the Quickstart.

## Dependencies
  * [NumPy](http://www.numpy.org/)
  * [Theano](http://deeplearning.net/software/theano/)
  * [Meta](https://github.com/numba/meta)


## With great thanks
  * [James Bergstra](https://github.com/jaberg) for bringing AutoDiff to light.
  * Travis Oliphant for posting a very early version of [numba](http://numba.pydata.org/) that provided the inspiration and starting point for this project.
  * The entire Theano team.

