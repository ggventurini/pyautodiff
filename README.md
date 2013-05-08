#PyAutoDiff


#### Automatic differentiation for NumPy


PyAutoDiff automatically compiles NumPy code using [Theano](http://deeplearning.net/software/theano/)'s powerful symbolic engine, allowing users to take advantage of features like mathematical optimization, GPU acceleration, and automatic  differentiation.

**This library is under active development. Features may break or change.**

## Quickstart

####Decorators

PyAutoDiff provides simple decorators for compiling NumPy functions or their derivatives.
```python
from autodiff import function, gradient

#--- compile a Theano function

@function
def f(x):
    return x ** 2

print f(5.0) # 25.0

#--- compile a function returning the gradient

@gradient
def f(x):
    return x ** 2

print f(5.0) # 10.0

#--- compile a function returning the gradient only with respect to a specific input

@gradient(wrt='y')
def f(x, y):
    return x * y
    
print f(3.0, 5.0) # 3.0
```

#### Optimization

Users can call a higher-level optimization interface that wraps SciPy minimization routines (currently L-BFGS-B, nonlinear conjugate gradient, and Newton-CG), using autodiff to compute the required derivatives and Hessian-vector products.
```python
from autodiff.optimize import fmin_l_bfgs_b

#--- A trivial least-squares minimization problem

y = np.arange(3.0)
def fn(x):
    return ((x - y) ** 2).mean()
    
x_opt = fmin_l_bfgs_b(fn, init_args=np.zeros(3))

print x_opt # [0.0, 1.0, 2.0]
print y     # [0.0, 1.0, 2.0]

```

#### Classes

Autodiff classes are also available (the decorators are simply convenient ways of automatically wrapping functions in classes). In addition to the function and gradient decorators/classes shown here, a Hessian-vector product class and decorator are also available.
```python
from autodiff import Function, Gradient

def fn(x):
    return x ** 2

f = Function(fn) # compile the function
g = Gradient(fn) # compile the gradient of the function

print f(5.0) # 25.0
print g(5.0) # 10.0

```

#### Constants

PyAutoDiff tries to replace every variable used in a function with a symbolic Theano version. This can cause problems, in particular when variables (usually `ints`) are used as arguments to other functions or methods. To resolve this, use a `Constant()` modifier, which instructs PyAutoDiff not to try and build a symbolic version of that variable.

The following function will compile, because the `axis` argument (1) is loaded as a constant.
```python
from autodiff import Constant, function
m = np.ones((3, 4))

@function
def fn(x):
    return x.sum(1)
    
print fn(m)
```
However, these next two examples won't work, because the `axis` argument is now a variable, and Theano's `sum` method doesn't accept variable axes.
```python
@function
def bad_fn1(x):
    a = 1
    return x.sum(a)

print bad_fn1(m) # error

@function
def bad_fn2(x, a):
    return x.sum(a)
    
print bad_fn2(m, 1) # error
```
By calling `Constant()` appropriately, we can avoid assigning a symbolic variable. These two functions will compile.
```python
@function
def good_fn1(x):
    a = Constant(1)
    return x.sum(a)
    
print good_fn1(m)
    
@function
def good_fn2(x, a):
    return x.sum(Constant(a))
    
print good_fn2(m, 1)
```


## Concepts

#### Functions

The `Function` class and `@function` decorator use Theano to compile the target function. PyAutoDiff has support for all NumPy operations with Theano equivalents and limited support for many Python behaviors (see caveats).

#### Gradients

The `Gradient` class and `@gradient` decorator compile functions which return the gradient of the the target function. The target function must be scalar-valued. A `wrt` keyword may be passed to the class or decorator to indicate which variables should be differentiated; otherwise all arguments are used.

#### Constants

PyAutoDiff attempts to shadow every numeric variable in a function with a Theano symbolic variable. This creates problems, for example, with `range()` and the `axis` argument of many Theano reductions, because they raise errors for symbolic arguments. PyAutoDiff provides a `Constant()` function that instructs it not to shadow a certain variable, meaning it can be used without problem (see above for examples).

#### Hessian-vector products

The `HessianVector` class and `@hessian_vector` decorator compile functions that return the product of an argument's Hessian and an arbitrary vector (or tensor). The vectors must be provided to the resulting function with the `_tensors` keyword argument.

#### Optimization

The `autodiff.optimize` module wraps some SciPy minimizers, automatically compiling functions to compute derivatives and Hessian-vector products that the minimizers require in order to optimize an arbitrary function.


## Caveats

#### dtypes

Pay attention to dtypes -- they are locked when Theano compiles a function. In particular, note the following:
- The gradient of an integer argument is defined as zero.
- Theano only supports `float32` operations on the GPU

#### Control Flow and Loops

Generally, PyAutoDiff supports any NumPy operation with a Theano equivalent. You will probably get unexpected results if you use more general Python operations like control flow tools (`for`, `if/else`, `try/except`, etc.) or iteraters without understanding how Theano handles them.

When PyAutoDiff prepares to compile, it calls the Python function one time in order to find out what it does. With the exception of NumPy arrays and numbers, whatever happens on that first run is locked into the compiled function: the length of every `for` loop, the selected branch of every `if/else` statement, even the axis of every `np.sum(axis=my_var)`.

As a rule of thumb: if the code you're writing doesn't operate directly on a NumPy array, then there's a good chance it won't behave as you expect.

Here is an example of compilation "locking" control flow:
```python
@function
def loop_mult(x, N):
    y = 0
    for i in range(N):
        y += x
    return y

print loop_mult(2, 4) # 8
print loop_mult(2, 5) # also 8! The loop is locked in the compiled function.
```

## Dependencies
  * [NumPy](http://www.numpy.org/)
  * [Theano](http://deeplearning.net/software/theano/)


## With great thanks
  * [James Bergstra](https://github.com/jaberg) for bringing PyAutoDiff to light.
  * Travis Oliphant for posting a very early version of [numba](http://numba.pydata.org/) that provided the inspiration and starting point for this project.
  * The entire Theano team.

