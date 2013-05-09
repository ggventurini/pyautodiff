#PyAutoDiff


#### Automatic differentiation for NumPy


PyAutoDiff automatically compiles NumPy code using [Theano](http://deeplearning.net/software/theano/)'s powerful symbolic engine, allowing users to take advantage of features like mathematical optimization, GPU acceleration, and automatic  differentiation.

**This library is under active development. Features may break or change.**

## Quickstart

####Decorators

PyAutoDiff provides simple decorators for compiling arbitrary NumPy functions and their derivatives. For most users, these will be the primary interface to autodiff.
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


## Concepts

#### Functions

The `Function` class and `@function` decorator use Theano to compile the target function. PyAutoDiff has support for all NumPy operations with Theano equivalents and limited support for many Python behaviors (see caveats).

#### Gradients

The `Gradient` class and `@gradient` decorator compile functions which return the gradient of the the target function. The target function must be scalar-valued. A `wrt` keyword may be passed to the class or decorator to indicate which variables should be differentiated; otherwise all arguments are used.


#### Constants

PyAutoDiff replaces many variables with symbolic Theano versions. This can cause problems, because some Theano functions do not support symbolic inputs. To resolve this, autodiff provides a `Constant()` modifier, which instructs PyAutoDiff not to try and build a symbolic version of that variable. 

Most of the time, users will not have to call Constant() -- it is only necessary in certain cases.

For example, the following functions will compile, because the `axis` argument `1` is loaded as a constant, even when bound to a variable `a`.
```python
from autodiff import Constant, function
m = np.ones((3, 4))

@function
def fn_1(x):
    return x.sum(axis=1)
    
@function
def fn_2(x):
    a = 1
    return x.sum(axis=a)
    
print fn_1(m)
```
However, function arguments are always assumed to be symbolic. The following function will fail because the `axis` argument is the symbolic variable `a` and `tensor.sum` does not accept symbolic arguments:
```python
@function
def bad_fn(x, a):
    return x.sum(axis=a)

print bad_fn(m, 1) # error
```

By calling `Constant()` appropriately, we can avoid assigning a symbolic variable. This function will compile, because the axis argument is treated as the constant `1`:
```python
@function
def good_fn(x, a):
    return x.sum(axis=Constant(a))
    
print good_fn1(m)
```

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

In the current version of PyAutoDiff, there **is** a way to avoid this problem, but at the cost of significantly more expensive calculations. If an autodiff class is instantiated with keyword `use_cache=False`, then it will not cache its compiled functions. Therefore, it will reevaluate all control flow statements at every call. However, it will call the NumPy function, compile a Theano function, and call the Theano function every time -- meaning functions will take at least twice as long to run and possibly more. This should only be used as a last resort if more clever designs are simply not possible.

**As a rule of thumb: if the code you're writing doesn't operate directly on a NumPy array, then there's a good chance it won't behave as you expect.**

Here is an example of compilation "locking" a control flow, and how to set `use_cache` to avoid it:
```python
from autodiff import function

def loop_mult(x, N):
    y = 0
    for i in range(N):
        y += x
    return y

f = Function(loop_mult)
print f(2, 4) # 8
print f(2, 5) # also 8! The loop is locked in the compiled function.

g = Function(loop_mult, use_cache=False)
print g(2, 4) # 8
print g(2, 5) # 10, but a much slower calculation than the cached version.
```

## Dependencies
  * [NumPy](http://www.numpy.org/)
  * [Theano](http://deeplearning.net/software/theano/)


## With great thanks
  * [James Bergstra](https://github.com/jaberg) for bringing PyAutoDiff to light.
  * Travis Oliphant for posting a very early version of [numba](http://numba.pydata.org/) that provided the inspiration and starting point for this project.
  * The entire Theano team.

