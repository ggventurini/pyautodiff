#PyAutoDiff v0.1


####Automatic differentiation for NumPy.

---
PyAutoDiff automatically compiles NumPy code using [Theano](http://deeplearning.net/software/theano/)'s powerful symbolic engine, allowing users to take advantage of features like mathematical optimization, GPU acceleration, and automatic  differentiation.

**This library is under active development. Features may break or change.**

## Quickstart
```python
from autodiff import function, gradient
```
The `@function` decorator instructs PyAutoDiff to compile the decorated function in Theano. It is possible to run the compiled function on an available GPU if float32 dtypes are used.
```python
@function
def f(x):
    return x ** 2

f(10.0) # 100.0
f(20.0) # 400.0 
```

The `@gradient` decorator automatically calculates the gradient (or derivative) of any scalar-valued function with respect to its inputs. 

```python
@gradient
def g(x):
    return x ** 2

g(10.0) # 20.0
g(20.0) # 40.0


@gradient
def g(x, y):
    return x * y
    
g(3.0, 5.0) # (5.0, 3.0)


@gradient(wrt='x')
def g(x, y):
    return x * y
    
g(3.0, 5.0) # 5.0
```
PyAutoDiff supports most NumPy operations that have Theano equivalents, and is fully compatible with array calculations. Here is a more complex example: 
```python
@gradient('y', 'z')
def g(x, y, z=0.2):
    tmp = np.dot(x, y) * z
    tmp[0, 1] = tmp[1, 1]
    return tmp.sum()

a = np.arange(4.).reshape(2, 2)
b = np.arange(6.).reshape(2, 3)

g(a, b)  # [[ 0.4,  0.8,  0.4]
         #  [ 0.8,  1.2,  0.8]]     64.0

```

Classes may be used instead of decorators:
```python
from autodiff import Function, Gradient
f = Function(lambda x : x ** 2)
g = Gradient(lambda x : x ** 2, wrt='x')
```

Note that compiling "locks" any control flow tools:
```python
@function
def loop_mult(x, N):
    y = 0
    for i in range(N):
        y += x
    return y

loop_mult(2, 4) # 8
loop_mult(2, 5) # also 8! The loop is locked in the compiled function.
```

---
### Caveats
Generally, PyAutoDiff supports any NumPy operation with a Theano equivalent. You will probably get unexpected results if you use more general Python operations like control flow tools (`for`, `if/else`, `try/except`, etc.) or iteraters without understanding how Theano handles them.

When PyAutoDiff prepares to compile, it calls the Python function one time in order to find out what it does. With the exception of NumPy arrays and numbers, whatever happens on that first run is locked into the compiled function: the length of every `for` loop, the selected branch of every `if/else` statement, even the axis of every `np.sum(axis=my_var)`.

As a rule of thumb: if the code you're writing doesn't operate directly on a NumPy array, then there's a good chance it won't behave as you expect.

---
### Dependencies:
  * [NumPy](http://www.numpy.org/)
  * [Theano](http://deeplearning.net/software/theano/)

---
### With thanks:
  * [James Bergstra](https://github.com/jaberg) for bringing PyAutoDiff to light.
  * Travis Oliphant for posting a very early version of [numba](http://numba.pydata.org/) that provided the inspiration and starting point for this project.
  * The entire Theano team.

