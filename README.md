#PyAutoDiff v0.1


####Automatic differentiation for NumPy.

---
PyAutoDiff automatically compiles NumPy code using [Theano](http://deeplearning.net/software/theano/)'s powerful symbolic engine, allowing users to take advantage of features like mathematical optimization, GPU acceleration, and automatic symbolic differentiation.

**This library is under active development. Features may break or change.**


## Quickstart
```python
from autodiff import function, gradient
```
The `@function` decorator instructs Theano to compile the decorated function. It is possible to run the compiled function on an available GPU if float32 dtypes are used.
```python
@function
def f(x):
    return x ** 2

f(10.0) # 100.0
f(20.0) # 400.0 
```

The `@gradient` decorator automatically calculates the gradient (or derivative) of any scalar-valued function with respect to its inputs. Use caution with integer arguments -- Theano usually defines any gradient with respect to an integer as 0!

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
def g(x, y, z=2):
    tmp = np.dot(x, y) * z
    tmp[0, 1] = tmp[1, 1]
    return tmp.sum()

a = np.array([1.0, 2.0])
b = np.arange(6.).reshape(2, 3)

g(a, b)  # [[ 0.2,  0.2,  0.2]
         #  [ 0.4,  0.4,  0.4]]     27.0

```

## Caveats
* PyAutoDiff expects function arguments to have Theano equivalents. Most NumPy array types or floats are ok; lists, tuples, dicts, and strings are not (with the exception of `*args` and `**kwargs`).
* Theano can only run float32 dtypes on the GPU (see the [documentation](http://deeplearning.net/software/theano/tutorial/using_gpu.html) for more information.)
* Theano usually defines any gradient with respect to an integer dtype as 0.
* Small Python `int` types can not be traced due to CPython caching. PyAutoDiff tries to convert them to NumPy `int` types when possible. However, this can create problems with some Theano functions that require `int` arguments. For example:

```python
@function
def this_works(x):
    return x.sum(axis=1)

@function
def this_fails(x, a):
    return x.sum(axis=a)
```

---
### Dependencies:
  * [NumPy](http://www.numpy.org/)
  * [Theano](http://deeplearning.net/software/theano/)

---
### With thanks:
  * [James Bergstra](https://github.com/jaberg) for bringing PyAutoDiff to light.
  * Travis Oliphant for posting a very early version of [numba](http://numba.pydata.org/) that provided the inspiration and starting point for this project.
  * The entire Theano team.

