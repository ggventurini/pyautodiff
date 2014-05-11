import unittest
import numpy as np
import copy
import builtins
import theano
import theano.tensor as T
import autodiff
import autodiff.utils as utils
import autodiff.context as c
from autodiff.functions import escape


context = autodiff.context.Context(force_floatX=False)
context_floatX = autodiff.context.Context(force_floatX=True)

def checkfn(f, var_ndim=None, *args, **kwargs):
    test_floatX = kwargs.pop('test_floatX', True)
    result1 = _checkfn(context, f, var_ndim, *args, **kwargs)
    if test_floatX:
        result2 = _checkfn(context_floatX, f, var_ndim, *args, **kwargs)
        return result1 and result2
    else:
        return result1

def _checkfn(context, f, var_ndim=None, *args, **kwargs):
    context.reset()

    override = kwargs.pop('override', None)
    var_ndim = utils.as_seq(var_ndim)
    dim = [[4] * nd for nd in var_ndim]
    values = tuple([np.random.random(d) for d in dim])
    # make shallow copies to avoid inplace corruption
    sym_values = copy.copy(values)
    sym_args = copy.copy(args)
    sym_kwargs = copy.copy(kwargs)

    F = context.recompile(f)

    sym_vars = F(*(sym_values + sym_args), **sym_kwargs)
    sym_result = [v.eval() if utils.isvar(v) else v
                  for v in utils.as_seq(sym_vars)]

    if len(sym_result) == 0:
        sym_result = None

    py_result = override or f(*(values + args), **kwargs)

    if sym_result is None:
        return sym_result is None and py_result is None
    else:
        return np.allclose(py_result, sym_result)


class GarbageCollection(unittest.TestCase):
    # make sure shadowed variables aren't garbage-collected
    # so their id's do not get reused. If gc takes effect, then
    # x and y will coexist in the same location in memory (weird...)
    def test_gc(self):
        def f(x, y):
            return [x, y]

        F = context.recompile(f)
        assert F(3, 4)[1].eval() == 4


class Tags(unittest.TestCase):
    def test_tagging(self):
        def f(arg1, arg2=1, *arg3, **arg4):
            pass

        F = context.recompile(f)
        F(1.0)
        self.assertTrue('arg1' in context.sym_vars)
        self.assertTrue('arg2' in context.sym_vars)
        self.assertTrue('arg3' not in context.sym_vars)
        self.assertTrue('arg4' not in context.sym_vars)


class ForceFloatX(unittest.TestCase):
    def test_force_floatX(self):
        def f(x):
            return x
        ctx = autodiff.context.Context(force_floatX=False)
        ctx_floatX = autodiff.context.Context(force_floatX=True)
        F = ctx.recompile(f)
        F_floatX = ctx_floatX.recompile(f)

        x = np.array([1, 2, 3])
        self.assertTrue(F(x).dtype == 'int64')
        self.assertTrue(F_floatX(x).dtype == theano.config.floatX)


class Signatures(unittest.TestCase):
    def test_sig_no_arg(self):
        def f():
            return 1
        self.assertTrue(checkfn(f))

    def test_sig_one_arg(self):
        def f(x):
            return x
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, a=2)
        self.assertTrue(checkfn(f, [], 2))
        self.assertTrue(checkfn(f, [], x=2))

    def test_sig_mult_args(self):
        # multiple args, no default
        def f(x, y):
            return x * y
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, 2)
        self.assertRaises(TypeError, f, a=2, b=2)
        self.assertTrue(checkfn(f, [], 2, 3))
        self.assertTrue(checkfn(f, [], y=4, x=5))

    def test_sig_var_args(self):
        # var args, no default
        def f(x, y, *z):
            return x * y * sum(z)
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, 2)
        self.assertRaises(TypeError, f, a=2, b=2)
        self.assertTrue(checkfn(f, [], 2, 3))
        self.assertTrue(checkfn(f, [], 2, 3, 4))
        self.assertTrue(checkfn(f, [], 2, 3, 4, 5))

    def test_sig_default_args(self):
        # multiple args, one default
        def f(x, y=2):
            return x * y
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, y=3)
        self.assertTrue(checkfn(f, [], 2))
        self.assertTrue(checkfn(f, [], 2, 3))
        self.assertTrue(checkfn(f, [], y=4, x=5))
        self.assertTrue(checkfn(f, [], x=5))

        # multiple args, all default
        def f(x=1, y=2):
            return x * y
        self.assertTrue(checkfn(f))
        self.assertTrue(checkfn(f, [], 1))
        self.assertTrue(checkfn(f, [], 1, 2))
        self.assertTrue(checkfn(f, [], y=2, x=1))
        self.assertTrue(checkfn(f, [], x=5))
        self.assertTrue(checkfn(f, [], y=5))

    def test_sig_default_var_args(self):
        # multiple var args, all default
        def f(x=1, y=2, *z):
            return x * y * sum(z)
        self.assertTrue(checkfn(f))
        self.assertTrue(checkfn(f, [], 1))
        self.assertTrue(checkfn(f, [], 1, 2))
        self.assertTrue(checkfn(f, [], 1, 2, 3))
        self.assertTrue(checkfn(f, [], 1, 2, 3, 4))

    def test_sig_kwargs(self):
        # kwargs
        def f(**kwargs):
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            return x * y * z
        self.assertRaises(KeyError, f)
        self.assertRaises(TypeError, f, 1)
        self.assertTrue(checkfn(f, [], x=1, y=2, z=3))

    def test_sig_varargs_kwargs(self):
        # varargs and kwargs
        def f(a, *b, **kwargs):
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            return x * y * z
        self.assertRaises(TypeError, f)
        self.assertRaises(KeyError, f, 1)
        self.assertRaises(TypeError, f, x=1, y=2, z=3)
        self.assertTrue(checkfn(f, [], 1, x=1, y=2, z=3))
        self.assertTrue(checkfn(f, [], 1, 2, 3, x=1, y=2, z=3))

        # varargs and kwargs, use varargs
        def f(a, *b, **kwargs):
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            return x * y * z * b[0]
        self.assertTrue(checkfn(f, [], 1, 2, x=1, y=2, z=3))
        self.assertTrue(checkfn(f, [], 1, 2, 3, x=1, y=2, z=3))

    def test_expand_varargs(self):
        def f(*args):
            return args[1]

        def g(x):
            args = (x, np.ones((2, 3)), 5)
            return f(*args)

        self.assertTrue(checkfn(g, [], 1))

    def test_expand_kwargs(self):
        def f(**args):
            return args['x']

        def g(x):
            args = dict(x=x, y=np.ones((2, 3)), z=5)
            return f(**args)

        self.assertTrue(checkfn(g, [], 1))


class Python(unittest.TestCase):
    def test_range(self):
        def f(x):
            for i in range(3):
                x += 5
            return x
        self.assertTrue(checkfn(f, [1]))

        def f(x):
            a = 3
            for i in range(a):
                x += 5
            return x
        self.assertTrue(checkfn(f, [1]))

        def f(x):
            a = x[0] + 10
            for i in range(int(a)):
                x += 5
            return x
        self.assertTrue(checkfn(f, [1]))

        def f(x, a):
            for i in range(a):
                x += 5
            return x
        self.assertTrue(checkfn(f, [1], 3))

        def f():
            l = []
            for i in range(3):
                l.append(i)
            return l
        self.assertTrue(checkfn(f, []))

        def f():
            l1 = {i:i for i in range(3)}
            l2 = [l1[i] for i in range(3)]
            return l2
        self.assertRaises(KeyError, checkfn, f, [])

        def f():
            l1 = {escape(i):i for i in range(3)}
            l2 = [l1[escape(i)] for i in range(3)]
            return l2
        self.assertTrue(checkfn(f, []))

    def test_pass(self):
        def fn(x):
            pass
        self.assertTrue(checkfn(fn, [1]))

    def test_if(self):
        # test that if statements escape their test arguments
        def f(switch):
            if switch > 0:
                return 1
            else:
                return -1
        self.assertTrue(checkfn(f, [], -10))

    def test_for(self):
        def f():
            x = 0
            for i in range(5):
                x += i
            return x
        self.assertTrue(checkfn(f))

        def f(x):
            for i in range(5):
                x += i
            return x
        self.assertTrue(checkfn(f, [1]))

    def test_enumerate(self):
        def f1(x):
            z = np.arange(x.shape[0])
            for i, xi in enumerate(range(4)):
                z[i] += xi
            return z
        self.assertTrue(checkfn(f1, [1]))

        def f2(x):
            z = np.arange(x.shape[0])
            for i, xi in enumerate(x):
                z[i] += xi
            return z
        self.assertRaises(TypeError, checkfn, f2, [1])

    def test_sum(self):
        def f():
            x = np.ones(5)
            y = np.ones(5) * 5
            return builtins.sum([x, y])
        self.assertTrue(checkfn(f, []))

    def test_max(self):
        def f():
            x = np.arange(5)
            return builtins.max(x)
        self.assertTrue(checkfn(f, []))

        def f(x):
            return builtins.max(x)
        self.assertTrue(checkfn(f, [1]))

    def test_min(self):
        def f():
            x = np.arange(5)
            return builtins.min(x)
        self.assertTrue(checkfn(f, []))

        def f(x):
            return builtins.min(x)
        self.assertTrue(checkfn(f, [1]))

    def test_isinstance(self):
        def f(x):
            if isinstance(x, int):
                return 1
            elif isinstance(x, float):
                return -1
        self.assertTrue(checkfn(f, [], 1, test_floatX=False))
        self.assertTrue(checkfn(f, [], 1.0, test_floatX=False))

    def test_tuple_index(self):
        def f(*x):
            return x[1]
        self.assertTrue(checkfn(f, [], 1, 2, 3))

    def test_nested_tuple_index(self):
        def f(*x):
            return x[1]
        def g(*x):
            return f(*x)
        self.assertTrue(checkfn(g, [], 1, 2, 3))

    def test_nested_def_tuple_index(self):
        def g(*x):
            def f(*x):
                return x[1]
            return f(*x)
        self.assertTrue(checkfn(g, [], 1, 2, 3))

    def test_append(self):
        def f():
            l = []
            for i in range(5):
                l.append(i)
            return l
        self.assertTrue(checkfn(f, []))

    def test_list_comprehension(self):
        def f():
            x = np.arange(10.0)
            y = [xi + 10 for xi in escape(x)]
            return y
        self.assertTrue(checkfn(f, []))

    def test_dict_comprehension(self):
        def f():
            x = np.arange(10.0)
            y = {escape(xi): xi + 10 for xi in escape(x)}
            return y[5]
        self.assertTrue(checkfn(f, []))

    def test_tuple_type(self):
        def f():
            x = tuple((3, 4, 5))
            return x

        def f2(x, y):
            return tuple(i for i in [x, y])

        self.assertTrue(checkfn(f, []))
        self.assertTrue(checkfn(f2, [], 1.0, 2.0))

    def test_inplace_container(self):
        def f():
            x = {3,4,5}
            x.remove(4)
            return sum(x)
        self.assertTrue(checkfn(f, []))

class BasicMath(unittest.TestCase):
    def test_basic_ops(self):
        for d in range(3):
            self.assertTrue(checkfn(lambda x: x + 2, [d]))
            self.assertTrue(checkfn(lambda x: x - 2, [d]))
            self.assertTrue(checkfn(lambda x: x * 2, [d]))
            self.assertTrue(checkfn(lambda x: x / 2, [d]))
            self.assertTrue(checkfn(lambda x: x / 2.0, [d]))
            self.assertTrue(checkfn(lambda x: x // 2.0, [d]))
            self.assertTrue(checkfn(lambda x: x ** 2, [d]))
            self.assertTrue(checkfn(lambda x: x % 2, [d]))

    def test_comparisons(self):
        for d in range(3):
            self.assertTrue(checkfn(lambda x, y: x > y, [d, d]))
            self.assertTrue(checkfn(lambda x, y: x < y, [d, d]))
            self.assertTrue(checkfn(lambda x, y: x >= y, [d, d]))
            self.assertTrue(checkfn(lambda x, y: x <= y, [d, d]))
            self.assertTrue(checkfn(lambda x, y: x == y, [d, d]))
            self.assertTrue(checkfn(lambda x, y: x != y, [d, d]))

    def test_inplace(self):

        def iadd(x):
            x += 10
            return x

        def isub(x):
            x -= 10
            return x

        def imul(x):
            x *= 10
            return x

        def idiv(x):
            x /= 10.0
            return x

        for d in range(3):
            for f in [iadd, isub, imul, idiv]:
                self.assertTrue(checkfn(f, [d]))


class NumpyFns(unittest.TestCase):
    """
    Test for coverage of functions in np namespace
    """
    def test_all(self):
        def fn(x):
            return np.all(x > .5)
        self.assertTrue(checkfn(fn, [2]))

    def test_any(self):
        def fn(x):
            return np.any(x > .5)
        self.assertTrue(checkfn(fn, [2]))

    def test_arange(self):
        self.assertTrue(checkfn(lambda: np.arange(3), []))
        # numpy arange doesn't return an array with the same dtype as its
        # argument, but theano arange does. In Context, the numpy arange
        # should be cast to match the theano one.
        self.assertTrue(checkfn(lambda: np.arange(np.float32(3.)), []))

    def test_abs(self):
        def fn1(x):
            return np.abs(x)

        def fn2(x):
            return abs(x)

        self.assertTrue(checkfn(fn1, [2]))
        self.assertTrue(checkfn(fn2, [2]))

    def test_dot(self):
        def fn(x, y):
            return np.dot(x, y)
        for nd in np.ndindex(*([3] * fn.__code__.co_argcount)):
            self.assertTrue(checkfn(fn, nd))

    def test_exp(self):
        def fn(x):
            return np.exp(x)
        self.assertTrue(checkfn(fn, [2]))

    def test_log(self):
        def fn(x):
            return np.log(x)
        self.assertTrue(checkfn(fn, [2]))

    def test_log1p(self):
        def fn(x):
            return np.log1p(x)
        self.assertTrue(checkfn(fn, [2]))

    def test_log10(self):
        def fn(x):
            return np.log10(x)
        self.assertTrue(checkfn(fn, [2]))

    def test_max(self):
        def fn(x):
            return np.max(x, 0)
        self.assertTrue(checkfn(fn, [2]))

    def test_min(self):
        def fn(x):
            return np.min(x, 0)
        self.assertTrue(checkfn(fn, [2]))

    def test_maximum(self):
        def fn(x, y):
            return np.maximum(x, y)
        self.assertTrue(checkfn(fn, [2, 2]))

    def test_minimum(self):
        def fn(x, y):
            return np.minimum(x, y)
        self.assertTrue(checkfn(fn, [2, 2]))

    def test_reshape(self):
        def fn(x, shape):
            return np.reshape(x, shape)
        self.assertTrue(checkfn(fn, [2], [2, 8]))

        def fn(x, shape1, shape2):
            return np.reshape(x, [shape1, shape2])
        self.assertTrue(checkfn(fn, [2], 2, 8))
        self.assertTrue(checkfn(fn, [2], 2, -1))
        self.assertTrue(checkfn(lambda x: np.reshape(x, x.shape), [2]))
        self.assertTrue(checkfn(
            lambda x: np.reshape(x, (x.shape[0], x.shape[1])), [2]))

    def test_sum(self):
        self.assertTrue(checkfn(lambda x: np.sum(x), [2]))
        self.assertTrue(checkfn(lambda x: np.sum(x, 1), [2]))
        self.assertTrue(checkfn(lambda x: np.sum(x, axis=1), [2]))
        self.assertTrue(checkfn(lambda x: np.sum(x, axis=1), [2]))
        self.assertTrue(checkfn(lambda x: np.sum(x, axis=None), [2]))
        self.assertTrue(checkfn(lambda x, a: np.sum(x, a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: np.sum(x, a), [2], None))
        self.assertTrue(checkfn(lambda x, a: np.sum(x, axis=a), [2], 0))

    def test_sqrt(self):
        def fn(x):
            return np.sqrt(x)
        self.assertTrue(checkfn(fn, [2]))

    def test_tanh(self):
        def fn(x):
            return np.tanh(x)
        self.assertTrue(checkfn(fn, [2]))

    def test_transpose(self):
        self.assertTrue(checkfn(lambda x: np.transpose(x), [2]))
        self.assertTrue(checkfn(lambda x: np.transpose(x, (0, 1)), [2]))
        self.assertTrue(checkfn(lambda x, a: np.transpose(x, a), [2], (0, 1)))
        self.assertTrue(checkfn(
            lambda x, a0, a1: np.transpose(x, (a0, a1)), [2], 0, 1))

    def test_zeros_like(self):
        def fn(x):
            return np.zeros_like(x)
        self.assertTrue(checkfn(fn, [2]))

    def test_astype(self):
        self.assertTrue(checkfn(lambda x: x.astype('float32'), [2]))

    def test_astype_numpy_class(self):
        self.assertTrue(checkfn(lambda x: x.astype(np.float32), [2]))

    def test_cast(self):
        self.assertTrue(checkfn(lambda x: int(x), [0]))
        self.assertTrue(checkfn(lambda x: float(x), [0]))
        self.assertTrue(checkfn(lambda x: bool(x), [0]))
        self.assertTrue(checkfn(lambda x: np.float_(x), [2]))
        self.assertTrue(checkfn(lambda x: np.float32(x), [2]))
        self.assertTrue(checkfn(lambda x: np.float64(x), [2]))
        self.assertTrue(checkfn(lambda x: np.int_(x), [2]))
        self.assertTrue(checkfn(lambda x: np.int16(x), [2]))
        self.assertTrue(checkfn(lambda x: np.bool_(x), [2]))
        self.assertTrue(checkfn(lambda x: np.bool(x), [0]))

    def test_alloc(self):
        self.assertTrue(checkfn(lambda: np.ones(5), []))
        self.assertTrue(checkfn(lambda: np.ones((2, 5)), []))
        self.assertTrue(checkfn(lambda x: np.ones(x.shape), [0]))
        self.assertTrue(checkfn(lambda x: np.ones(x.shape), [1]))
        self.assertTrue(checkfn(lambda x: np.ones(x.shape), [2]))

    def test_sort(self):
        self.assertTrue(checkfn(lambda x: np.sort(x), [2]))
        self.assertTrue(checkfn(lambda x: np.sort(x, 0), [2]))

    def test_concatenate(self):
        self.assertTrue(checkfn(lambda x, y: np.vstack((x, y)), [2, 2]))
        self.assertTrue(checkfn(lambda x, y: np.hstack((x, y)), [2, 2]))

    def test_axis(self):
        def f(x, axis=1):
            return np.std(x, axis=axis)
        self.assertTrue(checkfn(f, [2]))


class RandomNumbers(unittest.TestCase):
    def check_random(self, fn, *args, **kwargs):
        context.reset()
        F = context.recompile(fn)
        result1 = F(*args, **kwargs).eval()
        result2 = F(*args, **kwargs).eval()
        return np.allclose(result1, result2)

    def test_random(self):
        self.assertFalse(
            self.check_random(lambda: np.random.random((10, 10))))
        self.assertFalse(
            self.check_random(lambda s: np.random.random(s), 10.0))
        self.assertFalse(
            self.check_random(lambda s: np.random.random(s), (10, 10)))

    def test_random_shape(self):
        self.assertFalse(
            self.check_random(
                lambda x: np.random.random(x.shape), np.ones((10, 10))))

    def test_random_binomial(self):
        self.assertFalse(
            self.check_random(lambda: np.random.binomial(1, .5, (10, 10))))
        self.assertFalse(
            self.check_random(lambda s: np.random.binomial(1, .5, s), 10.0))
        self.assertFalse(self.check_random(
            lambda s: np.random.binomial(1, .5, s), (10, 10)))


class ArrayMethodsAttributes(unittest.TestCase):
    """
    Test for coverage of array methods and attributes
    """

    def test_argmax(self):
        self.assertTrue(checkfn(lambda x: x.argmax(), [2]))
        self.assertTrue(checkfn(lambda x: x.argmax(1), [2]))
        self.assertTrue(checkfn(lambda x: x.argmax(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.argmax(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.argmax(axis=a), [2], 0))

    def test_argmin(self):
        self.assertTrue(checkfn(lambda x: x.argmin(), [2]))
        self.assertTrue(checkfn(lambda x: x.argmin(1), [2]))
        self.assertTrue(checkfn(lambda x: x.argmin(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.argmin(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.argmin(axis=a), [2], 0))

    def test_argsort(self):
        self.assertTrue(checkfn(lambda x: x.argsort(), [2]))
        self.assertTrue(checkfn(lambda x: x.argsort(1), [2]))
        self.assertTrue(checkfn(lambda x: x.argsort(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.argsort(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.argsort(axis=a), [2], 0))

    def test_clip(self):
        def fn(x, a, b):
            return x.clip(a, b)
        self.assertTrue(checkfn(fn, [2], .4, .45))

    def test_conj(self):
        def fn(x):
            return x.conj()
        self.assertTrue(checkfn(fn, [2]))

    def test_conjugate(self):
        def fn(x):
            return x.conjugate()
        self.assertTrue(checkfn(fn, [2]))

    def test_copy(self):
        def fn(x):
            return x.copy()
        self.assertTrue(checkfn(fn, [2]))

    def test_diagonal(self):
        def fn(x):
            return x.diagonal()
        self.assertTrue(checkfn(fn, [2]))

    def test_dot(self):
        def fn(x, y):
            return x.dot(y)
        self.assertTrue(checkfn(fn, [2, 2]))
        self.assertTrue(checkfn(fn, [1, 2]))

    def test_imag(self):
        def fn(x):
            return x.imag
        self.assertTrue(checkfn(fn, [2]))

    def test_flatten(self):
        def fn(x):
            return x.flatten()
        self.assertTrue(checkfn(fn, [2]))

    def test_max(self):
        self.assertTrue(checkfn(lambda x: x.max(), [2]))
        self.assertTrue(checkfn(lambda x: x.max(1), [2]))
        self.assertTrue(checkfn(lambda x: x.max(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.max(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.max(axis=a), [2], 0))

    def test_mean(self):
        self.assertTrue(checkfn(lambda x: x.mean(), [2]))
        self.assertTrue(checkfn(lambda x: x.mean(1), [2]))
        self.assertTrue(checkfn(lambda x: x.mean(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.mean(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.mean(axis=a), [2], 0))

    def test_min(self):
        self.assertTrue(checkfn(lambda x: x.min(), [2]))
        self.assertTrue(checkfn(lambda x: x.min(1), [2]))
        self.assertTrue(checkfn(lambda x: x.min(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.min(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.min(axis=a), [2], 0))

    def test_prod(self):
        self.assertTrue(checkfn(lambda x: x.prod(), [2]))
        self.assertTrue(checkfn(lambda x: x.prod(1), [2]))
        self.assertTrue(checkfn(lambda x: x.prod(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.prod(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.prod(axis=a), [2], 0))

    def test_ravel(self):
        def fn(x):
            return x.ravel()
        self.assertTrue(checkfn(fn, [2]))

    def test_repeat(self):
        def fn(x, repeats):
            return x.repeat(repeats, axis=1)
        self.assertTrue(checkfn(fn, [2], 5))

    def test_real(self):
        def fn(x):
            return x.real
        self.assertTrue(checkfn(fn, [2]))

    def test_reshape(self):
        def fn(x, shape):
            return x.reshape(shape)
        self.assertTrue(checkfn(fn, [2], [2, 8]))

        def fn(x, s1, s2):
            return x.reshape(s1, s2)
        self.assertTrue(checkfn(fn, [2], 2, 8))
        self.assertTrue(checkfn(fn, [2], 2, -1))

        def fn(x):
            return x.reshape(2, 8)
        self.assertTrue(checkfn(fn, [2]))

    def test_sort(self):
        def fn(x):
            x.sort()
            return x
        self.assertRaises(ValueError, checkfn, fn, [2])

        def fn(x):
            x.sort(1)
            return x
        self.assertRaises(ValueError, checkfn, fn, [2])

        def fn(x):
            x.sort(axis=1)
            return x
        self.assertRaises(ValueError, checkfn, fn, [2])

        def fn(x, a):
            x.sort(a)
            return x
        self.assertRaises(ValueError, checkfn, fn, [2], 0)

    def test_sum(self):
        self.assertTrue(checkfn(lambda x: x.sum(), [2]))
        self.assertTrue(checkfn(lambda x: x.sum(1), [2]))
        self.assertTrue(checkfn(lambda x: x.sum(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.sum(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.sum(axis=a), [2], 0))

    def test_swapaxes(self):
        def fn(x, a1, a2):
            return x.swapaxes(a1, a2)
        self.assertTrue(checkfn(fn, [2], 0, 1))

    def test_astype(self):
        self.assertTrue(checkfn(lambda x: x.astype('int8'), [2]))
        self.assertTrue(checkfn(lambda x: x.astype('float32'), [2]))
        self.assertTrue(checkfn(lambda x: x.astype(np.float32), [2]))
        self.assertTrue(checkfn(lambda x: x.astype(dtype='float32'), [2]))
        self.assertTrue(checkfn(lambda x: x.astype(dtype=np.float32), [2]))

    def test_std(self):
        self.assertTrue(checkfn(lambda x: x.std(), [2]))
        self.assertTrue(checkfn(lambda x: x.std(1), [2]))
        self.assertTrue(checkfn(lambda x: x.std(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.std(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.std(axis=a), [2], 0))

    def test_size(self):
        self.assertTrue(checkfn(lambda x: np.arange(x.size), [1]))
        self.assertTrue(checkfn(lambda x: np.arange(x.size), [2]))

    def test_T(self):
        def fn(x):
            return x.T
        self.assertTrue(checkfn(fn, [1]))
        self.assertTrue(checkfn(fn, [2]))

    def test_transpose(self):
        def fn(x):
            return x.transpose()
        self.assertTrue(checkfn(fn, [1]))
        self.assertTrue(checkfn(fn, [2]))

    def test_var(self):
        self.assertTrue(checkfn(lambda x: x.var(), [2]))
        self.assertTrue(checkfn(lambda x: x.var(1), [2]))
        self.assertTrue(checkfn(lambda x: x.var(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.var(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.var(axis=a), [2], 0))


class Namespaces(unittest.TestCase):

    def test_global(self):
        x = np.ones((3, 4))

        def f():
            return x.swapaxes(0, 1)
        self.assertTrue(checkfn(f, []))

    def test_nested_functions(self):
        def g(x):
            def h(x):
                return x.swapaxes(1, 0)
            return h(x)

        def f(x):
            return g(x)

        self.assertTrue(checkfn(f, [2]))

    def test_define_class(self):
        """
        This fails due to shadowing of s (and then not being to set values)
        """
        def f():
            class StringAttr(object):
                def __init__(self):
                    self.s = "string"
            S = StringAttr()
            def f2(**kwargs):
                if kwargs['string'] == 5:
                    return 1
                else:
                    return 0
            return f2(**{S.s: 5})

        self.assertTrue(checkfn(f, []))

    def test_freevars(self):
        class Test(object):
            def __init__(self):
                self.x = np.arange(5.) - 10.0

            def getx(self):
                return self.x

        t = Test()

        def f(x):
            return np.dot(x, t.x)

        x = np.arange(5.)
        self.assertTrue(checkfn(f, [], x))
        self.assertTrue(id(t.x) in context.sym_vars)


class ArraySubscripts(unittest.TestCase):

    def test_indexing(self):
        self.assertTrue(checkfn(lambda x: x[2], [1]))
        self.assertTrue(checkfn(lambda x: x[-2], [1]))
        self.assertTrue(checkfn(lambda x: x[2], [2]))
        self.assertTrue(checkfn(lambda x: x[-2], [2]))
        self.assertTrue(checkfn(lambda x: x[2, 2], [2]))
        self.assertTrue(checkfn(lambda x: x[-2, -2], [2]))

    def test_adv_index(self):
        self.assertTrue(checkfn(lambda x: x[[3, 2, 1], [1, 2, 3]], [2]))
        self.assertTrue(checkfn(lambda x: x[x > .5], [2]))
        self.assertTrue(checkfn(lambda x: x[(x > .1) * (x < .5)], [2]))
        self.assertTrue(checkfn(lambda x: x[[2, 3], 1:], [2]))

    # @unittest.expectedFailure
    # def test_adv_index_known_failures(self):
        # self.assertTrue(checkfn(lambda x: x[1:, x > .5], [2]))
        # self.assertTrue(checkfn(lambda x: x[x > .5, 1:], [2]))

    def test_slicing(self):
        # SLICE+0
        self.assertTrue(checkfn(lambda x: x[:], [1]))
        self.assertTrue(checkfn(lambda x: x[:], [2]))

        # SLICE+1
        self.assertTrue(checkfn(lambda x: x[1:], [1]))
        self.assertTrue(checkfn(lambda x: x[-2:], [1]))
        self.assertTrue(checkfn(lambda x: x[1:, 1:], [2]))
        self.assertTrue(checkfn(lambda x: x[-2:, -2:], [2]))

        # SLICE+2
        self.assertTrue(checkfn(lambda x: x[:2], [1]))
        self.assertTrue(checkfn(lambda x: x[:-2], [1]))
        self.assertTrue(checkfn(lambda x: x[:2, :2], [2]))
        self.assertTrue(checkfn(lambda x: x[:-2, :-2], [2]))

        # SLICE+3
        self.assertTrue(checkfn(lambda x: x[1:3], [1]))
        self.assertTrue(checkfn(lambda x: x[-3:-1], [1]))
        self.assertTrue(checkfn(lambda x: x[1:3, 1:3], [2]))
        self.assertTrue(checkfn(lambda x: x[-3:-1, -3:-1], [2]))

    def test_index_and_slice(self):
        self.assertTrue(checkfn(lambda x: x[1:3, 2], [2]))

    def test_index_assign(self):
        def f():
            x = np.ones((3, 4))
            x[2] = 100
            return x
        self.assertTrue(checkfn(f, []))

        def f():
            x = np.ones((3, 4))
            x[2, 2] = 100
            return x
        self.assertTrue(checkfn(f, []))

        def f():
            x = np.ones((3, 4))
            x[2, 2] += 100
            return x
        self.assertTrue(checkfn(f, []))

        def f(x):
            x[2, 2] = 100
            return x
        self.assertTrue(checkfn(f, [2]))

        def f(x):
            x[2, 2] += 100
            return x
        self.assertTrue(checkfn(f, [2]))

    def test_slice_assign(self):
        def f():
            x = np.ones((3, 4))
            x[2:3] = 100
            return x
        self.assertTrue(checkfn(f, []))

        def f():
            x = np.ones((3, 4))
            x[2:3, 2:3] += 100
            return x
        self.assertTrue(checkfn(f, []))

        def f(x):
            x[2:3, 2:3] += 100
            return x
        self.assertTrue(checkfn(f, [2]))

    def test_store_slice(self):
        # STORE_SLICE+0
        def f(x):
            x[:] = 5
            x[:] += 5
            return x
        self.assertTrue(checkfn(f, [1]))
        self.assertTrue(checkfn(f, [2]))

        # STORE_SLICE+1
        def f(x):
            x[2:] = 5
            x[-2:] += 5
            return x

        def f2(x):
            x[2:, 2:] = 5
            x[-2:, -2:] += 5
            return x
        self.assertTrue(checkfn(f, [1]))
        self.assertTrue(checkfn(f, [2]))
        self.assertTrue(checkfn(f2, [2]))

        # STORE_SLICE+2
        def f(x):
            x[:2] = 5
            x[:-2] += 5
            return x

        def f2(x):
            x[:2, :2] = 5
            x[:-2, :-2] += 5
            return x
        self.assertTrue(checkfn(f, [1]))
        self.assertTrue(checkfn(f, [2]))
        self.assertTrue(checkfn(f2, [2]))

        # STORE_SLICE+3
        def f(x):
            x[1:3] = 5
            x[-3:-1] += 5
            return x

        def f2(x):
            x[1:3, 1:3] = 5
            x[-3:-1, -3:-1] += 5
            return x
        self.assertTrue(checkfn(f, [1]))
        self.assertTrue(checkfn(f, [2]))
        self.assertTrue(checkfn(f2, [2]))

    def test_array_assign(self):
        def f(x):
            o = np.ones((2, 3))
            x[1:3, 1:4] = o
            return x
        self.assertTrue(checkfn(f, [2]))

    def test_nested_assign(self):
        def f(x):
            x[2:4][1, 2] = 100
            return x
        self.assertTrue(checkfn(f, [2]))

        def f(x):
            x[2:4][1, 2] += 100
            return x
        self.assertTrue(checkfn(f, [2]))

        def f():
            d = {1: {2: 3}}
            d[1][2] = 4
            return d[1][2]
        self.assertTrue(checkfn(f, []))


class TestMethods(unittest.TestCase):
    def test_instance_method(self):
        class Test(object):
            def test(self, x):
                return x * 2

        t = Test()
        self.assertTrue(checkfn(t.test, [2]))

    def test_class_method(self):
        class Test(object):
            @classmethod
            def test(cls, x):
                return x * 2

        t = Test()
        self.assertTrue(checkfn(t.test, [2]))
        self.assertTrue(checkfn(Test.test, [2]))

    def test_static_method(self):
        class Test(object):
            @staticmethod
            def test(x):
                return x * 2

        t = Test()
        self.assertTrue(checkfn(t.test, [2]))
        self.assertTrue(checkfn(Test.test, [2]))


class NumberMethodsAttributes(unittest.TestCase):
    """
    Test for coverage of NumPy number methods and attributes
    """

    def test_reduce_method(self):
        self.assertTrue(checkfn(lambda x: np.dot(x, x).sum(), [1]))
        self.assertTrue(checkfn(lambda x: np.dot(x, x).mean(), [1]))


class Ops(unittest.TestCase):
    """
    test bytecode op coverage for misc cases
    """
    def test_DUP_TOP(self):
        def f(x):
            x[:] += 100
            return x
        self.assertTrue(checkfn(f, [2]))


class Collections(unittest.TestCase):

    def test_views(self):
        from collections import OrderedDict

        def f():
            d = {1: 2, 3: 4, 5: 6}
            return list(v for v in d.values())

        self.assertTrue(checkfn(f, []))

    def test_OrderedDict(self):
        from collections import OrderedDict

        o = OrderedDict(a=1, b=2, c=3)
        def f():
            x = 0
            for v in o.values():
                x += v
            return x

        self.assertTrue(checkfn(f, []))

class InferUpdates(unittest.TestCase):
    def test_assign_updates(self):
        c = autodiff.context.Context(infer_updates=False)
        c_upd = autodiff.context.Context(infer_updates=True)

        class Test:
            def __init__(self):
                self.reset()
            def reset(self):
                self.tmp = 0.0

        test = Test()

        def f(x):
            test.tmp = test.tmp + x
            return test.tmp

        F = c.recompile(f)
        F_upd = c_upd.recompile(f)

        inp = 5.0
        test.reset()
        out = F(inp)
        test.reset()
        out_upd = F_upd(inp)

        compiled = theano.function([], out, updates=c.updates)
        compiled_upd = theano.function([], out_upd, updates=c_upd.updates)

        self.assertTrue(np.allclose(compiled(), 5.0))
        self.assertTrue(np.allclose(compiled(), 5.0))
        self.assertTrue(np.allclose(compiled_upd(), 5.0))
        self.assertTrue(np.allclose(compiled_upd(), 10.0))

    def test_augassign_updates(self):
        c = autodiff.context.Context(infer_updates=False)
        c_upd = autodiff.context.Context(infer_updates=True)

        class Test:
            def __init__(self):
                self.reset()
            def reset(self):
                self.tmp = 0.0

        test = Test()

        def f(x):
            test.tmp += x
            return test.tmp

        F = c.recompile(f)
        F_upd = c_upd.recompile(f)

        inp = 5.0
        test.reset()
        out = F(inp)
        test.reset()
        out_upd = F_upd(inp)

        compiled = theano.function([], out, updates=c.updates)
        compiled_upd = theano.function([], out_upd, updates=c_upd.updates)

        self.assertTrue(np.allclose(compiled(), 5.0))
        self.assertTrue(np.allclose(compiled(), 5.0))
        self.assertTrue(np.allclose(compiled_upd(), 5.0))
        self.assertTrue(np.allclose(compiled_upd(), 10.0))
