import unittest
import numpy as np
import theano
import theano.tensor as T
import copy

from autodiff.context import Context
import autodiff.context as c


context = Context()


def checkfn(f, var_ndim, *args, **kwargs):
    override = kwargs.pop('override', None)
    dim = [[4] * nd for nd in var_ndim]
    values = tuple([np.random.random(d) for d in dim])
    # make shallow copies to avoid inplace corruption
    sym_values = tuple(copy.copy(v) for v in values)
    sym_args = tuple(copy.copy(a) for a in args)

    F = context.recompile(f)

    py_result = override or f(*(values + args))
    sym_result = F(*(sym_values + args)).eval()

    return np.allclose(py_result, sym_result)


class GarbageCollection(unittest.TestCase):
    # make sure shadowed variables aren't garbage-collected
    # so their id's do not get reused
    def test_gc(self):
        def f(x, y):
            return [x, y]

        F = context.recompile(f)
        assert F(3, 4)[1].eval() == 4


class AugAssign(unittest.TestCase):
    # AugAssign doesn't pick up the assignment of shadowed variables,
    # so they don't get updated. Make sure that the shadow is explicitly
    # updated.
    def test_aug_shadowing(self):
        def f(x):
            a = x
            x += 1
            return x

        F = context.recompile(f)
        assert F(1).eval() == 2


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
        for nd in np.ndindex(*([3] * fn.func_code.co_argcount)):
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
        self.assertRaises(TypeError, checkfn,
                          lambda x, a: np.sum(x, a), [2], 0)
        self.assertRaises(TypeError, checkfn,
                          lambda x, a: np.sum(x, axis=a), [2], 0)

    def test_sqrt(self):
        def fn(x):
            return np.sqrt(x)
        self.assertTrue(checkfn(fn, [2]))

    def test_tanh(self):
        def fn(x):
            return np.tanh(x)
        self.assertTrue(checkfn(fn, [2]))

    def test_zeros_like(self):
        def fn(x):
            return np.zeros_like(x)
        self.assertTrue(checkfn(fn, [2]))

    def test_astype(self):
        self.assertTrue(checkfn(lambda x: x.astype('float32'), [2]))

    @unittest.expectedFailure
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



