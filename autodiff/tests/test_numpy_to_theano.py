import unittest
import numpy as np
import theano


from autodiff.context import Context


def get_theano_fn(ctxt, xvals, yvals):
    if not isinstance(xvals, (list, tuple)):
        xvals = [xvals]
    if not isinstance(yvals, (list, tuple)):
        yvals = [yvals]
    sx = [ctxt.svars[id(x)] for x in xvals]
    vx = [x.type() for x in sx]
    givens = dict(zip(sx, vx))
    sy = [ctxt.svars[id(y)] for y in yvals]
    return theano.function(vx, sy, givens=givens)


def checkfn(fn, var_ndim, *args):
    """Given a function and a list of ndim for each input variable,
    get a result and compare it to the Theano result."""
    dim = [[4] * nd for nd in var_ndim]
    values = [np.random.random(d) for d in dim]
    context = Context()
    result = context.call(fn, tuple(values) + args)
    theano_fn = get_theano_fn(context, values, result)
    return np.allclose(theano_fn(*values), result)


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


class NumpyFns(unittest.TestCase):
    """
    Test for coverage of functions in np namespace
    """
    def test_all(self):
        def fn(x):
            return np.all(x>.5)
        self.assertTrue(checkfn(fn, [2]))

    def test_any(self):
        def fn(x):
            return np.any(x>.5)
        self.assertTrue(checkfn(fn, [2]))

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
        self.assertTrue(checkfn(fn, [2], [2, -1]))

    def test_sum(self):
        self.assertTrue(checkfn(lambda x: np.sum(x), [2]))
        self.assertTrue(checkfn(lambda x: np.sum(x, 1), [2]))
        self.assertTrue(checkfn(lambda x: np.sum(x, axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: np.sum(x, a), [2], None))
        self.assertTrue(checkfn(lambda x, a: np.sum(x, a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: np.sum(x, axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: np.sum(x, axis=a), [2], 0))

        def fn(x, axis=None):
            return np.sum(x, axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))

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

class ArrayMethodsAttributes(unittest.TestCase):
    """
    Test for coverage of array methods and attributes
    """

    def test_argmax(self):
        self.assertTrue(checkfn(lambda x: x.argmax(), [2]))
        self.assertTrue(checkfn(lambda x: x.argmax(1), [2]))
        self.assertTrue(checkfn(lambda x: x.argmax(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.argmax(a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.argmax(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.argmax(axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.argmax(axis=a), [2], 0))

        def fn(x, axis=None):
            return x.argmax(axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))

    def test_argmin(self):
        self.assertTrue(checkfn(lambda x: x.argmin(), [2]))
        self.assertTrue(checkfn(lambda x: x.argmin(1), [2]))
        self.assertTrue(checkfn(lambda x: x.argmin(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.argmin(a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.argmin(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.argmin(axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.argmin(axis=a), [2], 0))

        def fn(x, axis=None):
            return x.argmin(axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))

    def test_argsort(self):
        self.assertTrue(checkfn(lambda x: x.argsort(), [2]))
        self.assertTrue(checkfn(lambda x: x.argsort(1), [2]))
        self.assertTrue(checkfn(lambda x: x.argsort(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.argsort(a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.argsort(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.argsort(axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.argsort(axis=a), [2], 0))

        def fn(x, axis=None):
            return x.argsort(axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))

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
        self.assertTrue(checkfn(lambda x, a: x.max(a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.max(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.max(axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.max(axis=a), [2], 0))

        def fn(x, axis=None):
            return x.max(axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))

    def test_mean(self):
        self.assertTrue(checkfn(lambda x: x.mean(), [2]))
        self.assertTrue(checkfn(lambda x: x.mean(1), [2]))
        self.assertTrue(checkfn(lambda x: x.mean(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.mean(a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.mean(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.mean(axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.mean(axis=a), [2], 0))

        def fn(x, axis=None):
            return x.mean(axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))

    def test_min(self):
        self.assertTrue(checkfn(lambda x: x.min(), [2]))
        self.assertTrue(checkfn(lambda x: x.min(1), [2]))
        self.assertTrue(checkfn(lambda x: x.min(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.min(a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.min(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.min(axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.min(axis=a), [2], 0))

        def fn(x, axis=None):
            return x.min(axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))

    def test_prod(self):
        self.assertTrue(checkfn(lambda x: x.prod(), [2]))
        self.assertTrue(checkfn(lambda x: x.prod(1), [2]))
        self.assertTrue(checkfn(lambda x: x.prod(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.prod(a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.prod(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.prod(axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.prod(axis=a), [2], 0))

        def fn(x, axis=None):
            return x.prod(axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))

    def test_ravel(self):
        def fn(x):
            return x.ravel()
        self.assertTrue(checkfn(fn, [2]))

    def test_repeat(self):
        def fn(x, repeats, axis=None):
            return x.repeat(repeats, axis=axis)
        self.assertTrue(checkfn(fn, [2], 5))
        self.assertTrue(checkfn(fn, [2], 5, 0))
        self.assertTrue(checkfn(fn, [2], 5, 1))

    def test_real(self):
        def fn(x):
            return x.real
        self.assertTrue(checkfn(fn, [2]))

    def test_reshape(self):
        def fn1(x, shape):
            return x.reshape(shape)
        def fn2(x, s1, s2):
            return x.reshape(s1, s2)
        self.assertTrue(checkfn(fn1, [2], [2, 8]))
        self.assertTrue(checkfn(fn1, [2], [2, -1]))
        self.assertTrue(checkfn(fn2, [2], 2, 8))
        self.assertTrue(checkfn(fn2, [2], 2, -1))

    def test_sort(self):
        self.assertTrue(checkfn(lambda x: x.sort(), [2]))
        self.assertTrue(checkfn(lambda x: x.sort(1), [2]))
        self.assertTrue(checkfn(lambda x: x.sort(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.sort(a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.sort(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.sort(axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.sort(axis=a), [2], 0))

        def fn(x, axis=None):
            return x.sort(axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))

    def test_sum(self):
        self.assertTrue(checkfn(lambda x: x.sum(), [2]))
        self.assertTrue(checkfn(lambda x: x.sum(1), [2]))
        self.assertTrue(checkfn(lambda x: x.sum(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.sum(a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.sum(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.sum(axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.sum(axis=a), [2], 0))

        def fn(x, axis=None):
            return x.sum(axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))

    def test_swapaxes(self):
        def fn(x, a1, a2):
            return x.swapaxes(a1, a2)
        self.assertTrue(checkfn(fn, [2], 0, 1))

    def test_astype(self):
        def fn(x):
            return x.astype('int8')
        self.assertTrue(checkfn(fn, [2]))

    def test_std(self):
        self.assertTrue(checkfn(lambda x: x.std(), [2]))
        self.assertTrue(checkfn(lambda x: x.std(1), [2]))
        self.assertTrue(checkfn(lambda x: x.std(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.std(a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.std(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.std(axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.std(axis=a), [2], 0))

        def fn(x, axis=None):
            return x.std(axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))

    def test_T(self):
        def fn(x):
            return x.T
        self.assertTrue(checkfn(fn, [1]))
        self.assertTrue(checkfn(fn, [2]))

    @unittest.skip('skip trace')
    def test_trace(self):
        def fn(x):
            pass

    def test_transpose(self):
        def fn(x):
            return x.transpose()
        self.assertTrue(checkfn(fn, [1]))
        self.assertTrue(checkfn(fn, [2]))

    def test_var(self):
        self.assertTrue(checkfn(lambda x: x.var(), [2]))
        self.assertTrue(checkfn(lambda x: x.var(1), [2]))
        self.assertTrue(checkfn(lambda x: x.var(axis=1), [2]))
        self.assertTrue(checkfn(lambda x, a: x.var(a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.var(a), [2], 0))
        self.assertTrue(checkfn(lambda x, a: x.var(axis=a), [2], None))
        self.assertTrue(checkfn(lambda x, a: x.var(axis=a), [2], 0))

        def fn(x, axis=None):
            return x.var(axis=axis)
        self.assertTrue(checkfn(fn, [2]))
        self.assertTrue(checkfn(fn, [2], 0))


class IndexSlice(unittest.TestCase):
    """
    Test for coverage of operators
    """
    def test_index(self):
        self.assertTrue(checkfn(lambda x: x[1], [1]))
        self.assertTrue(checkfn(lambda x: x[-1], [1]))
        self.assertTrue(checkfn(lambda x: x[1, 1], [2]))
        self.assertTrue(checkfn(lambda x: x[-1, -1], [2]))

    def test_slice(self):
        self.assertTrue(checkfn(lambda x: x[1:], [1]))
        self.assertTrue(checkfn(lambda x: x[-2:], [1]))
        self.assertTrue(checkfn(lambda x: x[1:, 1:], [2]))
        self.assertTrue(checkfn(lambda x: x[-2:, -2:], [2]))

        self.assertTrue(checkfn(lambda x: x[:2], [1]))
        self.assertTrue(checkfn(lambda x: x[:-2], [1]))
        self.assertTrue(checkfn(lambda x: x[:2, :2], [2]))
        self.assertTrue(checkfn(lambda x: x[:-2, :-2], [2]))

        self.assertTrue(checkfn(lambda x: x[1:3], [1]))
        self.assertTrue(checkfn(lambda x: x[-3:-1], [1]))
        self.assertTrue(checkfn(lambda x: x[1:3, 1:3], [2]))
        self.assertTrue(checkfn(lambda x: x[-3:-1, -3:-1], [2]))

    def test_adv_index(self):
        self.assertTrue(checkfn(lambda x: x[[3,2,1], [1,2,3]], [2]))
        self.assertTrue(checkfn(lambda x: x[x > .5], [2]))

    @unittest.expectedFailure
    def test_adv_index_known_failures(self):
        self.assertTrue(checkfn(lambda x: x[1:, x > .5], [2]))
        self.assertTrue(checkfn(lambda x: x[x > .5, 1:], [2]))
        self.assertTrue(checkfn(lambda x: x[[2, 3], 1:], [2]))

