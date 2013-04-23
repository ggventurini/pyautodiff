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
    assert np.allclose(theano_fn(*values), result)


class Test_NumPy_fn_to_Theano(unittest.TestCase):
    """
    Test for coverage of functions in np namespace
    """
    def test_abs(self):
        def fn1(x):
            return np.abs(x)

        def fn2(x):
            return abs(x)

        checkfn(fn1, [2])
        checkfn(fn2, [2])

    def test_dot(self):
        def fn(x, y):
            return np.dot(x, y)
        for nd in np.ndindex(*([3] * fn.func_code.co_argcount)):
            checkfn(fn, nd)

    def test_exp(self):
        def fn(x):
            return np.exp(x)
        checkfn(fn, [2])

    def test_log(self):
        def fn(x):
            return np.log(x)
        checkfn(fn, [2])

    def test_log1p(self):
        def fn(x):
            return np.log1p(x)
        checkfn(fn, [2])

    def test_log10(self):
        def fn(x):
            return np.log10(x)
        checkfn(fn, [2])

    def test_maximum(self):
        def fn(x):
            return np.maximum(x)
        checkfn(fn, [2])
        raise NotImplementedError("pass axes")

    def test_max(self):
        def fn1(x):
            return np.max(x)

        def fn2(x):
            return max(x)

        checkfn(fn1, [2])
        checkfn(fn2, [2])

    def test_minimum(self):
        def fn(x):
            return np.minimum(x)
        checkfn(fn, [2])
        raise NotImplementedError("pass axes")

    def test_min(self):
        def fn1(x):
            return np.min(x)

        def fn2(x):
            return min(x)

        checkfn(fn1, [2])
        checkfn(fn2, [2])

    def test_reshape(self):
        raise NotImplementedError("pass dim")

    def test_sum(self):
        def fn(x):
            return np.sum(x)
        checkfn(fn, [2])
        raise NotImplementedError("pass axes")

    def test_sqrt(self):
        def fn(x):
            return np.sqrt(x)
        checkfn(fn, [2])

    def test_tanh(self):
        def fn(x):
            return np.tanh(x)
        checkfn(fn, [2])

    def test_zeros_like(self):
        def fn(x):
            return np.zeros_like(x)
        checkfn(fn, [2])


class Test_NumPy_method_to_Theano(unittest.TestCase):
    """
    Test for coverage of array methods.
    """

    def test_copy(self):
        def fn(x):
            return x.copy()
        checkfn(fn, [2])

    def test_max(self):
        def fn(x):
            return x.max()
        checkfn(fn, [2])
        raise NotImplementedError("pass axes")

    def test_mean(self):
        def fn(x):
            return x.mean()
        checkfn(fn, [2])
        raise NotImplementedError("pass axes")

    def test_min(self):
        def fn(x):
            return x.min()
        checkfn(fn, [2])
        raise NotImplementedError("pass axes")

    def test_reshape(self):
        raise NotImplementedError("pass dim")

    def test_sum(self):
        def fn(x):
            return x.sum()
        checkfn(fn, [2])
        raise NotImplementedError("pass axes")

    def test_astype(self):
        def fn(x):
            return x.astype(np.int8)
        checkfn(fn, [2])

    def test_std(self):
        def fn(x):
            return x.std()
        checkfn(fn, [2])
        raise NotImplementedError("pass axes")


class Test_Compare_to_Theano(unittest.TestCase):
    """
    Test for coverage of operators
    """
    def test_gt(self):
        def fn(x, y):
            return x > y
        checkfn(fn, [2, 2])

    def test_lt(self):
        def fn(x, y):
            return x < y
        checkfn(fn, [2, 2])

    def test_ge(self):
        def fn(x, y):
            return x >= y
        checkfn(fn, [2, 2])

    def test_le(self):
        def fn(x, y):
            return x <= y
        checkfn(fn, [2, 2])

    def test_eq(self):
        def fn(x, y):
            return x == y
        checkfn(fn, [2, 2])

    def test_neq(self):
        def fn(x, y):
            return x != y
        checkfn(fn, [2, 2])

    def test_is(self):
        def fn(x, y):
            return x is x
        checkfn(fn, [2])
