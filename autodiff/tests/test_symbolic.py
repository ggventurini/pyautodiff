import unittest
import numpy as np
import theano

from autodiff.symbolic import Symbolic, Function, Gradient


def checkfn(F, *args, **kwargs):
    return np.allclose(F(*args, **kwargs), F.pyfn(*args, **kwargs))

#========= Tests

class TestFunction(unittest.TestCase):
    def test_fn_signatures(self):
        # single arg, no default
        def fn(x):
            return x
        f = Function(fn)
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, a=2)
        self.assertTrue(checkfn(f, 2))
        self.assertTrue(checkfn(f, x=2))

        # multiple args, no default
        def fn(x, y):
            return x * y
        f = Function(fn)
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, 2)
        self.assertRaises(TypeError, f, a=2, b=2)
        self.assertTrue(checkfn(f, 2, 3))
        self.assertTrue(checkfn(f, y=4, x=5))

        # var args, no default
        def fn(x, y, *z):
            return x * y * sum(z)
        f = Function(fn)
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, 2)
        self.assertRaises(TypeError, f, a=2, b=2)
        self.assertTrue(checkfn(f, 2, 3))
        self.assertTrue(checkfn(f, y=4, x=5))
        self.assertTrue(checkfn(f, 2, 3, 4))
        self.assertTrue(checkfn(f, 2, 3, 4, 5))


class TestGradient(unittest.TestCase):

    def test_scalar_fn(self):
        g = Gradient(lambda x : x ** 2)
        self.assertTrue(np.allclose(g(np.array(2.0)), 4))

    def test_int_fn(self):
        """ FAILS! """
        g = Gradient(lambda x : x ** 2)
        self.assertTrue(np.allclose(g(2), 4))
