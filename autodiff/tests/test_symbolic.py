import unittest
import numpy as np

from autodiff.symbolic import Function, Gradient


def checkfn(symF, *args, **kwargs):
    py_result = symF.pyfn(*args, **kwargs)
    ad_result = symF(*args, **kwargs)
    return np.allclose(ad_result, py_result)


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
        self.assertRaises(Exception, f, 2, 3)
        self.assertTrue(checkfn(f, 2, 3, 4))
        self.assertTrue(checkfn(f, 2, 3, 4, 5))

        # make sure function recompiles for different numbers of varargs
        f = Function(fn)
        self.assertTrue(checkfn(f, 2, 3, 4, 5, 6))
        self.assertTrue(checkfn(f, 2, 3, 4))
        self.assertTrue(checkfn(f, 2, 3, 4, 5))

        # multiple args, one default
        def fn(x, y=2):
            return x * y
        f = Function(fn)
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, y=3)
        self.assertTrue(checkfn(f, 2))
        self.assertTrue(checkfn(f, 2, 3))
        self.assertTrue(checkfn(f, y=4, x=5))
        self.assertTrue(checkfn(f, x=5))

        # multiple args, all default
        def fn(x=1, y=2):
            return x * y
        f = Function(fn)
        self.assertTrue(checkfn(f))
        self.assertTrue(checkfn(f, 1))
        self.assertTrue(checkfn(f, 1, 2))
        self.assertTrue(checkfn(f, y=2, x=1))
        self.assertTrue(checkfn(f, x=5))
        self.assertTrue(checkfn(f, y=5))

        # multiple var args, all default
        def fn(x=1, y=2, *z):
            return x * y * sum(z)
        f = Function(fn)
        self.assertRaises(Exception, f)
        self.assertRaises(Exception, f, 1)
        self.assertRaises(Exception, f, 1, 2)
        self.assertTrue(checkfn(f, 1, 2, 3))
        self.assertTrue(checkfn(f, 1, 2, 3, 4))

        # kwargs
        def fn(**kwargs):
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            return x * y * z
        f = Function(fn)
        self.assertRaises(KeyError, f)
        self.assertRaises(TypeError, f, 1)
        self.assertTrue(checkfn(f, x=1, y=2, z=3))

        # varargs and kwargs
        def fn(a, *b, **kwargs):
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            return x * y * z
        f = Function(fn)
        self.assertRaises(TypeError, f)
        self.assertRaises(KeyError, f, 1)
        self.assertRaises(TypeError, f, x=1, y=2, z=3)
        self.assertTrue(checkfn(f, 1, x=1, y=2, z=3))
        self.assertTrue(checkfn(f, 1, 2, 3, x=1, y=2, z=3))

        # varargs and kwargs, use varargs
        def fn(a, *b, **kwargs):
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            return x * y * z * b[0]
        f = Function(fn)
        self.assertTrue(checkfn(f, 1, 2, x=1, y=2, z=3))
        self.assertTrue(checkfn(f, 1, 2, 3, x=1, y=2, z=3))

    def test_fn_constants(self):
        # access constant array
        def fn(x):
            return np.dot(x, np.ones((3, 4)))
        f = Function(fn)
        self.assertTrue(checkfn(f, np.ones((2, 3))))


class TestGradient(unittest.TestCase):
    def test_simple_gradients(self):
        # straightforward gradient
        g = Gradient(lambda x: x ** 2)
        self.assertTrue(np.allclose(g(3.0), 6))

        # int args may result in 0 gradients
        self.assertTrue(np.allclose(g(3), 0))

        # gradient of two arguments
        fn = lambda x, y: x * y
        g = Gradient(fn)
        self.assertTrue(np.allclose(g(3.0, 5.0), [5.0, 3.0]))

    def test_wrt(self):
        fn = lambda x, y: x * y

        # function of two arguments, gradient of one
        g = Gradient(fn, wrt='x')
        self.assertTrue(np.allclose(g(3.0, 5.0), 5.0))

        g = Gradient(fn, wrt='y')
        self.assertTrue(np.allclose(g(3.0, 5.0), 3.0))

        # test object tracking for wrt
        a = 3.0
        b = 5.0
        g = Gradient(fn, wrt=[a, b])
        self.assertTrue(np.allclose(g(a, b), [b, a]))

        g = Gradient(fn, wrt=a)
        self.assertTrue(np.allclose(g(a, b), b))

        g = Gradient(fn, wrt=b)
        self.assertTrue(np.allclose(g(a, b), a))
