import unittest
import numpy as np

from autodiff.decorators import function, gradient


#========= Tests


class TestFunction(unittest.TestCase):
    def test_basic_fn(self):
        @function
        def fn(x):
            return x
        self.assertTrue(np.allclose(fn(3), 3))
        self.assertTrue(0 in fn._cache)


class TestGradient(unittest.TestCase):
    def test_basic_grad(self):
        @gradient
        def fn(x):
            return x
        self.assertRaises(TypeError, fn, np.ones(1))
        self.assertTrue(np.allclose(fn(3), 1.0))
        self.assertTrue(0 in fn._cache)

    def test_grad_wrt(self):
        @gradient(wrt='x')
        def f(x, y):
            return x * y
        self.assertTrue(np.allclose(f(3.0, 5.0), 5.0))

        @gradient(wrt=('x', 'y'))
        def f(x, y):
            return x * y
        self.assertTrue(np.allclose(f(3.0, 5.0), [5.0, 3.0]))

        @gradient(wrt=('y', 'x'))
        def f(x, y):
            return x * y
        self.assertTrue(np.allclose(f(3.0, 5.0), [3.0, 5.0]))

        @gradient()
        def f(x, y):
            return x * y
        self.assertTrue(np.allclose(f(3.0, 5.0), [5.0, 3.0]))

        a = np.array(3.0)
        b = np.array(5.0)

        @gradient(wrt=a)
        def f(x, y):
            return x * y
        self.assertTrue(np.allclose(f(a, 5.0), 5.0))

        @gradient(wrt=b)
        def f(x, y):
            return x * y
        self.assertRaises(ValueError, f, a, 5.0)
        self.assertTrue(np.allclose(f(a, b), 3.0))
        self.assertTrue(np.allclose(f(3.0, b), 3.0))

        @gradient(wrt=(a, b))
        def f(x, y):
            return x * y
        self.assertRaises(ValueError, f, a, 5.0)
        self.assertTrue(np.allclose(f(a, b), [5.0, 3.0]))
