import unittest
import numpy as np

from autodiff.decorators import function, gradient, hessian_vector


#========= Tests


class TestFunction(unittest.TestCase):
    def test_basic_fn(self):
        @function
        def fn(x):
            return x
        self.assertTrue(np.allclose(fn(3), 3))
        self.assertTrue(len(fn.cache) == 1)

    def test_fn_cache(self):
        @function
        def fn(x):
            return x

        self.assertTrue(np.allclose(fn(3), 3))

        # check that fn was cached
        self.assertTrue(len(fn.cache) == 1)

        # check that arg of new input dim was cached
        fn(np.ones(10))
        self.assertTrue(len(fn.cache) == 2)

        # check that another arg of same input dim was not cached
        fn(np.ones(10) + 15)
        self.assertTrue(len(fn.cache) == 2)


class TestGradient(unittest.TestCase):
    def test_basic_grad(self):
        @gradient
        def fn(x):
            return x
        self.assertTrue(np.allclose(fn(3), 1.0))

    def test_nonscalar_grad(self):
        @gradient
        def fn(x):
            return x
        self.assertRaises(TypeError, fn, np.ones(1))

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


class TestHV(unittest.TestCase):
    def test_hv_missing_vectors(self):
        @hessian_vector
        def fn(x):
            return x
        self.assertRaises(ValueError, fn, np.array([[1, 1]]))

    def test_hv_no_scalar(self):
        @hessian_vector
        def fn(x):
            return np.dot(x, x)
        x = np.ones((3, 3))
        self.assertRaises(TypeError, fn, x, _vectors=x[0])

    def test_hv(self):
        @hessian_vector
        def fn(x):
            return np.dot(x, x).sum()
        x = np.ones((3, 3))
        self.assertTrue(np.allclose(x * 6, fn(x, _vectors=x)))
        self.assertTrue(np.allclose(x * 2, fn(x[0], _vectors=x[0])))
