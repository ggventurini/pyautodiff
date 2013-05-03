import unittest
import numpy as np

from autodiff.constant import Constant
from autodiff.symbolic import Function


def check(fn, *args, **kwargs):
    F = Function(fn)
    py_result = fn(*args, **kwargs)
    sym_result = F(*args, **kwargs)
    return np.allclose(py_result, sym_result)


class TestConstant(unittest.TestCase):
    def test_range(self):
        def f(x):
            for i in range(3):
                x = x + x
            return x
        self.assertTrue(check(f, 1))

        def f(x):
            for i in range(x):
                x = x + x
            return x
        self.assertRaises(NotImplementedError, check, f, 1)

        def f(x, r):
            for i in range(r):
                x = x + x
            return x
        self.assertRaises(NotImplementedError, check, f, np.ones(3), 3)

        def f(x):
            for i in range(Constant(3)):
                x = x + x
            return x
        self.assertTrue(check(f, 1))

        def f(x):
            for i in range(Constant(x)):
                x = x + x
            return x
        self.assertTrue(check(f, 1))

        def f(x, r):
            for i in range(Constant(r)):
                x = x + x
            return x
        self.assertTrue(check(f, np.ones(3), 3))

    def test_sum(self):
        def f(x):
            return x.sum(1)
        self.assertTrue(check(f, np.ones((3, 4))))

        def f(x):
            a = 1
            return x.sum(a)
        self.assertRaises(TypeError, check, f, np.ones((3, 4)))

        def f(x, a):
            return x.sum(a)
        self.assertRaises(TypeError, check, f, np.ones((3, 4)), 1)

        def f(x):
            return x.sum(Constant(1))
        self.assertTrue(check(f, np.ones((3, 4))))

        def f(x):
            a = 1
            return x.sum(Constant(a))
        self.assertTrue(check(f, np.ones((3, 4))))

        def f(x, a):
            return x.sum(Constant(a))
        self.assertTrue(check(f, np.ones((3, 4)), 1))
