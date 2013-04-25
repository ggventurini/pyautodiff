import unittest
import numpy as np
import theano

from autodiff.symbolic import Gradient

def f1(x):
    return x

def f2(x, y):
    return x * y

def f3(x, y=1):
    return x * y

def f4(x, y, *z):
    return x * y * sum(z)

def f5(x, y=2, *z):
    return x * y * sum(z)

def f6(x=1, y=2, *z):
    return x * y * sum(z)

def f7(x, y=2, **z):
    return x * y * z['f7']

def f7(x):
    def f7i(x):
        return x * x
    return f7i(x) + 1



class TestFunction(unittest.TestCase):
    pass


class TestGradient(unittest.TestCase):

    def test_scalar_fn(self):
        g = Gradient(lambda x : x ** 2)
        self.assertTrue(np.allclose(g(np.array(2.0)), 4))

    def test_int_fn(self):
        """ FAILS! """
        g = Gradient(lambda x : x ** 2)
        self.assertTrue(np.allclose(g(2), 4))
