import unittest
import numpy as np
import theano

from autodiff.symbolic import Gradient


class TestGradient(unittest.TestCase):

    def test_scalar_fn(self):
        g = Gradient(lambda x : x ** 2)
        self.assertTrue(np.allclose(g(np.array(2.0)), 4))

    def test_int_fn(self):
        """ FAILS! """
        g = Gradient(lambda x : x ** 2)
        self.assertTrue(np.allclose(g(2), 4))
