import unittest
import numpy as np
import theano
import theano.tensor as T

from autodiff.context import Context
from autodiff.AST import TheanoTransformer

class Tests(unittest.TestCase):
    def setUp(self):
        self.tt = TheanoTransformer()

    def test_shadow(self):
        def fn(x):
            return x

        f2 = self.tt.test_run(fn)
        fn(5)
        f2(5)
