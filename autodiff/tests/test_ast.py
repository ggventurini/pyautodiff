import unittest
import numpy as np
import theano
import theano.tensor as T

from autodiff.context import Context
from autodiff.ast_context import TheanoTransformer


TT = TheanoTransformer()


def checkfn(fn, var_ndim, *args):
    dim = [[4] * nd for nd in var_ndim]
    values = tuple([np.random.random(d) for d in dim])
    F = TT.transform(fn)
    py_result = fn(*(values + args))
    sym_result = F(*(values + args)).eval()
    return np.allclose(py_result, sym_result)


class Tests(unittest.TestCase):

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

