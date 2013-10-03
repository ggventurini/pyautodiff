import unittest
import numpy as np
import theano
import theano.tensor as T

from autodiff.context import Context
from autodiff.ast_context import TheanoTransformer


TT = TheanoTransformer()


def checkfn(f, var_ndim, *args):
    dim = [[4] * nd for nd in var_ndim]
    values = tuple([np.random.random(d) for d in dim])
    F = TT.transform(f)
    py_result = f(*(values + args))
    sym_result = F(*(values + args)).eval()
    return np.allclose(py_result, sym_result)



class Python(unittest.TestCase):
    def test_range(self):
        def f(x):
            for i in range(3):
                x += 5
            return x
        self.assertTrue(checkfn(f, [1]))

        def f(x):
            a = 3
            for i in range(a):
                x += 5
            return x
        self.assertTrue(checkfn(f, [1]))

        def f(x):
            a = x[0] + 10
            for i in range(int(a)):
                x += 5
            return x
        self.assertTrue(checkfn(f, [1]))

        def f(x, a):
            for i in range(a):
                x += 5
            return x
        self.assertTrue(checkfn(f, [1], 3))


class BasicMath(unittest.TestCase):
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

    def test_comparisons(self):
        for d in range(3):
            self.assertTrue(checkfn(lambda x, y: x > y, [d, d]))
            self.assertTrue(checkfn(lambda x, y: x < y, [d, d]))
            self.assertTrue(checkfn(lambda x, y: x >= y, [d, d]))
            self.assertTrue(checkfn(lambda x, y: x <= y, [d, d]))
            self.assertTrue(checkfn(lambda x, y: x == y, [d, d]))
            self.assertTrue(checkfn(lambda x, y: x != y, [d, d]))

    def test_inplace(self):

        def iadd(x):
            x += 10
            return x

        def isub(x):
            x -= 10
            return x

        def imul(x):
            x *= 10
            return x

        def idiv(x):
            x /= 10.0
            return x

        for d in range(3):
            for f in [iadd, isub, imul, idiv]:
                self.assertTrue(checkfn(f, [d]))


