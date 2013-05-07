import unittest
import numpy as np
from autodiff.optimize import fmin_l_bfgs_b, fmin_cg


def subtensor_loss(x):
    if np.any(x < -100):
        return float('inf')
    x2 = x.copy()
    x2[0] += 3.0
    x2[1] -= 4.0
    rval = (x2 ** 2).sum()
    rval += 1.3
    rval *= 1.0
    return rval


class TestMinimizers(unittest.TestCase):
    def test_subtensor(self):
        x0 = np.zeros(2)
        opt = fmin_l_bfgs_b(subtensor_loss, x0)
        self.assertTrue(np.allclose(opt, [-3, 4]))
        opt = fmin_cg(subtensor_loss, x0)
        self.assertTrue(np.allclose(opt, [-3, 4]))
