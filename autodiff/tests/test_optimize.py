import unittest
import numpy as np
from autodiff.optimize import fmin_l_bfgs_b, fmin_cg, fmin_ncg


def L2(x, y):
    return ((x - y) ** 2).mean()


def l2_loss(p):
    l2_x = np.arange(20).reshape(4, 5)
    l2_b = np.arange(3) - 1.5
    loss = L2(np.dot(l2_x, p) - l2_b, np.arange(3))
    return loss


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
        ans = [-3, 4]

        opt = fmin_l_bfgs_b(subtensor_loss, x0)
        self.assertTrue(np.allclose(opt, ans))

        opt = fmin_cg(subtensor_loss, x0)
        self.assertTrue(np.allclose(opt, ans))

        opt = fmin_ncg(subtensor_loss, x0)
        self.assertTrue(np.allclose(opt, ans))

    def test_L2(self):
        x0 = np.zeros((5, 3))
        ans = np.array([[+3.0, -1.0, -5.0],
                        [+1.5, -0.5, -2.5],
                        [+0.0,  0.0,  0.0],
                        [-1.5,  0.5,  2.5],
                        [-3.0,  1.0,  5.0]]) / 10.0

        opt = fmin_l_bfgs_b(l2_loss, x0)
        self.assertTrue(np.allclose(opt, ans))

        opt = fmin_cg(l2_loss, x0)
        self.assertTrue(np.allclose(opt, ans))

        opt = fmin_ncg(l2_loss, x0)
        self.assertTrue(np.allclose(opt, ans))
