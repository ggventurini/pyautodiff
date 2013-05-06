import numpy as np
# from autodiff import fmin_l_bfgs_b as fmin

from autodiff.symbolic import *
from autodiff.decorators import *

def test_subtensor_increment():
    def loss(x):
        if np.any(x < -100):
            return float('inf')
        x2 = x.copy()
        x2[0] += 3.0
        x2[1] -= 4.0
        rval = (x2 ** 2).sum()
        rval += 1.3
        rval *= 1.0
        return rval

    opt = fmin(loss, (np.zeros(2),))
    print opt
    assert np.allclose(opt, [-3, 4])



def loss(x):
    if np.any(x < -100):
        return float('inf')
    x2 = x.copy()
    x2[0] += 3.0
    x2[1] -= 4.0
    rval = (x2 ** 2).sum()
    rval += 1.3
    rval *= 1.0
    return rval


loss2 = VectorArgs(loss, (np.zeros(2),), compile_fn=True, compile_grad=True, compile_hv=True)
loss2(np.array([-3., 4]))


