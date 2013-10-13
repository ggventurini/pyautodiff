import logging

logger = logging.getLogger('autodiff')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

import autodiff.optimize

from autodiff.symbolic import Symbolic, Function, Gradient, HessianVector
from autodiff.decorators import function, gradient, hessian_vector
from autodiff.functions import escape, tag, escaped_call

