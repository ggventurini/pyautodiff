import unittest
import numpy as np

from autodiff.functions import escape, tag
from autodiff.symbolic import Function
from autodiff.decorators import function


class TestTag(unittest.TestCase):
    def test_tag(self):
        def f(x):
            y = tag(x + 2, 'y')
            z = y * 3
            return z

        F = Function(f)
        self.assertFalse('y' in F.tags)
        F(10)
        self.assertTrue('y' in F.tags)

    def test_tag_arg(self):
        def f(x):
            y = tag(x + 2, 'x')
            z = y * 3
            return z

        F = Function(f)
        self.assertFalse('x' in F.sym_vars)
        self.assertFalse('x' in F.tags)
        F(10)
        self.assertTrue('x' in F.sym_vars)
        self.assertTrue('x' in F.tags)
        self.assertTrue(F.sym_vars['x'] is not F.tags['x'])


    def test_tag_decorator(self):
        @function
        def F(x):
            y = tag(x + 2, 'y')
            z = y * 3
            return z

        self.assertFalse('y' in F.tags)
        F(10)
        self.assertTrue('y' in F.tags)
