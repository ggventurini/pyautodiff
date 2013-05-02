import inspect

import autodiff.utils as utils
from autodiff.symbolic import Function, Gradient


def function(fn):
    """
    Wraps a function with an AutoDiff Function instance, converting it to a
    symbolic representation.

    The function is compiled the first time it is called.
    """
    return Function(fn)


def _gradient(wrt):
    def wrapper(fn):
        return Gradient(fn, wrt=wrt)
    return wrapper


def gradient(*wrt, **kwargs):
    if wrt:
        # decorator called with no args
        if inspect.isfunction(wrt[0]):
            return Gradient(wrt[0])

        # decorator called with positional wrt args
        else:
            # check for keyword wrt args
            kw_wrt = utils.as_seq(kwargs.get('wrt'), tuple)
            return _gradient(wrt + kw_wrt)
    else:
        # decorator called with keyword wrt args
        kw_wrt = kwargs.pop('wrt', None)
        if kwargs:
            raise ValueError(
                'gradient called with unsupported '
                'arguments: {0}'.format(' '.join(kwargs.keys())))
        return _gradient(kw_wrt)
