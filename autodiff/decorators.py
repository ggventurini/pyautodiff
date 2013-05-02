import inspect

import autodiff.utils as utils
from autodiff.symbolic import Function, Gradient


def _function(**kwargs):
    def wrapper(fn):
        return Function(fn, **kwargs)
    return wrapper


def function(*fn, **kwargs):
    """
    Wraps a function with an AutoDiff Function instance, converting it to a
    symbolic representation.

    The function is compiled the first time it is called.
    """
    if len(fn) == 1:
        return Function(fn[0], **kwargs)
    elif len(fn) > 1:
        raise ValueError('function called with unsupported arguments.')
    else:
        return _function(**kwargs)


def _gradient(wrt, **kwargs):
    def wrapper(fn):
        return Gradient(fn, wrt=wrt, **kwargs)
    return wrapper


def gradient(*wrt, **kwargs):
    """
    Wraps a function with an AutoDiff Gradient instance, converting it to a
    symbolic representation that returns the derivative with respect to either
    all inputs or a subset (if specified).

    The function is compiled the first time it is called.
    """
    if wrt:
        # decorator called with no args
        if inspect.isfunction(wrt[0]):
            return Gradient(wrt[0], **kwargs)

        # decorator called with positional wrt args
        else:
            # check for keyword wrt args
            kw_wrt = utils.as_seq(kwargs.pop('wrt', None), tuple)
            return _gradient(wrt + kw_wrt, **kwargs)
    else:
        # decorator called with keyword wrt args
        kw_wrt = kwargs.pop('wrt', None)
        if kwargs:
            raise ValueError(
                'gradient called with unsupported '
                'arguments: {0}'.format(' '.join(kwargs.keys())))
        return _gradient(kw_wrt, **kwargs)
