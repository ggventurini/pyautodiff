import numpy as np
import theano
import theano.tensor as T
import types
import inspect

from autodiff.context import Context
from autodiff.compat import OrderedDict
import autodiff.utils as utils


def clean_int_args(*args, **kwargs):
    """
    Given args and kwargs, replaces small integers with numpy int16 objects, to
    allow tracing.
    """
    flatargs = utils.flat_from_doc(args)
    for i, a in enumerate(flatargs):
        if type(a) is int and -5 <= a <= 256:
            flatargs[i] = np.int16(a)
    clean_args = utils.doc_from_flat(args, flatargs)

    flatkwargs = utils.flat_from_doc(kwargs)
    for i, a in enumerate(flatkwargs):
        if type(a) is int and -5 <= a <= 256:
            flatkwargs[i] = np.int16(a)
    clean_kwargs = utils.doc_from_flat(kwargs, flatkwargs)
    return clean_args, clean_kwargs


def clean_function_defaults(fn):
    """
    Copy a function (or method) and replace int defaults with traceable int16
    objects.

    """
    # make deepcopy of fn because we might change its defaults
    # -- note that copy.deepcopy doesn't work on functions
    fn_copy = types.FunctionType(fn.func_code,
                                 fn.func_globals,
                                 fn.func_name,
                                 fn.func_defaults,
                                 fn.func_closure)

    # if pyfn is a method, make sure to make the copy a method as well
    if isinstance(fn, types.MethodType):
        fn_copy = types.MethodType(fn_copy,
                                   fn.im_self,
                                   fn.im_class)

    # replace integer defaults in fn to avoid tracing problems
    if fn_copy.func_defaults:
        a = inspect.getargspec(fn_copy)
        clean_defaults = tuple(clean_int_args(*a.defaults)[0])
        fn_copy.func_defaults = clean_defaults

    return fn_copy


class Symbolic(object):
    def __init__(self, context=None, borrowable=None):
        """
        Arguments
        ---------

        borrow : tuple of objects
            If an object in this tuple is encountered while tracing the
            function, then its symbolic representation will alias that object's
            memory location. This means that *inplace* operations on the Python
            (likely NumPy) object will affect the symbolic function.

        """
        if context is None:
            self.context = Context(borrowable=utils.as_seq(borrowable, tuple))
        elif isinstance(context, Context):
            self.context = context
        else:
            raise TypeError(
                'Received unrecognized Context: {0}'.format(context))

    @property
    def s_vars(self):
        return self.context.s_vars

    def trace(self, fn, *args, **kwargs):
        clean_fn = clean_function_defaults(fn)
        recompiled_fn = self.context.recompile(clean_fn)
        clean_args, clean_kwargs = clean_int_args(*args, **kwargs)
        return recompiled_fn(*clean_args, **clean_kwargs)
