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
    flatargs = utils.flatten(args)
    for i, a in enumerate(flatargs):
        if type(a) is int and -5 <= a <= 256:
            flatargs[i] = np.int16(a)
    clean_args = utils.unflatten(args, flatargs)

    flatkwargs = utils.flatten(kwargs)
    for i, a in enumerate(flatkwargs):
        if type(a) is int and -5 <= a <= 256:
            flatkwargs[i] = np.int16(a)
    clean_kwargs = utils.unflatten(kwargs, flatkwargs)
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

    def get_symbolic(self, x):
        """
        Attempts to retrieve the symbolic version of x.

        if x is an numeric object (int, float, numpy array), it must have been
        traced by the Symbolic class.

        if x is a string, it must have been tagged with
        autodiff.functions.tag().
        """
        if isinstance(x, basestring):
            if x in self.s_vars:
                return self.s_vars[x]
            else:
                raise ValueError(
                    'Requested the symbolic variable of tag `{0}`'
                    ', but `{0}` was not traced.'.format(x))
        elif utils.isvar(x):
            return x
        elif id(x) in self.s_vars:
            return self.s_vars[id(x)]
        else:
            raise ValueError(
                'Requested the symbolic variable shadowing object {0}'
                ', but it was not traced.'.format(repr(x)))

    def trace(self, fn, *args, **kwargs):
        """
        Given a Python function and arguments, recompiles a symbolic function
        and calls it on the [symbolic version of the] arguments.

        Returns the symbolic function result, and also stores any traced
        objects in self.s_vars.
        """
        clean_fn = clean_function_defaults(fn)
        recompiled_fn = self.context.recompile(clean_fn)
        clean_args, clean_kwargs = clean_int_args(*args, **kwargs)
        return recompiled_fn(*clean_args, **clean_kwargs)

    def get_theano_vars(self, inputs=None, outputs=None):
        """
        Returns a dict containing inputs, outputs and graph corresponding to
        the Theano version of the pyfn.
        """
        inputs = utils.as_seq(inputs, tuple)
        sym_inputs = [self.get_symbolic(x) for x in inputs]

        outputs = utils.as_seq(outputs, tuple)
        sym_outputs = [self.get_symbolic(x) for x in outputs]

        # get symbolic inputs corresponding to shared inputs in s_inputs
        # this dict maps each shared variable to its (non-shared) type.
        s_memo = OrderedDict((var, var.type())
                             for var in utils.flatten(sym_inputs))
        theano_inputs = tuple(s_memo.values())

        # get new graph, replacing shared inputs with symbolic ones
        # graph is a dict mapping "old" variables to "new" ones, where "old"
        # is the chain including shared variables, and "new" is the chain
        # with the non-shared replacements.
        graph = theano.gof.graph.clone_get_equiv(
            inputs=theano.gof.graph.inputs(sym_outputs),
            outputs=sym_outputs,
            memo=s_memo.copy())

        # get symbolic outputs
        theano_outputs = tuple([graph[o] for o in sym_outputs])

        return theano_inputs, theano_outputs, graph

    def get_function_compile_args(self, fn_inputs, fn_outputs):
        """
        Helper function: given the symbolic fn_inputs and fn_outputs,
        return the appropriate arguments for theano.function to compile
        a function.
        """
        if len(fn_outputs) == 1:
            fn_outputs = fn_outputs[0]

        return dict(inputs=fn_inputs, outputs=fn_outputs)

    def get_gradient_compile_args(self,
                                  fn_inputs,
                                  fn_outputs,
                                  graph,
                                  wrt=None,
                                  reduction=None):
        """
        Helper function: given the symbolic fn_inputs and fn_outputs, as well as
        a theano graph and wrt/reduction info, return the appropriate arguments
        for theano.function to compile a gradient.
        """
        wrt = utils.as_seq(wrt)

        if reduction is None:
            reduction = T.sum

        if reduction in ['sum', 'max', 'mean', 'min', 'prod', 'std', 'var']:
            reduction = getattr(theano.tensor, reduction)

        if callable(reduction):
            if 'numpy' in reduction.__module__:
                reduction = getattr(theano.tensor, reduction.__name__)
            fn_outputs = [reduction(o) for o in fn_outputs]

        if np.any([o.ndim != 0 for o in fn_outputs]):
            raise TypeError('Gradient requires either scalar outputs or a '
                            'reduction that returns a scalar.')

        # get wrt variables. If none were specified, use inputs.
        if len(wrt) == 0:
            wrt = [i for i in fn_inputs]
        else:
            wrt = [graph[self.get_symbolic(w)] for w in wrt]

        grads = utils.flatten([T.grad(o, wrt=wrt) for o in fn_outputs])

        if len(grads) == 1:
            grads = grads[0]

        return dict(inputs=fn_inputs, outputs=grads)

    def compile_function(self, inputs=None, outputs=None):
        """
        Based on traced variables, compile a Theano function of the inputs that
        returns the outputs.
        """
        fn_inputs, fn_outputs, _ = self.get_theano_vars(inputs, outputs)
        args = self.get_function_compile_args(fn_inputs, fn_outputs)
        return theano.function(on_unused_input='ignore', **args)

    def compile_gradient(self,
                         inputs=None,
                         outputs=None,
                         wrt=None,
                         reduction=None):
        """
        Based on traced variables, compile a Theano function of the
        inputs that returns the gradient of the outputs with respect to wrt.
        If wrt is None, it is assumed to be all of the inputs. A reduction may
        be specified (since gradients are defined with respect to scalars); if
        None is supplied, it is assumed to be 'sum'.
        """
        fn_inputs, fn_outputs, graph = self.get_theano_vars(inputs, outputs)
        args = self.get_gradient_compile_args(fn_inputs=fn_inputs,
                                              fn_outputs=fn_outputs,
                                              graph=graph,
                                              wrt=wrt,
                                              reduction=reduction)
        return theano.function(on_unused_input='ignore', **args)

    def compile_function_gradient(self,
                                  inputs=None,
                                  outputs=None,
                                  wrt=None,
                                  reduction=None):
        """
        Based on traced variables, compile a Theano function of the
        inputs that returns both the outputs and the gradient of the outputs
        with respect to wrt. If wrt is None, it is assumed to be all of the
        inputs. A reduction may be specified (since gradients are defined with
        respect to scalars); if None is supplied, it is assumed to be 'sum'.
        """
        fn_inputs, fn_outputs, graph = self.get_theano_vars(inputs, outputs)
        f_args = self.get_function_compile_args(fn_inputs, fn_outputs)
        g_args = self.get_gradient_compile_args(fn_inputs=fn_inputs,
                                                fn_outputs=fn_outputs,
                                                graph=graph,
                                                wrt=wrt,
                                                reduction=reduction)
        assert f_args['inputs'] == g_args['inputs']

        outputs = utils.as_seq(f_args['outputs'])
        outputs += utils.as_seq(g_args['outputs'])

        return theano.function(inputs=f_args['inputs'],
                               outputs=outputs,
                               on_unused_input='ignore')

class Function(Symbolic):
    pass


class Gradient(Function):
    pass


class HessianVector(Gradient):
    pass


class VectorArg(Function):
    pass