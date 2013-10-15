import numpy as np
import theano
import theano.tensor as T
import types
import inspect

from autodiff.context import Context
from autodiff.compat import OrderedDict
import autodiff.utils as utils


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
        return self.context.get_symbolic(x)

    def trace(self, fn, *args, **kwargs):
        """
        Given a Python function and arguments, recompiles a symbolic function
        and calls it on the [symbolic version of the] arguments.

        Returns the symbolic function result, and also stores any traced
        objects in self.s_vars.
        """
        recompiled_fn = self.context.recompile(fn)
        clean_args, clean_kwargs = utils.clean_int_args(*args, **kwargs)
        return recompiled_fn(*clean_args, **clean_kwargs)

    def get_theano_vars(self, inputs=None, outputs=None):
        """
        Returns a dict containing inputs, outputs and graph corresponding to
        the Theano version of the pyfn.
        """
        inputs = utils.as_seq(inputs, tuple)
        sym_inputs = [self.get_symbolic(x) for x in inputs]

        sym_outputs = utils.as_seq(outputs, tuple)

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
    """
    A Symbolic tracer which is specialized for a specific function, passed at
    initialization.
    """
    def __init__(self, pyfn, borrowable=None, context=None, use_cache=True):
        super(Function, self).__init__(borrowable=borrowable, context=context)

        # if the fn is an autodiff Function class, get its own fn
        if isinstance(pyfn, Function):
            pyfn = pyfn.pyfn
        self._pyfn = pyfn

        self._symfn = self.context.recompile(self.pyfn)

        self._cache = dict()
        self.use_cache = use_cache
        self.argspec = inspect.getargspec(self.pyfn)

        # set the instance docstring to look like that of the function
        ds = 'AutoDiff class: {0}\n\nWrapped docstring:\n\n'.format(
            self.__class__.__name__)
        if self.pyfn.__doc__ is not None:
            fn_ds = self.pyfn.__doc__
        else:
            fn_ds = '[no docstring found]\n '
        self.__doc__ = ds + fn_ds

    @property
    def pyfn(self):
        return self._pyfn

    @property
    def symfn(self):
        return self._symfn

    @property
    def cache(self):
        return self._cache

    def __get__(self, instance, owner=None):
        """
        Necessary descriptor for decorator compatibility.

        At decoration time, methods have not been bound. However, when bound
        methods are accessed, the __get__ method is called, so we can monitor
        that call and bind the method as necessary.
        """
        if instance is not None:
            method = self.pyfn.__get__(instance, owner)
            self._pyfn = method
        return self

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        all_args = utils.expandedcallargs(self.pyfn, *args, **kwargs)

        key = tuple(np.asarray(a).ndim for a in all_args)
        if key not in self.cache or not self.use_cache:
            self.trace(*args, **kwargs)
            self.cache[key] = self.compile_function(inputs=self.s_inputs,
                                                    outputs=self.s_outputs)
        fn = self.cache[key]
        return fn(*all_args)

    def trace(self, *args, **kwargs):
        """
        Given args and kwargs, call the Python function and get its
        symbolic representation.

        A dictionary of shadowed symbolic variables is maintained:
            self.s_vars   : {id(obj) : sym_var}
                            Contains all symbolic variables traced during
                            function execution, indexed by the id of the
                            corresponding Python object.

        Additionally, self.s_inputs and self.s_outputs are lists of symbolic
        arguments and results, respectively.
        """

        # clean args and kwargs
        c_args, c_kwargs = utils.clean_int_args(*args, **kwargs)

        # call the symfn
        results = self.symfn(*c_args, **c_kwargs)

        # get a tuple of the symbolic inputs
        # but avoid 'self' and 'cls' bound arguments
        all_args = utils.expandedcallargs(self.pyfn, *c_args, **c_kwargs)
        if (inspect.ismethod(self.pyfn) or
           (len(all_args) > 0 and type(all_args[0]) is type)):
            all_args = all_args[1:]
        self.s_inputs = tuple(self.get_symbolic(a) for a in all_args)

        # get a tuple of the symbolic outputs
        self.s_outputs = utils.as_seq(results, tuple)

        return results

class Gradient(Function):
    pass


class HessianVector(Gradient):
    pass


class VectorArg(Function):
    pass