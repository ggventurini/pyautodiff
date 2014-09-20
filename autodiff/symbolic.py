import numpy as np
import theano
import theano.tensor as T
import inspect

from autodiff.context import Context
from autodiff.compat import OrderedDict
import autodiff.utils as utils
from autodiff.functions import escape, escaped_call


class Symbolic(object):
    """
    A class that converts a Python function into a symbolic function of Theano
    objects. The symbolic function can be used to compile Theano functions of
    the original (Python) function, its gradient, or a Hessian-vector product.
    """

    def __init__(self,
                 pyfn,
                 context=None,
                 borrowable=None,
                 ignore=None,
                 escape_on_error=False):
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
            context = Context(borrowable=utils.as_seq(borrowable, tuple),
                              ignore=utils.as_seq(ignore, tuple),
                              escape_on_error=escape_on_error)
        assert isinstance(context, Context)
        self.context = context

        if isinstance(pyfn, Symbolic):
            pyfn = pyfn.pyfn
        self._pyfn = pyfn

        self._symfn = self.context.recompile(self.pyfn)

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
            self._symfn = self.context.recompile(self.pyfn)
        return self

    def __call__(self, *args, **kwargs):
        return self.trace(*args, **kwargs)[1]

    @property
    def pyfn(self):
        return self._pyfn

    @property
    def symfn(self):
        return self._symfn

    @property
    def cache(self):
        return self._cache

    @property
    def sym_vars(self):
        return self.context.sym_vars

    @property
    def tags(self):
        return self.context.tags

    def get_symbolic(self, x):
        return self.context.get_symbolic(x)

    def trace(self, *args, **kwargs):
        """
        Call the symbolic function on args and kwargs, returning the symbolic
        result and storing all shadowed variables.
        """
        # clean args and kwargs
        c_args, c_kwargs = utils.clean_int_args(*args, **kwargs)

        # call the symfn
        results = self.symfn(*c_args, **c_kwargs)

        # get a tuple of the symbolic inputs
        # but avoid 'self' and 'cls' bound arguments
        all_args = utils.expandedcallargs(self.symfn, *c_args, **c_kwargs)
        if (inspect.ismethod(self.pyfn) or
           (len(all_args) > 0 and type(all_args[0]) is type)):
            all_args = all_args[1:]

        # store the inputs and outputs so they can be accessed later
        # self.s_inputs = tuple(self.get_symbolic(a) for a in all_args)
        # self.s_outputs = utils.as_seq(results, tuple)
        inputs = tuple(self.get_symbolic(a) for a in all_args)
        # outputs = utils.as_seq(results, tuple)

        return inputs, results

    def get_theano_vars(self, inputs=None, outputs=None):
        """
        Returns a dict containing inputs, outputs and graph corresponding to
        the Theano version of the pyfn.
        """
        sym_inputs = tuple(self.get_symbolic(i)
                           for i in utils.as_seq(inputs))

        sym_outputs = tuple(self.get_symbolic(o)
                            for o in utils.as_seq(outputs))

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

    def get_function_compile_args(self, inputs, outputs):
        """
        Helper function: given the symbolic inputs and outputs,
        return the appropriate arguments for theano.function to compile
        a function.
        """
        return dict(inputs=inputs, outputs=outputs)

    def get_gradient_compile_args(self,
                                  inputs,
                                  outputs,
                                  graph,
                                  wrt=None,
                                  reduction=None):
        """
        Helper function: given the symbolic inputs and outputs, as well as
        a theano graph and wrt/reduction info, return the appropriate arguments
        for theano.function to compile a gradient.
        """
        wrt = utils.as_seq(wrt)

        if reduction in ['sum', 'max', 'mean', 'min', 'prod', 'std', 'var']:
            reduction = getattr(theano.tensor, reduction)

        if callable(reduction):
            if 'numpy' in reduction.__module__:
                reduction = getattr(theano.tensor, reduction.__name__)
            outputs = [reduction(o) if o.ndim > 0 else o for o in outputs]

        if np.any([o.ndim != 0 for o in outputs]):
            raise TypeError('Gradient requires either scalar outputs or a '
                            'reduction that returns a scalar.')

        # get wrt variables. If none were specified, use inputs.
        if len(wrt) == 0:
            wrt = [i for i in inputs]
        else:
            wrt = [graph[self.get_symbolic(w)] for w in wrt]

        grads = utils.flatten([T.grad(o, wrt=wrt) for o in outputs])

        return dict(inputs=inputs, outputs=utils.as_seq(grads, tuple))

    def get_hessian_vector_compile_args(self,
                                        inputs,
                                        outputs,
                                        graph,
                                        wrt=None,
                                        reduction=None):
        """
        Helper function: given the symbolic inputs and outputs, as well as
        a theano graph and wrt/reduction/vectors info, return the appropriate
        argumentsfor theano.function to compile a Hessian-vector product.
        """
        wrt = utils.as_seq(wrt)

        if reduction in ['sum', 'max', 'mean', 'min', 'prod', 'std', 'var']:
            reduction = getattr(theano.tensor, reduction)

        if callable(reduction):
            if 'numpy' in reduction.__module__:
                reduction = getattr(theano.tensor, reduction.__name__)
            outputs = [reduction(o) if o.ndim > 0 else o for o in outputs]

        if np.any([o.ndim != 0 for o in outputs]):
            raise TypeError('Gradient requires either scalar outputs or a '
                            'reduction that returns a scalar.')

        # get wrt variables. If none were specified, use inputs.
        if len(wrt) == 0:
            wrt = [i for i in inputs]
        else:
            wrt = [graph[self.get_symbolic(w)] for w in wrt]

        grads = utils.flatten([T.grad(o, wrt=wrt) for o in outputs])

        sym_vectors = tuple(T.TensorType(
            dtype=w.dtype, broadcastable=[False] * w.ndim)()
            for w in wrt)
        hessian_vectors = utils.as_seq(T.Rop(grads, wrt, sym_vectors), tuple)

        return dict(inputs=inputs + sym_vectors, outputs=hessian_vectors)

    def compile(self,
                function=False,
                gradient=False,
                hessian_vector=False,
                inputs=None,
                outputs=None,
                wrt=None,
                reduction=None):

        assert isinstance(function, bool)
        assert isinstance(gradient, bool)
        assert isinstance(hessian_vector, bool)

        if not (function or gradient or hessian_vector):
            raise ValueError(
                'At least one of `function`, `gradient`, or `hessian_vector` '
                'must be True when calling `compile()`.')

        fn_inputs, fn_outputs, fn_graph = self.get_theano_vars(inputs, outputs)

        inputs = fn_inputs
        outputs = ()

        if function:
            fn_args = self.get_function_compile_args(inputs=fn_inputs,
                                                     outputs=fn_outputs)
            outputs += fn_args['outputs']

        if gradient:
            g_args = self.get_gradient_compile_args(inputs=fn_inputs,
                                                    outputs=fn_outputs,
                                                    graph=fn_graph,
                                                    wrt=wrt,
                                                    reduction=reduction)
            outputs += g_args['outputs']

        if hessian_vector:
            hv_args = self.get_hessian_vector_compile_args(inputs=fn_inputs,
                                                           outputs=fn_outputs,
                                                           graph=fn_graph,
                                                           wrt=wrt,
                                                           reduction=reduction)
            inputs = hv_args['inputs']
            outputs += hv_args['outputs']

        if len(outputs) == 1:
            outputs = outputs[0]

        fn = theano.function(inputs=inputs,
                             outputs=outputs,
                             on_unused_input='ignore')

        return fn

    def compile_function(self, inputs=None, outputs=None):
        """
        Based on traced variables, compile a Theano function of the inputs that
        returns the outputs.
        """
        fn = self.compile(function=True, inputs=inputs, outputs=outputs)
        return fn

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
        fn = self.compile(gradient=True,
                          inputs=inputs,
                          outputs=outputs,
                          wrt=wrt,
                          reduction=reduction)
        return fn

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

        fn = self.compile(function=True,
                          gradient=True,
                          inputs=inputs,
                          outputs=outputs,
                          wrt=wrt,
                          reduction=reduction)
        return fn


class Tracer(Symbolic):
    """
    A Symbolic class for tracing variables through multiple functions.
    """

    def __init__(self,
                 context=None,
                 borrowable=None,
                 ignore=None,
                 escape_on_error=False):
        super(Tracer, self).__init__(pyfn=lambda: None,
                                     context=context,
                                     borrowable=borrowable,
                                     ignore=ignore,
                                     escape_on_error=escape_on_error)

    def trace(self, pyfn, *args, **kwargs):
        symbolic = Symbolic(pyfn=pyfn, context=self.context)
        return symbolic.trace(*args, **kwargs)[1]


class Function(Symbolic):
    """
    A Symbolic tracer which is specialized for a specific function, passed at
    initialization.
    """

    def __init__(self,
                 pyfn,
                 context=None,
                 borrowable=None,
                 ignore=None,
                 escape_on_error=False,
                 use_cache=True):
        super(Function, self).__init__(pyfn=pyfn,
                                       context=context,
                                       borrowable=borrowable,
                                       ignore=ignore,
                                       escape_on_error=escape_on_error)

        self._cache = dict()
        self.use_cache = use_cache

    def __call__(self, *args, **kwargs):
        all_args = utils.expandedcallargs(self.symfn, *args, **kwargs)
        if (inspect.ismethod(self.pyfn) or
           (len(all_args) > 0 and type(all_args[0]) is type)):
            all_args = all_args[1:]

        key = tuple(
            (np.asarray(a).ndim, np.asarray(a).dtype) for a in all_args)
        if key not in self.cache or not self.use_cache:
            self.context.reset()
            inputs, outputs = self.trace(*args, **kwargs)
            self.cache[key] = self.get_theano_function(inputs, outputs)
        fn = self.cache[key]
        return fn(*all_args)

    def get_theano_function(self, inputs, outputs):
        fn = self.compile_function(inputs=inputs, outputs=outputs)
        return fn


class Gradient(Function):
    def __init__(self,
                 pyfn,
                 wrt=None,
                 reduction=None,
                 borrowable=None,
                 ignore=None,
                 escape_on_error=False,
                 context=None,
                 use_cache=True):
        super(Gradient, self).__init__(pyfn=pyfn,
                                       borrowable=borrowable,
                                       ignore=ignore,
                                       context=context,
                                       escape_on_error=escape_on_error,
                                       use_cache=use_cache)
        self.wrt = utils.as_seq(wrt, tuple)
        self.reduction = reduction

    def get_theano_function(self, inputs, outputs):
        fn = self.compile_gradient(inputs=inputs,
                                   outputs=outputs,
                                   wrt=self.wrt,
                                   reduction=self.reduction)
        return fn


class HessianVector(Gradient):

    def __call__(self, *args, **kwargs):
        if 'vectors' in kwargs:
            vectors = kwargs.pop('vectors')
        else:
            raise ValueError(
                'HessianVector must be called with the keyword \'vectors\'.')
        vectors = utils.as_seq(vectors, tuple)

        all_args = utils.expandedcallargs(self.symfn, *args, **kwargs)

        key = tuple(np.asarray(a).ndim for a in all_args)
        if key not in self.cache or not self.use_cache:
            self.context.reset()
            inputs, outputs = self.trace(*args, **kwargs)
            self.cache[key] = self.get_theano_function(inputs, outputs)
        fn = self.cache[key]

        if len(self.wrt) > 0 and len(vectors) != len(self.wrt):
            raise ValueError('Expected {0} items in `vectors`; received '
                             '{1}.'.format(len(self.wrt), len(vectors)))
        elif len(self.wrt) == 0 and len(vectors) != len(inputs):
            raise ValueError('Expected {0} items in `vectors`; received '
                             '{1}.'.format(len(inputs), len(vectors)))

        return fn(*(all_args + vectors))

    def get_theano_function(self, inputs, outputs):
        fn = self.compile(hessian_vector=True,
                          inputs=inputs,
                          outputs=outputs,
                          wrt=self.wrt,
                          reduction=self.reduction)
        return fn


class VectorArg(object):

    def __init__(self,
                 pyfn,
                 init_args=None,
                 init_kwargs=None,
                 context=None,
                 borrowable=None,
                 ignore=None,
                 escape_on_error=False,
                 function=False,
                 gradient=False,
                 hessian_vector=False):

        if isinstance(pyfn, Symbolic):
            pyfn = pyfn.pyfn
        self.pyfn = pyfn

        init_args = utils.as_seq(init_args, tuple)
        init_kwargs = utils.as_seq(init_kwargs, dict)

        self.init_args = utils.expandedcallargs(
            pyfn, *init_args, **init_kwargs)

        def wrapped_function(vector):
            return pyfn(*escaped_call(self.args_from_vector, vector))

        def wrapper(*args, **kwargs):
            vector = self.vector_from_args(args, kwargs)
            v_args = self.args_from_vector(vector)
            return vector, pyfn(*v_args)

        symbolic = Symbolic(pyfn=wrapper,
                            context=context,
                            borrowable=borrowable,
                            ignore=ignore,
                            escape_on_error=escape_on_error)

        _, (sym_vector, result) = symbolic.trace(*init_args, **init_kwargs)

        fn = symbolic.compile(function=function,
                              gradient=gradient,
                              hessian_vector=hessian_vector,
                              inputs=sym_vector,
                              outputs=result)
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def vector_from_args(self, args, kwargs):
        if len(args) + len(kwargs) > 1:
            all_args = utils.expandedcallargs(self.pyfn, *args, **kwargs)
            return np.concatenate([np.asarray(a).flatten() for a in all_args])
        elif len(args) > 0:
            return np.asarray(args[0]).flatten()
        elif len(kwargs) > 0:
            return np.asarray(kwargs.values()[0]).flatten()
        else:
            return None

    def args_from_vector(self, vector):
        new_args = []
        idx = 0
        for arg in escape(self.init_args):
            if arg.ndim:
                new_args.append(vector[idx: idx + arg.size].reshape(*arg.shape))
            else:
                new_args.append(vector[idx: idx + arg.size][0])
            idx += arg.size
        return new_args
