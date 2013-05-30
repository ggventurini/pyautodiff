import numpy as np
import theano
import theano.tensor as tt
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


def copy_function(fn):
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
    def __init__(self,
                 borrow=None,
                 force_floatX=False,
                 context=None):
        """
        Arguments
        ---------

        borrow : tuple of objects
            If an object in this tuple is encountered while tracing the
            function, then its symbolic representation will alias that object's
            memory location. This means that *inplace* operations on the Python
            (likely NumPy) object will affect the symbolic function.

        force_floatX : bool
            If True, floats and float NumPy ndarrays will be cast to the dtype
            specified at theano.config.floatX when forming symbolic shared
            variables, if they do not have it already. Objects in `borrowable`
            are never cast.

        """
        if context is None:
            self.context = Context(borrowable=utils.as_seq(borrow, tuple),
                                   force_floatX=force_floatX)
        elif isinstance(context, Context):
            self.context = context
        else:
            raise TypeError(
                'Received unrecognized Context: {0}'.format(context))

    @property
    def s_vars(self):
        return self.context.svars

    def trace(self, fn, *args, **kwargs):
        fn_copy = copy_function(fn)
        clean_args, clean_kwargs = clean_int_args(*args, **kwargs)
        return self.context.call(fn_copy, clean_args, clean_kwargs)

    def get_theano_variables(self, inputs=None, outputs=None):
        """
        Returns a dict containing inputs, outputs and graph corresponding to
        the Theano version of the pyfn.
        """
        inputs = utils.as_seq(inputs, tuple)
        sym_inputs = [self.get_symbolic(x) for x in inputs]

        outputs = utils.as_seq(outputs, tuple)
        sym_outputs = [self.get_symbolic(x) for x in outputs]

        # get symbolic inputs corresponding to shared inputs in s_inputs
        s_memo = OrderedDict((arg, arg.type(name=arg.name))
                             for arg in utils.flat_from_doc(sym_inputs))
        theano_inputs = tuple(s_memo.values())

        # get new graph, replacing shared inputs with symbolic ones
        graph = theano.gof.graph.clone_get_equiv(
            theano.gof.graph.inputs(sym_outputs),
            sym_outputs,
            memo=s_memo.copy())

        # get symbolic outputs
        theano_outputs = tuple([graph[o] for o in sym_outputs])

        return theano_inputs, theano_outputs, graph

    def compile_function(self,
                         inputs=None,
                         outputs=None):

        fn_inputs, fn_outputs, graph = self.get_theano_variables(inputs,
                                                                 outputs)

        if len(fn_outputs) == 1:
            fn_outputs = fn_outputs[0]

        # compile function
        fn = theano.function(inputs=fn_inputs,
                             outputs=fn_outputs,
                             on_unused_input='ignore')

        return fn

    def compile_gradient(self,
                         inputs=None,
                         outputs=None,
                         wrt=None,
                         reduction=None):

        fn_inputs, fn_outputs, graph = self.get_theano_variables(inputs,
                                                                 outputs)
        wrt = utils.as_seq(wrt)

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

        grads = utils.flat_from_doc([tt.grad(o, wrt=wrt) for o in fn_outputs])

        if len(grads) == 1:
            grads = grads[0]

        # compile function
        fn = theano.function(inputs=fn_inputs,
                             outputs=grads,
                             on_unused_input='ignore')

        return fn

    def get_symbolic(self, x):
        """
        Retrieve the symbolic version of x.

        x : python object or string or Theano variable

        If x is an object, it must have been traced by the Symbolic class.
        If x is a string, it must have been tagged with
            autodiff.functions.tag() or have been placed in s_vars by
            other means.
        If x is a Theano variable, it must be in the s_vars dict.
        """
        if isinstance(x, basestring):
            if x in self.s_vars:
                return self.s_vars[x]
            else:
                raise ValueError(
                    'Requested the symbolic variable of tag `{0}`'
                    ', but `{0}` was not traced.'.format(x))

        elif isinstance(x, (tt.TensorConstant,
                            tt.TensorVariable,
                            tt.sharedvar.SharedVariable)):
            if x in self.s_vars.values():
                return x
            else:
                raise ValueError(
                    'Requested the symbolic variable {0}'
                    ', but it was not traced.'.format(repr(x)))

        elif id(x) in self.s_vars:
            return self.s_vars[id(x)]
        else:
            raise ValueError(
                'Requested the symbolic variable shadowing object {0}'
                ', but it was not traced.'.format(repr(x)))


class Function(Symbolic):
    def __init__(self,
                 pyfn,
                 borrow=None,
                 force_floatX=False,
                 context=None,
                 use_cache=True):
        """
        Arguments
        ---------

        pyfn : Python function
            The function that will be traced. The Function object will attempt
            to build symbolic representations of any variables referenced in or
            created by this function.

        borrow : tuple of objects
            If an object in this tuple is encountered while tracing the
            function, then its symbolic representation will alias that object's
            memory location. This means that *inplace* operations on the Python
            (likely NumPy) object will affect the symbolic function.

        force_floatX : bool
            If True, floats and float NumPy ndarrays will be cast to the dtype
            specified at theano.config.floatX when forming symbolic shared
            variables, if they do not have it already. Objects in `borrowable`
            are never cast.

        """
        super(Function, self).__init__(borrow=borrow,
                                       force_floatX=force_floatX,
                                       context=context)

        # if the pyfn is an autodiff Function, get the pyfn
        if isinstance(pyfn, Function):
            pyfn = pyfn.pyfn

        self._pyfn = copy_function(pyfn)

        self._cache = dict()
        self.use_cache = use_cache
        self.argspec = inspect.getargspec(self._pyfn)

        # set the instance docstring to look like that of the function
        ds = 'AutoDiff class: {0}\n\nWrapped docstring:\n\n'.format(
            self.__class__.__name__)
        if self.pyfn.__doc__ is not None:
            fn_ds = self.pyfn.__doc__
        else:
            fn_ds = '[no docstring found]\n '
        self.__doc__ = ds + fn_ds

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def reset(self, reset_cache=True):
        # self.s_vars.clear()
        self.s_inputs.clear()
        self.s_outputs.clear()
        self.context.reset()

        if reset_cache:
            self.cache.clear()

    @property
    def pyfn(self):
        return self._pyfn

    @property
    def s_vars(self):
        return self.context.svars

    @property
    def cache(self):
        return self._cache

    def cache_id(self, args=None, kwargs=None, inputs=None, outputs=None):
        """
        Generates a unique id for caching a function
        """
        if self.use_cache:
            if args is None:
                args = ()
            if kwargs is None:
                kwargs = {}
            inputs = utils.as_seq(inputs)
            outputs = utils.as_seq(outputs)
            callargs = utils.orderedcallargs(self.pyfn, *args, **kwargs)
            varargs = len(callargs.get(self.argspec.varargs, ()))
            dim = tuple(np.asarray(a).ndim for a in callargs.values())
            inputs_id = tuple(id(i) for i in inputs)
            outputs_id = tuple(id(o) for o in outputs)
            return (varargs, dim, inputs_id, outputs_id)

    def trace(self, args=None, kwargs=None):
        """
        Given args and kwargs, call the Python function and get its
        symbolic representation.

        Three ordered dictionaries are maintained:
            self.s_vars   : {id(obj) : sym_var}
                            Contains all symbolic variables traced during
                            function execution, indexed by the id of the
                            corresponding Python object.

            self.s_inputs   : {arg name : sym_var}
                            Contains all symbolic inputs to the function,
                            indexed by the argument name.

            self.s_outputs : {id(obj) : sym_var}
                            Contains the symbolic function results, indexed by
                            the id of the corresponding Python object.

        """
        if args is None:
            args = ()
        elif not isinstance(args, tuple):
            raise TypeError('args must be a tuple')

        if kwargs is None:
            kwargs = dict()
        elif not isinstance(kwargs, dict):
            raise TypeError('kwargs must be a dict')

        def check(name, i):
            if type(i) is int and -5 <= i <= 256:
                i = np.int_(i)
            elif isinstance(i, (list, tuple, dict)):
                raise TypeError(
                    'Function arguments can not be containers (received '
                    '{0} for argument \'{1}\').'.format(i, name))
            return i

        args = list(args)

        for i, (n, a) in enumerate(zip(self.argspec.args, args)):
            args[i] = check(n, a)

        for i, a in enumerate(args[len(self.argspec.args):]):
            args[len(self.argspec.args) + i] = check(self.argspec.varargs, a)

        for k, v in kwargs.iteritems():
            kwargs[k] = check(k, v)

        args = tuple(args)

        self.reset(reset_cache=False)

        # call the Context
        results = self.context.call(self.pyfn, args, kwargs)

        # collect symbolic variables in s_vars
        self.s_vars.update(self.context.svars)

        # collect symbolic arguments in s_inputs
        callargs = utils.orderedcallargs(self.pyfn, *args, **kwargs)

        for name, arg in callargs.iteritems():

            # collect variable args
            if name == self.argspec.varargs:
                self.s_inputs[name] = ()
                for i, a in enumerate(arg):
                    try:
                        self.s_inputs[name] += (self.s_vars[id(a)],)
                        self.s_inputs[name][-1].name = '{0}_{1}'.format(
                            name, i)
                    except KeyError:
                        raise KeyError('Unable to trace item {0} of variable '
                                       'argument \'{1}\'.'.format(i, name))
                    except:
                        raise

            # collect variable kwargs
            elif name == self.argspec.keywords:
                for n, a in arg.iteritems():
                    try:
                        self.s_inputs[n] = self.s_vars[id(a)]
                        self.s_inputs[n].name = n
                    except KeyError:
                        raise KeyError('Unable to trace argument '
                                       '\'{0}\'.'.format(n))
                    except:
                        raise

            # collect positional args
            else:
                # avoid tracing 'self' and 'cls' args
                if arg is not getattr(self.pyfn, 'im_self', None) \
                   and type(arg) is not type:
                    try:
                        self.s_inputs[name] = self.s_vars[id(arg)]
                        self.s_inputs[name].name = name
                    except KeyError:
                        raise KeyError('Unable to trace argument '
                                       '\'{0}\'.'.format(name))
                    except:
                        raise

        # collect symbolic results in s_outputs
        results = utils.as_seq(results, tuple)
        for i, r in enumerate(results):
            try:
                self.s_outputs[id(r)] = self.s_vars[id(r)]
            except KeyError:
                raise KeyError('Unable to trace result #{0} '
                               '(indexed from 1).'.format(i + 1))
            except:
                raise

        if len(results) == 1:
            return results[0]
        else:
            return results

    def get_theano_variables(self, inputs=None, outputs=None):
        """
        Returns a dict containing inputs, outputs and graph corresponding to
        the Theano version of the pyfn.
        """
        inputs = utils.as_seq(inputs, tuple)
        outputs = utils.as_seq(outputs, tuple)

        if inputs:
            sym_inputs = [self.get_symbolic(x) for x in inputs]
        else:
            sym_inputs = self.s_inputs.values()

        if outputs:
            sym_outputs = [self.get_symbolic(x) for x in outputs]
        else:
            sym_outputs = self.s_outputs.values()

        # get symbolic inputs corresponding to shared inputs in s_inputs
        s_memo = OrderedDict((arg, arg.type(name=arg.name))
                             for arg in utils.flat_from_doc(sym_inputs))

        # get new graph, replacing shared inputs with symbolic ones
        graph = theano.gof.graph.clone_get_equiv(
            theano.gof.graph.inputs(sym_outputs),
            sym_outputs,
            memo=s_memo.copy())

        # get symbolic outputs
        theano_outputs = tuple([graph[o] for o in sym_outputs])

        defaults = dict()
        if self.argspec.defaults:
            defaults.update(zip(reversed(self.argspec.args),
                                reversed(self.argspec.defaults)))
        theano_inputs = tuple([theano.Param(variable=i,
                                            default=defaults.get(i.name, None),
                                            name=i.name)
                               for i in s_memo.values()])

        final_in, final_out = self.finalize_inputs_outputs(theano_inputs,
                                                           theano_outputs,
                                                           graph)

        return final_in, final_out

    def finalize_inputs_outputs(self, inputs, outputs, graph):
        """
        This function accepts a "default" set of inputs and outputs
        corresponding to the traced function (or specified) and returns the
        final inputs and outputs for compilation.

        For example, to create a "standard" function, the inputs and outputs
        can be passed through directly. However, to create a gradient, the
        tensor.grad op might be called on the outputs before returning them.
        """
        return inputs, outputs

    def compile_function(self,
                         trace_args=None,
                         trace_kwargs=None,
                         fn_inputs=None,
                         fn_outputs=None):

        trace_args = utils.as_seq(trace_args, tuple)
        trace_kwargs = utils.as_seq(trace_kwargs, dict)

        cache_id = self.cache_id(trace_args,
                                 trace_kwargs,
                                 fn_inputs,
                                 fn_outputs)

        if cache_id and cache_id in self.cache:
            return self.cache[cache_id]

        self.trace(trace_args, trace_kwargs)
        inputs, outputs = self.get_theano_variables(fn_inputs, fn_outputs)

        if len(outputs) == 1:
            outputs = outputs[0]

        # compile function
        fn = theano.function(inputs=inputs,
                             outputs=outputs,
                             on_unused_input='ignore')

        # store in cache if it has a valid id
        self.cache[cache_id] = fn

        return fn

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

    def call(self, *args, **kwargs):
        fn = self.compile_function(args, kwargs, None, None)
        pos_args = utils.flat_from_doc(args)
        pos_args = [a for a in args if not type(a) is type]
        return fn(*pos_args, **kwargs)

    def get_symbolic(self, x):
        """
        Retrieve the symbolic version of x.

        x : python object or string or int

        If x is a string, it is matched first to any tagged objects, then to
            the names of the function arguments.
        If x is an int, it is matched to the index of the function results
        If x is an object, it must have been traced by the Symbolic class.
        """
        if type(x) is int:
            return self.s_outputs.values()[x]
        elif isinstance(x, basestring):
            if x in self.s_vars:
                return self.s_vars[x]
            elif x in self.s_inputs:
                return self.s_inputs[x]
            else:
                raise ValueError(
                    'Requested the symbolic variable shadowing object {0}'
                    ', but it was not traced.'.format(repr(x)))
        else:
            if id(x) in self.s_vars:
                return self.s_vars[id(x)]
            else:
                raise ValueError(
                    'Requested the symbolic variable shadowing object {0}'
                    ', but it was not traced.'.format(repr(x)))


class Gradient(Function):
    """
    Build a symbolic Theano gradient from a scalar-valued NumPy function.

    The resulting function returns the gradient of the NumPy function with
    respect to (optionally specified) variables.
    """

    def __init__(self,
                 pyfn,
                 wrt=None,
                 borrow=None,
                 force_floatX=False,
                 context=None):
        super(Gradient, self).__init__(pyfn=pyfn,
                                       borrow=borrow,
                                       force_floatX=force_floatX,
                                       context=context)
        self.wrt = utils.as_seq(wrt, tuple)

    def finalize_inputs_outputs(self, inputs, outputs, graph):
        if np.any([o.ndim != 0 for o in outputs]):
            raise TypeError('Gradient requires scalar outputs.')

        # get wrt variables. If none were specified, use inputs.
        if len(self.wrt) == 0:
            wrt = [i.variable for i in inputs]
        else:
            wrt = [graph[self.get_symbolic(w)] for w in self.wrt]

        grads = utils.flat_from_doc([tt.grad(o, wrt=wrt) for o in outputs])

        return inputs, grads


class HessianVector(Gradient):
    """
    Build a symbolic Theano Hessian-vector product from a scalar-valued NumPy
    function.

    The resulting function returns the Hessian-vector product of the NumPy
    function with respect to (optionally specified) variables and a vector
    or tuple of vectors (for multiple wrt variables).

    The vectors must be passed to the resulting function with the keyword
    '_vectors'.
    """

    def finalize_inputs_outputs(self, inputs, outputs, graph):
        if np.any([o.ndim != 0 for o in outputs]):
            raise TypeError('HessianVector requires scalar outputs.')

        # get wrt variables. If none were specified, use inputs.
        if len(self.wrt) == 0:
            wrt = [i.variable for i in inputs]
        else:
            wrt = [graph[self.get_symbolic(w)] for w in self.wrt]

        grads = utils.flat_from_doc([tt.grad(o, wrt=wrt) for o in outputs])

        sym_vecs = tuple(tt.TensorType(dtype=w.dtype,
                                       broadcastable=[False]*w.ndim)()
                         for w in wrt)
        hess_vec = tt.Rop(grads, wrt, sym_vecs)

        return inputs + sym_vecs, hess_vec

    def call(self, *args, **kwargs):
        if '_vectors' in kwargs:
            vectors = kwargs.pop('_vectors')
        else:
            raise ValueError(
                'Vectors must be passed the keyword \'_vectors\'.')
        vectors = utils.as_seq(vectors, tuple)

        fn = self.compile_function(args, kwargs)

        if len(self.wrt) > 0 and len(vectors) != len(self.wrt):
            raise ValueError('Expected {0} items in _vectors; received '
                             '{1}.'.format(len(self.wrt), len(vectors)))
        elif len(self.wrt) == 0 and len(vectors) != len(self.s_inputs):
            raise ValueError('Expected {0} items in _vectors; received '
                             '{1}.'.format(len(self.s_inputs), len(vectors)))

        pos_args = tuple(utils.flat_from_doc(args)) + vectors
        return fn(*pos_args, **kwargs)


class VectorArg(Function):
    """
    Many function optimizers do not support multiple arguments; they pass a
    single vector containing all parameter values.

    This class builds symbolic function, gradient, and Hessian-vector product
    functions from arbitrary NumPy functions, all of which accept a single
    vector argument. To compile gradient and Hessian-vector products, the NumPy
    function must return a scalar. The Hessian-vector product functions will
    take an additional vector argument.

    Users can specify any combination of 'compile_fn', 'compile_grad', or
    'compile_hv' at instantiation, and the VectorArg will return the
    appropriate values, in that order. At least one 'compile' keyword must be
    True.

    Also, VectorArg classes can be provided 'init_args', an initial set of
    function arguments. The shape and dtype of these initial arguments is used
    to build the resulting function. 'init_kwargs' may also be passed, and are
    stored as positional arguments in the order specified by the function
    signature.

    Note that VectorArg functions are compiled only once, at instantiation.
    """
    def __init__(self,
                 pyfn,
                 init_args=None,
                 init_kwargs=None,
                 compile_fn=False,
                 compile_grad=False,
                 compile_hv=False,
                 borrow=None,
                 force_floatX=False,
                 context=None):
        if not (compile_fn or compile_grad or compile_hv):
            raise ValueError('At least one of \'compile_fn\', '
                             '\'compile_grad\', or \'compile_hv\' '
                             'must be True.')
        super(VectorArg, self).__init__(pyfn=pyfn,
                                        borrow=borrow,
                                        force_floatX=force_floatX,
                                        context=context)

        self.compile_fn = compile_fn
        self.compile_grad = compile_grad
        self.compile_hv = compile_hv
        if init_kwargs is None:
            init_kwargs = dict()
        self.init_args = init_args
        self.init_kwargs = init_kwargs
        self.all_init_args = utils.expandedcallargs(self.pyfn,
                                                    *init_args,
                                                    **init_kwargs)

        self.cache['fn'] = self.compile_function(init_args, init_kwargs)

    def finalize_inputs_outputs(self, inputs, outputs, graph):
        if self.compile_grad or self.compile_hv:
            if outputs.ndim != 0:
                raise TypeError('Gradient requires scalar outputs.')
            grad = tt.grad(outputs, wrt=inputs)

        if self.compile_hv:
            sym_vec = tt.vector('hv_vector', dtype=grad.dtype)
            hess_vec = tt.Rop(grad, inputs, sym_vec)

        vector_inputs = [inputs]
        vector_outputs = []

        if self.compile_fn:
            vector_outputs.append(outputs)
        if self.compile_grad:
            vector_outputs.append(grad)
        if self.compile_hv:
            vector_inputs.append(sym_vec)
            vector_outputs.append(hess_vec)

        return vector_inputs, vector_outputs

    def get_theano_variables(self, inputs=None, outputs=None):
        """
        Returns a dict containing inputs, outputs and graph corresponding to
        the Theano version of the pyfn.

        This version of the function returns a single vector input.
        """
        inputs = utils.as_seq(inputs, tuple)
        outputs = utils.as_seq(outputs, tuple)

        if inputs:
            sym_inputs = [self.get_symbolic(x) for x in inputs]
        else:
            sym_inputs = self.s_inputs.values()

        if outputs:
            sym_outputs = [self.get_symbolic(x) for x in outputs]
        else:
            sym_outputs = self.s_outputs.values()

        if len(sym_outputs) > 1:
            raise ValueError(
                'VectorArg functions should return a single output.')

        # get symbolic inputs corresponding to shared inputs in s_inputs
        s_memo = OrderedDict()
        sym_args = utils.flat_from_doc(sym_inputs)
        real_args = utils.flat_from_doc(self.all_init_args)

        # create a symbolic vector, then split it up into symbolic input
        # args
        inputs_dtype = self.vector_from_args(self.all_init_args).dtype
        theano_input = tt.vector(name='theta', dtype=inputs_dtype)
        i = 0
        for sa, ra in zip(sym_args, real_args):
            if sa.ndim > 0:
                vector_arg = theano_input[i: i + ra.size].reshape(ra.shape)
            else:
                vector_arg = theano_input[i]
            s_memo[sa] = tt.patternbroadcast(
                vector_arg.astype(str(sa.dtype)),
                broadcastable=sa.broadcastable)
            i += ra.size

        # get new graph, replacing shared inputs with symbolic ones
        graph = theano.gof.graph.clone_get_equiv(
            theano.gof.graph.inputs(sym_outputs),
            sym_outputs,
            memo=s_memo.copy())

        # get symbolic outputs
        theano_outputs = graph[sym_outputs[0]]

        final_in, final_out = self.finalize_inputs_outputs(theano_input,
                                                           theano_outputs,
                                                           graph)

        return final_in, final_out

    def vector_from_args(self, args=None, kwargs=None):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = dict()
        all_args = utils.expandedcallargs(self.pyfn, *args, **kwargs)
        return np.concatenate([np.asarray(a).flat for a in all_args])

    def args_from_vector(self, vector):
        args = []
        last_idx = 0
        for a in self.all_init_args:
            args.append(vector[last_idx:last_idx+a.size].reshape(a.shape))
            last_idx += a.size

        return args

    def call(self, *args, **kwargs):
        return self.cache['fn'](*args, **kwargs)
