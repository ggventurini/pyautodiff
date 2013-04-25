import inspect
import numpy as np
import theano
import theano.tensor as tt
from collections import OrderedDict
from autodiff.context import Context


class Symbolic(object):
    def __init__(self, pyfn):
        self._pyfn = pyfn
        self.s_vars = OrderedDict()
        self.s_args = OrderedDict()
        self.s_results = OrderedDict()

    @property
    def pyfn(self):
        return self._pyfn

    def _small_int_check(self, arg_dict, var_args):
        """
        Replace any small integer arguments NumPy ints.

        CPython caches and reuses small integers (-5 <= i <= 256), meaning
        there is no way to differentiate them while tracing a function.
        """

        new_arg_dict = OrderedDict()
        for k, v in arg_dict.iteritems():
            if type(v) is int and -5 <= v <= 256:
                new_arg_dict[k] = np.int_(v)
            else:
                new_arg_dict[k] = v

        new_var_args = []
        for a in var_args:
            if type(a) is int and -5 <= a <= 256:
                new_var_args.append(np.int_(a))
            else:
                new_var_args.append(a)

        return new_arg_dict, new_var_args

    def trace(self, *args, **kwargs):
        """
        Given args and kwargs, call the Python function and get its
        symbolic representation.

        Three ordered dictionaries are maintained:
            self.s_vars   : {id(obj) : sym_var}
                            Contains all symbolic variables traced during
                            function execution, indexed by the id of the
                            corresponding Python object.

            self.s_args   : {arg name : sym_var}
                            Contains all symbolic inputs to the function,
                            indexed by the argument name.

            self.s_results : {id(obj) : sym_var}
                            Contains the symbolic function results, indexed by
                            the id of the corresponding Python object.

            The dictionaries are cleared every time this method is run.

        """
        # get information about arguments
        argspec = inspect.getargspec(self.pyfn)
        callargs = inspect.getcallargs(self.pyfn, *args, **kwargs)

        # collect arguments, sorted in calling order. Note that arg_dict
        # includes both positional and keyword args. The only variables
        # not included explicitly are varargs, which are stored under the
        # appropriate keyword as a tuple. Varargs must come first, if present.
        arg_dict = OrderedDict()
        for arg in argspec.args:
            arg_dict[arg] = callargs[arg]
        var_args = callargs.get(argspec.varargs, [])

        # check for small ints
        arg_dict, var_args = self._small_int_check(arg_dict, var_args)

        # clear symbolic dictionaries
        self.s_vars.clear()
        self.s_args.clear()
        self.s_results.clear()

        # trace the function
        c = Context()
        results = c.call(self.pyfn, tuple(arg_dict.values() + var_args))

        # collect symbolic variables in s_vars
        self.s_vars.update(c.svars)

        # collect symbolic arguments in s_args
        for name, arg in arg_dict.iteritems():
            try:
                self.s_args[name] = self.s_vars[id(arg)]
                self.s_args[name].name = name
            except KeyError:
                raise KeyError('Unable to trace argument '
                               '\'{0}\'.'.format(name))
            except:
                raise

        # collect symbolic variable positional arguments in s_args
        if argspec.varargs is not None:
            va = argspec.varargs
            self.s_args[va] = ()
            for i, arg in enumerate(var_args):
                try:
                    self.s_args[va] += (self.s_vars[id(arg)],)
                    self.s_args[va][-1].name = '{0}_{1}'.format(va, i)
                except KeyError:
                    raise KeyError('Unable to trace variable argument '
                                   'at position {0}.'.format(i))
                except:
                    raise

        # collect symbolic results in s_results
        if not isinstance(results, tuple):
            results = [results]
        for i, r in enumerate(results):
            try:
                self.s_results[id(r)] = self.s_vars[id(r)]
            except KeyError:
                raise KeyError('Unable to trace result #{0} '
                               '(indexed from 1).'.format(i + 1))
            except:
                raise


class Function(Symbolic):
    def __init__(self, pyfn):
        super(Function, self).__init__(pyfn)
        self._fn = None

    @property
    def fn(self):
        return self._fn

    def _compile_function(self, args, kwargs):
        argspec = inspect.getargspec(self.pyfn)

        # trace the function
        self.trace(*args, **kwargs)

        # collect givens
        givens = OrderedDict()
        for name, arg in self.s_args.iteritems():
            # check for the varargs tuple
            if name != argspec.varargs:
                givens[arg] = arg.type(name='{0}'.format(arg.name))
            else:
                givens.update(
                    (v, v.type(name='{0}_{1}'.format(argspec.varargs, i)))
                    for i, v in enumerate(arg))

        # collect inputs
        defaults = dict()
        if argspec.defaults:
            default_slice = slice(-len(argspec.defaults),
                                  -1 if argspec.varargs else None)
            defaults.update(zip(argspec.args[default_slice],
                                argspec.defaults))
        inputs = [theano.Param(
            givens[arg], default=defaults.get(name), name=name)
            for name, arg in self.s_args.iteritems()
            if name is not argspec.varargs]

        inputs.extend(givens[a] for a in self.s_args.get(argspec.varargs, ()))

        # collect outputs
        outputs = self.s_results.values()
        if len(outputs) == 1:
            outputs = outputs[0]

        self._fn = theano.function(inputs=inputs,
                                   outputs=outputs,
                                   givens=givens,
                                   on_unused_input='ignore')

    def __call__(self, *args, **kwargs):
        if self._fn is None:
            self._compile_function(args, kwargs)

        argspec = inspect.getargspec(self.pyfn)
        callargs = inspect.getcallargs(self.pyfn, *args, **kwargs)

        arg_dict = OrderedDict()
        for arg in argspec.args:
            arg_dict[arg] = callargs[arg]
        var_args = list(callargs.get(argspec.varargs, ()))

        return self.fn(*(arg_dict.values() + var_args))



class Gradient(object):
    def __init__(self, fn):
        self._fn = fn
        self._grad_fn = None

    @property
    def fn(self):
        return self._fn

    @property
    def grad_fn(self):
        return self._grad_fn

    def __call__(self, *args, **kwargs):
        # TODO: convert small ints to arrays to allow tracking, but
        # watch out for functions that require int arguments
        if self.grad_fn is None:
            ctxt = Context()
            result = ctxt.call(self.fn, args, kwargs)

            try:
                s_args = [ctxt.svars[id(a)] for a in args]
                s_kwargs = [ctxt.svars[id(v)] for v in kwargs.values()]
                s_result = ctxt.svars[id(result)]
            except KeyError:
                print 'ERROR: PyAD was unable to trace the requested variable.'
                raise
            except:
                raise
            grad = tt.grad(s_result, s_args + s_kwargs)
            if len(grad) == 1:
                grad = grad[0]

            self._grad_fn = theano.function([s_args + s_kwargs], grad)
            self._sym_grad = grad

        all_args = args + tuple(kwargs.values())
        try:
            grad_result = self.grad_fn(*all_args)
        except:
            self._grad_fn = None
            raise
        return grad_result

