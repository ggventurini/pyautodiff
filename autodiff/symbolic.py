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

    def _small_int_check(self, arg_dict):
        """
        Replace any small integer arguments NumPy ints.

        CPython caches and reuses small integers (-5 <= i <= 256), meaning
        there is no way to differentiate them while tracing a function.
        """
        varargs_name = inspect.getargspec(self.pyfn).varargs

        new_arg_dict = OrderedDict()
        for k, v in arg_dict.iteritems():
            # if handling varargs, need to check inside the tuple
            if k == varargs_name:
                varargs = [np.array(a) if isinstance(a, int) else a for a in v]
                new_arg_dict[k] = tuple(varargs)
            elif type(v) is int and -5 <= v <= 256:
                new_arg_dict[k] = np.array(v)
            else:
                new_arg_dict[k] = v

        return new_arg_dict

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
        _small_int_check = kwargs.pop('_small_int_check', True)

        # get information about arguments
        argspec = inspect.getargspec(self.pyfn)
        callargs = inspect.getcallargs(self.pyfn, *args, **kwargs)

        # collect arguments, sorted in calling order. Note that arg_dict
        # includes both positional and keyword args. The only variables
        # not included explicitly are varargs, which are stored under the
        # appropriate keyword as a tuple. Varargs must come first, if present.
        if argspec.varargs in callargs:
            arg_dict = OrderedDict(
                [(argspec.varargs, callargs[argspec.varargs])])
        arg_dict.update(OrderedDict((a, callargs[a]) for a in argspec.args))

        # check for small ints
        if _small_int_check:
            arg_dict = self._small_int_check(arg_dict)

        # clear symbolic dictionaries
        self.s_vars.clear()
        self.s_args.clear()
        self.s_results.clear()

        # trace the function
        c = Context()
        var_args = arg_dict.pop(argspec.varargs, ())
        results = c.call(self.pyfn, var_args, arg_dict)

        # collect symbolic variables in s_vars
        self.s_vars.update(c.svars)

        # collect symbolic variable positional arguments in s_args
        if argspec.varargs is not None:
            self.s_args[argspec.varargs] = [self.s_vars[id(a)] for a in var_args]

        # collect symbolic arguments in s_args
        for name, arg in arg_dict.iteritems():
            try:
                self.s_args[name] = self.s_vars[id(arg)]
            except KeyError:
                if isinstance(name, int):
                    raise ValueError('Unable to trace variable '
                                     'positional argument {0}.'.format(name))
                else:
                    raise ValueError('Unable to trace argument '
                                     '\'{0}\'.'.format(name))
            except:
                raise

        # collect symbolic results in s_results
        if not isinstance(results, tuple):
            results = [results]
        for i, r in enumerate(results):
            try:
                self.s_results[id(r)] = self.s_vars[id(r)]
            except KeyError:
                raise ValueError('Unable to trace result #{0} '
                                 '(indexed from 1).'.format(i + 1))
            except:
                raise


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

