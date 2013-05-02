import __builtin__
import copy
import numpy as np
import theano
import theano.tensor as tt
from inspect import getargspec

from autodiff.context import Context
from autodiff.compat import OrderedDict
import autodiff.utils as utils


class Symbolic(object):
    def __init__(self, pyfn, borrow=None, force_floatX=False):
        """
        Arguments
        ---------

        pyfn : Python function
            The function that will be traced. The will attempt to build
            symbolic representations of any variables referenced in or created
            by this function.

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
        # make deepcopy of pyfn because we might change its defaults
        self._pyfn = copy.deepcopy(pyfn)

        self.s_vars = OrderedDict()
        self.s_args = OrderedDict()
        self.s_results = OrderedDict()
        self._cache = dict()
        self.force_floatX = force_floatX
        self.borrow = utils.as_seq(borrow)

        # replace integer defaults in pyfn to avoid problems
        if self._pyfn.func_defaults:
            a = getargspec(self._pyfn)
            new_defaults = []
            for n, d in zip(reversed(a.args), reversed(a.defaults)):
                if (n not in self.keep_int
                    and type(d) is int
                    and -5 <= d <= 256):
                    new_defaults.append(np.int_(d))
                else:
                    new_defaults.append(d)
            self._pyfn.func_defaults = new_defaults

    def __call__(self, *args, **kwargs):
        return self.get_theano_vars(args, kwargs)

    @property
    def pyfn(self):
        return self._pyfn

    @property
    def cache(self):
        return self._cache

    def trace(self, args=None, kwargs=None):
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
        if args is None:
            args = ()
        elif not isinstance(args, tuple):
            raise TypeError('args must be a tuple')

        if kwargs is None:
            kwargs = dict()
        elif not isinstance(kwargs, dict):
            raise TypeError('args must be a dict')

        # check for small ints and collections
        def check(name, i):
            # Check argument i (with name 'name') for small ints or
            # collections.  If it is a small int, replace it with a numpy int.
            # If a collection, raise a helpful error.
            #
            # This is required because:
            #     1. PyAutoDiff can not shadow CPython ints because they are
            #     cached objects that reuse ids.
            #
            #     2. Theano functions can not accept arguments that are
            #     collections.
            if type(i) is int and -5 <= i <= 256:
                return np.int_(i)
            elif isinstance(i, (list, tuple, dict)):
                raise TypeError('Function arguments can not be '
                                'containers (received {0} for '
                                'argument \'{1}\').'.format(i, name))
            else:
                return i

        argspec = getargspec(self.pyfn)
        tmp_args = tuple(check(n, a) for n, a in zip(argspec.args, args))
        args = tmp_args + tuple(check(argspec.varargs, a)
                                for a in args[len(argspec.args):])
        kwargs = OrderedDict((k, check(k, v)) for k, v in kwargs.iteritems())

        # clear symbolic dictionaries
        self.s_vars.clear()
        self.s_args.clear()
        self.s_results.clear()

        # trace the function
        c = Context(borrowable=self.borrow, force_floatX=self.force_floatX)
        results = c.call(self.pyfn, args, kwargs)

        # collect symbolic variables in s_vars
        self.s_vars.update(c.svars)

        # collect symbolic arguments in s_args
        callargs = utils.orderedcallargs(self.pyfn, *args, **kwargs)

        for name, arg in callargs.iteritems():

            # collect variable args
            if name == argspec.varargs:
                self.s_args[name] = ()
                for i, a in enumerate(arg):
                    try:
                        self.s_args[name] += (self.s_vars[id(a)],)
                        self.s_args[name][-1].name = '{0}_{1}'.format(name, i)
                    except KeyError:
                        raise KeyError('Unable to trace item {0} of variable '
                                       'argument \'{1}\'.'.format(i, name))
                    except:
                        raise

            # collect variable kwargs
            elif name == argspec.keywords:
                for n, a in arg.iteritems():
                    try:
                        self.s_args[n] = self.s_vars[id(a)]
                        self.s_args[n].name = n
                    except KeyError:
                        raise KeyError('Unable to trace argument '
                                       '\'{0}\'.'.format(n))
                    except:
                        raise

            # collect positional args
            else:
                try:
                    self.s_args[name] = self.s_vars[id(arg)]
                    self.s_args[name].name = name
                except KeyError:
                    raise KeyError('Unable to trace argument '
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
                raise KeyError('Unable to trace result #{0} '
                               '(indexed from 1).'.format(i + 1))
            except:
                raise

    def get_theano_vars(self, args, kwargs):
        """
        Returns a dict containing inputs, outputs and givens corresponding to
        the Theano version of the pyfn.

        """

        argspec = getargspec(self.pyfn)

        # trace the function
        self.trace(args, kwargs)

        defaults = dict()
        if argspec.defaults:
            defaults.update(zip(reversed(argspec.args),
                                reversed(argspec.defaults)))

        # ========== collect inputs, givens
        inputs, givens = OrderedDict(), OrderedDict()
        for name, arg in self.s_args.iteritems():
            if name != argspec.varargs:
                givens[arg] = arg.type(name=arg.name)
                inputs[name] = theano.Param(givens[arg],
                                            default=defaults.get(name, None),
                                            name=name)
            else:
                for i, a in enumerate(arg):
                    givens[a] = a.type(name='{0}_{1}'.format(name, i))
                    inputs[i] = givens[a]

        # ========== collect outputs
        outputs = OrderedDict(enumerate(self.s_results.values()))

        theano_vars = {'inputs': inputs,
                       'outputs': outputs,
                       'givens': givens}

        return theano_vars

    def get_symbolic_arg(self, x):
        """
        Retrieve the symbolic version of x.

        x : python object or string

        If x is a string, it is matched to the names of the function arguments.
        If x is an object, it must have been traced by the Symbolic class.
        If x is a small int, raises an error.
        """
        if type(x) is int and -5 <= x <= 256:
            raise ValueError('Small integer arguments can not be traced '
                             'selectively. Either recast or redesign your '
                             'function.')
        elif isinstance(x, basestring):
            if x in self.s_args:
                return self.s_args[x]
            else:
                raise ValueError('Requested the symbolic variable '
                                 'shadowing object {0}, but it was '
                                 'not traced.'.format(repr(x)))
        else:
            if id(x) in self.s_vars:
                return self.s_vars[id(x)]
            else:
                raise ValueError('Requested the symbolic variable '
                                 'shadowing object {0}, but it was '
                                 'not traced.'.format(repr(x)))


class Function(Symbolic):

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def compile_function(self, args, kwargs):
        theano_vars = self.get_theano_vars(args, kwargs)

        inputs = theano_vars['inputs'].values()
        outputs = theano_vars['outputs'].values()
        givens = theano_vars['givens']

        if len(outputs) == 1:
            outputs = outputs[0]

        # compile function
        fn = theano.function(inputs=inputs,
                             outputs=outputs,
                             givens=givens,
                             on_unused_input='ignore')

        # store in cache corresponding to the number of positional inputs
        argspec = getargspec(self.pyfn)
        callargs = utils.orderedcallargs(self.pyfn, *args, **kwargs)
        self.cache[len(callargs.get(argspec.varargs, ()))] = fn

        return fn

    def call(self, *args, **kwargs):
        argspec = getargspec(self.pyfn)
        callargs = utils.orderedcallargs(self.pyfn, *args, **kwargs)

        # try to retrieve function from cache; otherwise compile
        fn = self.cache.get(len(callargs.get(argspec.varargs, ())),
                            self.compile_function(args, kwargs))

        pos_args = [callargs[arg] for arg in argspec.args]
        pos_args.extend(callargs.get(argspec.varargs, ()))
        kw_args = callargs.get(argspec.keywords, {})

        return fn(*pos_args, **kw_args)


class Gradient(Function):

    def __init__(self, pyfn, wrt=None, borrow=None, force_floatX=False):
        super(Gradient, self).__init__(pyfn=pyfn,
                                       borrow=borrow,
                                       force_floatX=force_floatX)
        self.wrt = utils.as_seq(wrt)

    def compile_function(self, args, kwargs):
        theano_vars = self.get_theano_vars(args, kwargs)

        inputs = theano_vars['inputs'].values()
        outputs = theano_vars['outputs'].values()
        givens = theano_vars['givens']

        # get wrt variables. If none were specified, use inputs.
        if len(self.wrt) == 0:
            wrt = givens.keys()
        else:
            wrt = [self.get_symbolic_arg(w) for w in self.wrt]

        grads = [tt.grad(o, wrt=wrt) for o in outputs]

        if len(grads) == 1:
            grads = grads[0]

        if __builtin__.any(w.dtype[:3] == 'int' for w in wrt):
            print ('WARNING: the gradient with respect to an '
                   'int is defined as 0.')

        # compile function
        fn = theano.function(inputs=inputs,
                             outputs=grads,
                             givens=givens,
                             on_unused_input='ignore')

        # store in cache corresponding to the number of positional inputs
        argspec = getargspec(self.pyfn)
        callargs = utils.orderedcallargs(self.pyfn, *args, **kwargs)
        self.cache[len(callargs.get(argspec.varargs, ()))] = fn

        return fn
