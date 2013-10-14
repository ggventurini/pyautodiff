import logging
import meta
from ast import *
import numpy as np
import types
import theano
import theano.tensor as T

import autodiff.utils as utils
import autodiff.functions

# import with triple underscore to use [hopefully] safely in ASTs
# basically, need to make sure that these modules aren't overwritten
import autodiff.utils as ___utils
import theano.tensor as ___T
import autodiff.functions as ___functions

logger = logging.getLogger('autodiff')

# XXX FIXME This will not do - seed must be exposed.
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
global_randomstreams = RandomStreams(seed=12345)#np.random.randint(1, 999999))


def get_ast(func, flags=0):
    func_def = meta.decompiler.decompile_func(func)
    if isinstance(func_def, Lambda):
        func_def = FunctionDef(
            name='<lambda>', args=func_def.args,
            body=[Return(func_def.body)],
            decorator_list=[])
    assert isinstance(func_def, FunctionDef)
    return func_def


def get_source(ast):
    if hasattr(ast, 'func_code'):
        ast = get_ast(ast)
    elif callable(ast):
        ast = get_ast(ast.__call__)
    return meta.asttools.dump_python_source(ast)


def print_ast(ast):
    if hasattr(ast, 'func_code'):
        ast = get_ast(ast)
    elif callable(ast):
        ast = get_ast(ast.__call__)
    meta.asttools.print_ast(ast)


def print_source(ast):
    if hasattr(ast, 'func_code'):
        ast = get_ast(ast)
    elif callable(ast):
        ast = get_ast(ast.__call__)
    meta.asttools.python_source(ast)


def compile_func(ast, new_globals=None, file_name=None):
    global_dict = globals().copy()
    if new_globals is not None:
        global_dict.update(new_globals)
    if file_name is None:
        file_name = '<Context-AST>'
    if not isinstance(ast, FunctionDef):
        ast = fix_missing_locations(FunctionDef(name='<tmp_fn>',
                                                args=arguments(args=[],
                                                               defaults=[],
                                                               kwarg=None,
                                                               vararg=None),
                                                body=[Return(ast)],
                                                decorator_list=[]))
    return meta.decompiler.compile_func(ast, file_name, global_dict)


def escape(x):
    def _escape(x):
        if utils.isvar(x):
            try:
                return x.eval()
            except:
                raise ValueError('Could not escape {0}'.format(x))
        elif isinstance(x, ShadowClass):
            return x._obj__
        else:
            return x
    return utils.unflatten(x, [_escape(i) for i in utils.flatten(x)])


def simple_Call(func, args):
    """
    Simple alias for building Call nodes that doesn't require specification of
    keywords, kwargs or starargs.
    """
    if not isinstance(args, (list, tuple)):
        args = [args]
    call = Call(args=args,
                func=func,
                keywords=[],
                kwargs=None,
                starargs=None)
    return call


def isvar_ast(name):
    """
    Wraps a Name node in a call to utils.isvar.
    """
    isvar = simple_Call(args=utils.as_seq(name),
                        func=Attribute(attr='isvar',
                                       ctx=Load(),
                                       value=Name(ctx=Load(), id='___utils')))
    return isvar



class Context(object):

    def __init__(self, borrowable=()):
        self.s_vars = dict()
        self.tags = dict()
        # FIXME do we need to hold on to all of these itermediates?
        # ensure these id's do not get recycled by garbage collection
        self._nogc = []
        self._noshadow = set()
        self._top_node = None
        self.borrowable = [id(b) for b in borrowable]

    def recompile(self, f, nested=False):
        """
        Accepts a function f that operates on numerical objects and
        returns a function that operates on Theano objects.

        nested : bool
            `recompile` resets the context and sets the 'top_node' of the
            function, which helps in tracing arguments. By passing nested=True,
            this reset can be bypassed. This is used, for example, when
            transforming nested functions. In this case, we want to use the
            same context but keep it when calling recompile.
        """
        transformer = TheanoTransformer(context=self)
        f_ast = get_ast(f)

        if not nested:
            self._top_node = f_ast
            self.tags.clear()

        transformed_ast = fix_missing_locations(transformer.visit(f_ast))

        func_globals = f.func_globals.copy()
        func_globals.update({'___ctx' : transformer})
        if f.func_closure:
            func_globals.update((v, transformer.shadow(c.cell_contents)) for v, c in
                                zip(f.func_code.co_freevars, f.func_closure))

        for name in f.func_code.co_names:
            if name in func_globals.iterkeys():
                func_globals[name] = transformer.shadow(func_globals[name])

        new_f = compile_func(transformed_ast, func_globals, repr(self))

        if isinstance(f, types.MethodType):
            new_f = types.MethodType(new_f, f.im_self, f.im_class)

        return new_f

    def get_symbolic(self, x):
        """
        Attempts to retrieve the symbolic version of x.

        if x is an numeric object (int, float, numpy array), it must have been
        traced by the context during recompiled function execution.

        if x is a string, it must have been tagged with
        autodiff.functions.tag().
        """
        if isinstance(x, basestring):
            if x in self.s_vars:
                return self.s_vars[x]
            elif x in self.tags:
                return self.tags[x]
            else:
                raise ValueError(
                    'Requested the symbolic variable of tag `{0}`'
                    ', but `{0}` was not tagged.'.format(x))
        elif utils.isvar(x):
            return x
        elif id(x) in self.s_vars:
            return self.s_vars[id(x)]
        elif isinstance(x, int) and -5 <= x <= 256:
            raise ValueError(
                'Small integers (-5 <= x <= 256) can not be shadowed due to '
                'CPython caching. Try casting the variable as a NumPy int type '
                'or array before tracing: {0}'.format(x))
        else:
            raise ValueError(
                'Requested the symbolic variable shadowing object {0}'
                ', but it was not traced.'.format(repr(x)))

    def reset(self):
        self.s_vars.clear()
        self.tags.clear()
        self._nogc = []
        self._noshadow = set()
        self._top_node = None


class ShadowClass(object):
    """
    A class that [almost] transparently wraps other objects, shadowing any
    requested attributes and function calls. Attributes can not be set.
    """
    __wraps___ = None
    __ignore___ = ['__class__',
                  '__mro__',
                  '__repr__',
                  '__new__',
                  '__init__',
                  '__dict__',
                  '__call__',
                  '__name__',
                  '__setattr__',
                  '__getattr__',
                  '__getattribute__',
                  '__shadow__']

    def __init__(self, obj, context):
        if isinstance(obj,  type):
            raise TypeError()
        if self.__wraps___ is None:
            raise TypeError('__wraps__ not set.')
        assert isinstance(obj, self.__wraps___)
        self.__dict__['_obj__'] = obj
        self.__dict__['_transformer__'] = TheanoTransformer(context)

    def __setattr__(SetComp, name, value):
         raise TypeError(
            'To protect code integrity, attributes of shadowed objects '
            'can not be set: Shadowed {0}'.format(self._obj__))

    def __getattr__(self, name):
        attr = getattr(self._obj__, name)
        return self._transformer__.shadow(attr)

    def __call__(self, *args, **kwargs):
        handle_functions = self._transformer__.handle_functions
        rval = handle_functions(self._obj__).__call__(*args, **kwargs)
        return self._transformer__.shadow(rval)

    class __metaclass__(type):
        def __init__(cls, name, bases, dct):

            def make_proxy(name):
                def proxy(self, *args):
                    print name
                    return self._transformer__.shadow(getattr(self._obj__, name))
                return proxy

            type.__init__(cls, name, bases, dct)
            if cls.__wraps___:
                for name in dir(cls.__wraps___):
                    if name.startswith("__"):
                        if (name not in cls.__ignore___
                            and name not in dct):
                            attr = getattr(cls.__wraps___, name, None)
                            try:
                                setattr(cls, name, property(make_proxy(name)))
                            except:
                                pass

class TheanoTransformer(NodeTransformer):

    def __init__(self, context):
        super(TheanoTransformer, self).__init__()
        self.context = context

    def ast_wrap(self, method_name, args):
        """
        Allows Python methods to be applied to AST nodes at runtime.

        `method_name` is a method of the TheanoTransformer class that accepts
        Python objects as arguments.

        `args` are the AST nodes representing the arguments for `method_name`
        (not including `self`!).

        ast_wrap returns an `ast.Call()` node which calls the method on the
        specified arguments at runtime.
        """
        wrapped = simple_Call(func=Attribute(attr=method_name,
                                              ctx=Load(),
                                              value=Name(ctx=Load(),
                                                         id='___ctx')),
                               args=args)

        return wrapped

    # ** --------------------------------------------------------
    # ** Direct Manipulation (Methods)

    def shadow(self, args):
        """
        Helper function for `_shadow` that calls it on a flattened version of
        its argument.
        """
        shadow_vars = [self._shadow_inner(x) for x in utils.flatten(args)]
        return utils.unflatten(args, shadow_vars)

    def _shadow_inner(self, x):

        """
        Given a numerical variable x, return an equivalent Theano shared
        variable and store the relationship in self.s_vars. Otherwise return x.
        """

        if id(x) in self.context._noshadow:
            return x

        if utils.isvar(x):
            return x

        if isinstance(x, (int, float, np.ndarray)):
            # take special care with small ints, because CPython caches them.
            if isinstance(x, int) and -5 <= x <= 256:
                x = np.int_(x)

            if getattr(x, 'dtype', None) == bool:
                logger.info('Warning: Theano has no bool type; upgrading to int8.')
                x = x.astype('int8')

            if id(x) not in self.context.s_vars:
                # add to _nogc to ensure that the id won't be reused
                self.context._nogc.append(x)
                # create symbolic version:
                if isinstance(x, np.ndarray) and id(x) in self.context.borrowable:
                    sym_x = theano.shared(x, borrow=True)
                else:
                    sym_x = theano.shared(x)
                # store symbolic version
                self.context.s_vars[id(x)] = sym_x
                # return symbolic version
                return sym_x
            else:
                return self.context.s_vars[id(x)]
        else:
            if isinstance(x, type):
                return x
            if isinstance(x, ShadowClass):
                return x
            else:
                class Shadow(ShadowClass):
                    __wraps___ = x.__class__
                return Shadow(x, self.context)

    def update_inplace(self, obj, new_value):
        """
        Object `obj` is updated inplace with value `value`; they must be
        compatible. This is basically a hack to mimic inplace operations.
        """
        raise ValueError('cant update inplace')

    def handle_functions(self, func):
        """
        Given some function for, return another function.

        Generally used to exchange NumPy functions for Theano equivalents.
        """
        # ** ======================= first handle functions defined here!

        if getattr(func, '__module__', None) == __name__:
            return func

        # ** ======================= special autodiff functions

        elif func is autodiff.functions.escape:
            # escapes a variable from Tensor representation
            return escape

        elif func is autodiff.functions.escaped_call:
            # call a function on escaped arguments, without transforming the AST
            def escaped_call(fn, *args, **kwargs):
                esc_args = utils.unflatten(
                    args, [escape(a) for a in utils.flatten(args)])
                esc_kwargs = utils.unflatten(
                    kwargs, [escape(a) for a in utils.flatten(kwargs)])
                return fn(*esc_args, **esc_kwargs)
            return escaped_call

        elif func is autodiff.functions.tag:
            def tag(obj, tag):
                assert isinstance(tag, basestring)
                if tag in self.context.s_vars:
                    logger.warning(
                        '{0} was tagged as {1}, but {1} is a top-level '
                        'function argument. The tag will not be '
                        'available.'.format(self, obj, tag))
                else:
                    if tag in self.context.tags:
                        logger.warning(
                            '{0} was tagged as {1}, but {1} was already '
                            'tagged. Note that the new tag will overwrite '
                            'the old one.'.format(self, obj, tag))

                    self.context.tags[tag] = obj
                return obj
            return tag


        # ** ======================= __theano_op__

        elif hasattr(func, '__theano_op__'):
            return func.__theano_op__

        # ** ======================= array methods (with tensor instances)

        elif utils.isvar(getattr(func, '__self__', None)):
            return self.handle_array_methods(func.__self__, func.__name__)

        # ** ======================= Theano function

        elif (getattr(func, '__module__', None)
              and getattr(func, '__module__').startswith('theano')):
            return func

        # ** ======================= type/casting functions

        elif type(func) is type:
            if func.__name__ in ['bool', 'bool_', 'bool8']:
                logger.info('Warning: Theano has no bool type; '
                            'upgrading to int8.')
                def bool_(x):
                    return T.neq(x, 0)
                return bool_
            elif func.__name__ in T.basic._cast_mapping.keys():
                def cast(x):
                    return T.cast(x, dtype=func.__name__)
                return cast
            elif func.__name__ == 'float':
                def float_(x):
                    return T.cast(x, dtype=theano.config.floatX)
                return float_
            elif func.__name__ == 'int':
                def int_(x):
                    return T.cast(x, dtype='int' + theano.config.floatX[-2:])
                return int_
            elif func.__name__ == 'enumerate':
                def enumerate_(iterable, start=0):
                    if utils.isvar(iterable):
                        raise TypeError(
                            'Called enumerate() on Tensor {0} but Tensors '
                            'do not support iteration. Maybe try escaping '
                            'the tensor?'.format(iterable))
                    else:
                        return enumerate(iterable, start=start)
                return enumerate_
            else:
                def new_type(*args, **kwargs):
                    try:
                        return self.shadow(func(*args, **kwargs))
                    except:
                        raise ValueError('Unsupported type: {0}'.format(func))
                return new_type

        # ** ======================= numpy functions

        elif (getattr(func, '__module__', None)
              and getattr(func, '__module__').startswith('numpy')
              or isinstance(func, np.ufunc)
              or str(func) == '<built-in function max>'
              or str(func) == '<built-in function min>'):

            # abs
            if func.__name__ in ('abs', 'absolute'):
                return abs

            # ones/zeros
            # FIXME submitted a PR to Theano to make syntax more
            # like Numpy; this change shouldn't be needed afterward.
            elif func.__name__ in ('ones', 'zeros'):
                def alloc(shp, dtype=None):
                    if (not isinstance(shp, (list, tuple))
                            and not utils.isvar(shp)):
                        shp = [shp]
                    return getattr(T, func.__name__)(shp, dtype)
                return alloc

            elif hasattr(T, func.__name__):
                return getattr(T, func.__name__)
            else:
                raise ValueError('Unsupported function: {0}'.format(func))

        # ** ======================= built-ins

        elif '<built-in' in str(func):

            # ranges
            if func.__name__ in ('range', 'xrange'):
                def range_(*args):
                    return func(*(escape(a) for a in args))
                return range_

            # zip
            elif func.__name__ == 'zip':
                def zip_(*args):
                    if __builtin__.any(utils.isvar(a) for a in args):
                        raise TypeError(
                            'Called zip() on Tensor but Tensors '
                            'do not support iteration. Maybe try escaping '
                            'the tensor?')
                    else:
                        return zip(*args)
                return zip_

            # uniform random numbers (np.random.uniform)
            elif 'method uniform of mtrand.RandomState' in str(func):
                def rand_u(low=0.0, high=1.0, size=1):
                    return global_randomstreams.uniform(low=low,
                                                        high=high,
                                                        size=size)
                return rand_u

            # standard uniform random numbers (np.random.random)
            elif ('method random of mtrand.RandomState' in str(func)
                  or 'method random_sample of mtrand.RandomState' in str(func)):
                def rand_u(size):
                    return global_randomstreams.uniform(size=size)
                return rand_u

            # normal random numbers (np.random.normal)
            elif 'method normal of mtrand.RandomState' in str(func):
                def rand_n(loc=0.0, scale=1.0, size=None):
                    return global_randomstreams.normal(avg=loc,
                                                       std=scale,
                                                       size=size)
                return rand_n

            # standard normal random numbers (np.random.randn)
            elif 'method randn of mtrand.RandomState' in str(func):
                def rand_n(*size):
                    return global_randomstreams.normal(size=size)
                return rand_n

            # binomial random numbers (np.random.binomial)
            elif 'method binomial randn of mtrand.RandomState' in str(func):
                def rand_b(n, p, size=None):
                    return global_randomstreams.binomial(n=n, p=p, size=size)
                return rand_b

            # isinstance
            elif func is isinstance:
                def isinstance_(obj, types):
                    escaped_obj = escape(obj)
                    if obj.ndim == 0:
                        escaped_obj = np.asscalar(escaped_obj)
                    return isinstance(escaped_obj, escape(types))
                return isinstance_

            # anything else (like getattr)
            else:
                return func

        # ** ======================= Misc

        elif (('ipdb' in getattr(func, '__module__', '')
              or 'pdb' in getattr(func, '__module__', ''))
              and func.__name__ == 'set_trace'):
            return func

        # ** ======================= Anything else

        else:
            try:
                return self.context.recompile(func, nested=True)
            except:
                raise ValueError('Unsupported function: {0}'.format(func))

        # ** ======================= Catchall (shouldn't be called)

        raise ValueError(
            'handle_functions: No case matched function {0}'.format(func))

    def handle_array_methods(self, var, method_name):
        """
        This method is called whenever:
            1. An array method is requested that doesn't exist for Theano
               variables (like _.swapaxes()). `handle_array_methods` is used
               to supply a replacement method. Note that in this case,
               `handle_array_methods` is called directly.
            2. A method is requested that DOES exist for Theano variables. In
               this case, `handle_array_methods` is called by `handle_functions`
               prior to calling the method. `handle_array_methods` is used to
               supply a replacement function that properly handles the supplied
               arguments (since they are compliant with the Numpy signature,
               not the Theano one).
        """
        # if we're not dealing with a Theano variable, nothing to do here.
        if not utils.isvar(var):
            return getattr(var, method_name)

        # Theano's reshape requires dim to be in a collection, unlike Numpy.
        if method_name == 'reshape':
            def reshape(*args, **kwargs):
                if not isinstance(args[0], (list, tuple)):
                    args = [args]
                return var.reshape(*args, **kwargs)
            return reshape

        # Theano has no swapaxes method
        elif method_name == 'swapaxes':
            def swapaxes(*args, **kwargs):
                axis1, axis2 = (escape(a) for a in args)
                dims = range(var.ndim)
                dims[axis1], dims[axis2] = dims[axis2], dims[axis1]
                return var.dimshuffle(*dims)
            return swapaxes

        # Theano doesn't process numpy dtype objects or 'bool'
        elif method_name == 'astype':
            def astype(*args, **kwargs):
                dtype = kwargs.pop('dtype', None)
                if not dtype:
                    dtype = args[0]
                if not isinstance(dtype, str):
                    # get numpy dtype objects like np.float32
                    try:
                        dtype = dtype.__name__
                    except:
                        raise NotImplementedError(
                            'Unsupported dtype: {0}'.format(dtype))
                if 'bool' in dtype:
                    dtype = 'int8'
                    logger.info('Warning: Theano has no bool type; '
                                'upgrading to int8.')
                return var.astype(dtype)
            return astype

        elif method_name == 'sort':
            def sort_(*args, **kwargs):
                raise ValueError(
                    'Calling an array\'s `sort()` method is not supported '
                    'because in NumPy it is an inplace operation, but in '
                    'Theano it is not. Please use numpy.sort() instead.')
            return sort_

        # ...Otherwise, try to access the method on the Theano variable
        else:
            return getattr(var, method_name)

    def handle_comparison(self, operator, left, right):
        """
        This method is called whenever an 'eq' or 'neq' operator is
        encountered with a single rhs comparator, since tensors do not properly
        resolve == and !=.
        """
        if utils.isvar(left) or utils.isvar(right):
            return getattr(T, operator)(left, right)
        elif operator == 'eq':
            return left == right
        elif operator == 'neq':
            return left != right
        else:
            raise ValueError(
                'Not sure how to handle operator: {0}'.format(operator))

    # ** --------------------------------------------------------
    # ** AST Manipulation (Node Visitors)

    def visit_Assign(self, node):
        """
        Applies the following transformations:

        - Transform subscripts. Tensor variables do not support inplace
          assignment, so subscript assigns must be changed to call the
          `set_subtensor` function.

            Statements of the form:
                x[a:b][c] = y
            Become:
                if utils.isvar(x):
                    x = T.set_subtensor(x[a:b], T.set_subtensor(x[a:b][c], y))
                else:
                    x[a:b][c] = y

        """
        self.generic_visit(node)

        # handle subscripted assignment for tensor variables
        if isinstance(node.targets[0], Subscript):

            # helper function to transform subscript into (possibly nested)
            # T.set_subtensor statements
            def build_subt(subscript, value):
                subscript_load = Subscript(ctx=Load(),
                                           slice=subscript.slice,
                                           value=subscript.value)
                set_subtensor = simple_Call(
                    args=[subscript_load, value],
                    func=Attribute(attr='set_subtensor',
                                   ctx=Load(),
                                   value=Name(ctx=Load(), id='___T')))
                if isinstance(subscript.value, Subscript):
                    set_subtensor = build_subt(subscript.value, set_subtensor)
                return set_subtensor

            # get root tensor; check for nested subscripts
            tensor = node.targets[0]
            while not isinstance(tensor, Name):
                tensor = tensor.value

            # transform subscript into set_subtensor
            set_subt = build_subt(subscript=node.targets[0], value=node.value)

            # wrap set_subtensor statements in Assign to root tensor
            assign_subtensor = Assign(targets=[Name(ctx=Store(), id=tensor.id)],
                                      value=set_subt)

            # wrap assign_subtensor in If to ensure that the modification
            # is only applied to tensor args
            check_var = If(test=isvar_ast(tensor),
                           body=[assign_subtensor],
                           orelse=[node])
            return check_var
        else:
            return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        new_node = simple_Call(args=[node.value,
                                Str(s=node.attr),
                                self.ast_wrap('handle_array_methods',
                                              [node.value, Str(s=node.attr)])],
                                func=Name(ctx=Load(), id='getattr'))
        return new_node

    def visit_AugAssign(self, node):
        """
        See documentation for self.visit_Assign() for information on
        transformations applied here.
        """

        if isinstance(node.target, Subscript):
            # apply op directly
            value = BinOp(left=Subscript(ctx=Load(),
                                         slice=node.target.slice,
                                         value=node.target.value),
                          right=node.value,
                          op=node.op)

            # farm out the work to visit_Assign
            assign_node = self.visit_Assign(Assign(targets=[node.target],
                                                   value=value))
            return assign_node
        else:
            self.generic_visit(node)
            return node

    def visit_Call(self, node):
        """
        Whenever a function is called, first pass it to the 'handle_functions'
        method. This method examines the function and modifies it prior to
        calling it. For example, it might replace `numpy.ones` with
        `theano.ones`.
        """
        self.generic_visit(node)
        node.func = self.ast_wrap('handle_functions', node.func)
        return node

    def visit_Compare(self, node):
        """
        Replaces comparison operators with Theano functions, if either argument
        is a tensor variable. Only applies to '==' and '!=' since those are not
        handled well by Tensors.

        Given:

            x == y

        Becomes:

            ___ctx.handle_comparison('eq', x, y)

        Which internally performs:

            if utils.isvar(x) or utils.isvar(y):
                T.eq(x, y)
            else:
                x == y

        The function call is required to allow seamless replacement of the
        comparison in places like if statements, where the nested if would raise
        a syntax error.
        """
        self.generic_visit(node)

        if isinstance(node.ops[0], Eq):
            theano_op = Str(s='eq')
        elif isinstance(node.ops[0], NotEq):
            theano_op = Str(s='neq')
        else:
            # Gt, GtE, Lt, LtE, Is, IsNot, In, Not In
            return node

        if len(node.comparators) == 1:
            return self.ast_wrap('handle_comparison',
                                 [theano_op, node.left, node.comparators[0]])
        else:
            return node

    def visit_FunctionDef(self, node):
        """
        When a function is defined, shadow each of its arguments immediately.

        The AST is modified so that a function defined as:

            def f(a, b=None, *c, **d):
                ...

        is changed via this method to:

            def f(a, b=None, *c, **d):
                a = self.shadow(a)
                b = self.shadow(b)
                c = self.shadow(c)
                d = self.shadow(d)
                tag(a, 'a')
                tag(b, 'b')
                for k, v in d.items():
                    tag(v, k)
                ...

        This way, any future references to these variables will access their
        shadowed values. This is important because inplace modifications do
        not always force the `shadow` method to get called, and so the inplace
        changes might not be reflected the next (and first!) time the variable
        is loaded.
        """
        self.generic_visit(node)
        assigns = []
        tags = []

        # shadow and tag args
        for param in node.args.args:
            assigns.append(Assign(
                targets=[Name(ctx=Store(), id=param.id)],
                value=self.ast_wrap('shadow', Name(ctx=Load(), id=param.id))))

            tags.append(Expr(value=simple_Call(
                args=[Name(ctx=Load(), id=param.id), Str(s=param.id)],
                func=Attribute(attr='tag',
                               ctx=Load(),
                               value=Name(ctx=Load(),
                                          id='___functions')))))

        # shadow the varargs
        if node.args.vararg:
            assigns.append(Assign(
                targets=[Name(ctx=Store(), id=node.args.vararg)],
                value=self.ast_wrap('shadow', Name(ctx=Load(),
                                                   id=node.args.vararg))))

        # shadow and tag the kwargs
        if node.args.kwarg:
            assigns.append(Assign(
                targets=[Name(ctx=Store(), id=node.args.kwarg)],
                value=self.ast_wrap('shadow', Name(ctx=Load(),
                                                   id=node.args.kwarg))))

            tags.append(For(
                body=[Expr(value=simple_Call(
                    args=[Name(ctx=Load(), id='v'),
                          Name(ctx=Load(), id='k')],
                    func=Attribute(attr='tag',
                                   ctx=Load(),
                                   value=Name(ctx=Load(),
                                              id='___functions'))))],
                iter=simple_Call(
                    func=Attribute(attr='iteritems',
                                   ctx=Load(),
                                   value=Name(ctx=Load(),
                                              id=node.args.kwarg))),
                orelse=[],
                target=Tuple(ctx=Store(), elts=[Name(ctx=Store(), id='k'),
                                                Name(ctx=Store(), id='v')])))

        if node is self.context._top_node:
            node.body = assigns + tags + node.body
            self.context._top_node = None
        else:
            node.body = assigns + node.body

        return node

    def visit_Name(self, node):
        """
        Whenever a literal variable name is loaded, call the
        'shadow' method on its value.
        """
        self.generic_visit(node)
        # if isinstance(node.ctx, Load):
            # node = self.ast_wrap('shadow', node)
        return node

    def visit_Num(self, node):
        # don't shadow because these are typically function arguments
        # node = self.ast_wrap('shadow', node)
        return node
