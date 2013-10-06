import logging
import meta
import ast as ast_module
import numpy as np
import theano
import theano.tensor as T

import autodiff.utils as utils

logger = logging.getLogger('autodiff')


# XXX FIXME This will not do - seed must be exposed.
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
global_randomstreams = RandomStreams(seed=123)

def isvar(x):
    vartypes = (theano.tensor.sharedvar.SharedVariable,
                theano.tensor.TensorConstant,
                theano.tensor.TensorVariable)
    return isinstance(x, vartypes)

def get_ast(func, flags=0):
    func_def = meta.decompiler.decompile_func(func)
    if isinstance(func_def, ast_module.Lambda):
        func_def = ast_module.FunctionDef(
            name='<lambda>', args=func_def.args,
            body=[ast_module.Return(func_def.body)],
            decorator_list=[])
    assert isinstance(func_def, ast_module.FunctionDef)
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
    if not isinstance(ast, ast_module.FunctionDef):
        ast = ast_module.fix_missing_locations(
            ast_module.FunctionDef(name='<tmp_fn>',
                                   args=ast_module.arguments(args=[],
                                                             defaults=[],
                                                             kwarg=None,
                                                             vararg=None),
                                   body=[ast_module.Return(ast)],
                                   decorator_list=[]))
    return meta.decompiler.compile_func(ast, file_name, global_dict)

def unshadow(x):
    if isvar(x):
        try:
            return x.eval()
        except:
            return x
    else:
        return x

def _simple_call(func, args):
    if not isinstance(args, (list, tuple)):
        args = [args]
    call = ast_module.Call(args=args,
                           func=func,
                           keywords=[],
                           kwargs=None,
                           starargs=None)
    return call


class Context(object):
    def __init__(self, borrowable=()):
        self.s_vars = dict()
        self.inplace_updates = dict()
        # FIXME do we need to hold on to all of these itermediates?
        # ensure these id's do not get recycled by garbage collection
        self._nogc = []
        self._noshadow = set()
        self.borrowable = [id(b) for b in borrowable]

    def transform(self, f):
        t = TheanoTransformer(watcher=self)
        return t.transform(f)

    def recompile(self, f):
        t = TheanoTransformer(watcher=self)
        return t.recompile(f)


class ASTTransformer(ast_module.NodeTransformer):

    def ast_wrap(self, method_name, args):
        """
        Allows Python methods to be applied to AST nodes at runtime.

        `method_name` is a method of the ASTTransformer class that accepts Python
        objects as arguments.

        `args` are the AST nodes representing the arguments for `method_name` (not
        including `self`!).

        ast_wrap returns an `ast.Call()` node which calls the method on the specified
        arguments at runtime.
        """
        wrapped = _simple_call(
            func=ast_module.Attribute(attr=method_name,
                                      ctx=ast_module.Load(),
                                      value=ast_module.Name(ctx=ast_module.Load(),
                                                            id='__C')),
            args=args)

        return wrapped

    def transform(self, f):
        f_ast = get_ast(f)
        transformed_ast = self.visit(f_ast)
        return ast_module.fix_missing_locations(transformed_ast)

    def recompile(self, f):
        ast = self.transform(f)
        func_globals = f.func_globals.copy()
        if f.func_closure:
            func_globals.update((v, c.cell_contents) for v, c in
                                zip(f.func_code.co_freevars, f.func_closure))
        func_globals.update({'__C' : self})
        return compile_func(ast, func_globals, '<Context-AST>')


class TheanoTransformer(ASTTransformer):

    def __init__(self, watcher):
        super(TheanoTransformer, self).__init__()
        self.watcher = watcher

    # ** --------------------------------------------------------
    # ** Direct Manipulation (Methods)

    def shadow(self, args):
        """
        Helper function for `_shadow` that calls it on a flattened version of
        its argument.
        """
        shadow_vars = [self._shadow(x) for x in utils.flat_from_doc(args)]
        return utils.doc_from_flat(args, shadow_vars)

    def _shadow(self, x):
        """
        Given a numerical variable x, return an equivalent Theano shared variable
        and store the relationship in self.s_vars. Otherwise return x.
        """
        if id(x) in self.watcher._noshadow:
            return x

        if not isinstance(x, (int, float, np.ndarray)):
            # if x is a Theano variable, it is possible that it was modified by
            # an inplace Numpy operation, like array.sort(). Theano variables
            # can't be updated inplace, so we keep track of inplace updates in
            # a special dictionary. We check for updates before returning the
            # variable.
            return self.watcher.inplace_updates.get(id(x), x)

        # take special care with small ints, because CPython caches them.
        if isinstance(x, int) and -5 <= x <= 256:
            x = np.int_(x)

        elif isinstance(x, float):
            x = np.float_(x)

        if getattr(x, 'dtype', None) == bool:
            logger.info('Warning: Theano has no bool type; upgrading to int8.')
            x = x.astype('int8')

        if id(x) not in self.watcher.s_vars:
            # add to _nogc to ensure that the id won't be reused
            self.watcher._nogc.append(x)
            # create symbolic version
            sym_x = theano.shared(x)
            # store symbolic version
            self.watcher.s_vars[id(x)] = theano.shared(x)
            # return symbolic version
            return sym_x
        else:
            return self.watcher.s_vars[id(x)]

    def handle_functions(self, func):
        """
        Given some function for, return another function.

        Generally used to exchange NumPy functions for Theano equivalents.
        """
        # ** ======================= first handle functions defined here!

        if getattr(func, '__module__', None) == __name__:
            return func

        # ** ======================= __theano_op__

        elif hasattr(func, '__theano_op__'):
            return func.__theano_op__

        # ** ======================= array methods (with tensor instances)

        elif isvar(getattr(func, '__self__', None)):
            return self.handle_array_methods(func.__self__, func.__name__)

        # ** ======================= Theano function

        elif (getattr(func, '__module__', '').startswith('theano')):
            return func

        # ** ======================= type/casting functions

        elif type(func) is type:
            if func.__name__ in ['bool', 'bool_', 'bool8']:
                logger.info('Warning: Theano has no bool type; upgrading to int8.')
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
                    if isvar(iterable):
                        raise TypeError(
                            'Called enumerate() on Tensor {0} but Tensors '
                            'do not support iteration. Maybe try escaping '
                            'the tensor?'.format(iterable))
                    else:
                        return enumerate(iterable, start=start)
                return enumerate_
            else:
                raise ValueError('Unsupported type: {0}'.format(func))

        # ** ======================= numpy functions

        elif (getattr(func, '__module__', '').startswith('numpy')
               or isinstance(func, np.ufunc)
               or str(func) == '<built-in function abs>'
               or str(func) == '<built-in function max>'
               or str(func) == '<built-in function min>'
               or str(func) == '<built-in function sum>'):
            if func.__name__ in ('abs', 'absolute'):
                return abs
            elif hasattr(T, func.__name__):
                return getattr(T, func.__name__)
            else:
                raise ValueError('Unsupported function: {0}'.format(func))

        # ** ======================= built-ins

        elif 'built-in' in str(func):
            if func.__name__ in ('range', 'xrange'):
                def range_(*args):
                    return func(*(unshadow(a) for a in args))
                return range_
            elif func.__name__ == 'zip':
                def zip_(*args):
                    if __builtin__.any(isvar(a) for a in args):
                        raise TypeError(
                            'Called zip() on Tensor but Tensors '
                            'do not support iteration. Maybe try escaping '
                            'the tensor?')
                    else:
                        return zip(*args)
                return zip_
            elif ('method random of mtrand.RandomState' in str(func)
                  or 'method random_sample of mtrand.RandomState' in str(func)):
                def rand_u(shape):
                    return global_randomstreams.uniform( low=0, high=1, size=shape)
                return rand_u
            else:
                raise ValueError('Unsupported function: {0}'.format(func))

        # ** ======================= Anything else

        else:
            try:
                # transform and recompile the function
                t = TheanoTransformer(watcher=self.watcher)
                new_func = t.recompile(func)
                return new_func
            except:
                raise ValueError('Unsupported function: {0}'.format(func))


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
        if not isvar(var):
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
                axis1, axis2 = (unshadow(a) for a in args)
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

        # Numpy's sort operates inplace, need to explicitly handle that using
        # the inplace_updates dict.
        elif method_name == 'sort':
            def sort(*args, **kwargs):
                sorted_var = var.sort(*args, **kwargs)
                self.watcher.inplace_updates[id(var)] = sorted_var
                return None
            return sort

        # ...Otherwise, try to access the method on the Theano variable
        else:
            return getattr(var, method_name)

    # ** --------------------------------------------------------
    # ** AST Manipulation (Node Visitors)

    def visit_FunctionDef(self, node):
        """
        When a function is defined, shadow each of its arguments immediately.

        The AST is modified so that a function defined as:

            def f(a, b=None, *c, **d):
                ...

        is changed via this method to:

            def f(a, b=None, *c, **d):
                a = a
                b = b
                c = c
                d = d
                ...

        which is eventually transformed by the visitor to:

            def f(a, b=None, *c, **d):
                a = self.shadow(a)
                b = self.shadow(b)
                c = self.shadow(c)
                d = self.shadow(d)
                ...

        This way, any future references to these variables will access their
        shadowed values. This is important because inplace modifications do
        not always force the `shadow` method to get called, and so the inplace
        changes might not be reflected the next (and first!) time the variable
        is loaded.
        """
        body = []
        for param in node.args.args + [node.args.vararg] + [node.args.kwarg]:
            if param:
                body.append(ast_module.Assign(
                    targets=[ast_module.Name(ctx=ast_module.Store(),
                                             id=getattr(param, 'id', param))],
                    value=ast_module.Name(ctx=ast_module.Load(),
                                          id=getattr(param, 'id', param))))
        node.body = body + node.body
        return self.generic_visit(node)

    def visit_Num(self, node):
        # don't make changes because these are typically function arguments
        # return self.ast_wrap('shadow', node)
        return node

    def visit_Name(self, node):
        """
        Whenever a literal variable name is loaded, call the
        'shadow' method on its value.
        """
        self.generic_visit(node)
        if isinstance(node.ctx, ast_module.Load):
            node = self.ast_wrap('shadow', node)
        return node

    def visit_Call(self, node):
        """
        Whenever a function is called, first pass it to
        the 'handle_functions' method.
        """
        self.generic_visit(node)
        node.func = self.ast_wrap('handle_functions', node.func)
        return node

    def visit_Compare(self, node):
        """
        Theano operators must be called as functions.
        This replaces literal operators with the appropriate functions.
        """
        self.generic_visit(node)
        op = node.ops[0]
        if isinstance(op, ast_module.Gt):
            theano_op = 'gt'
        elif isinstance(op, ast_module.GtE):
            theano_op = 'ge'
        elif isinstance(op, ast_module.Lt):
            theano_op = 'lt'
        elif isinstance(op, ast_module.LtE):
            theano_op = 'le'
        elif isinstance(op, ast_module.Eq):
            theano_op = 'eq'
        elif isinstance(op, ast_module.NotEq):
            theano_op = 'neq'
        else:
            # Is, IsNot, In, Not In
            return node

        new_node = _simple_call(args=[node.left] + node.comparators,
                                func=ast_module.Attribute(
                                    attr=theano_op,
                                    ctx=ast_module.Load(),
                                    value=ast_module.Name(ctx=ast_module.Load(),
                                                          id='T')))

        return new_node

    # def visit_AugAssign(self, node):
    #     """
    #     Inplace assigns circumvent shadowing, so we change them into regular assigns.
    #     """
    #     self.generic_visit(node)
    #     new_node = ast_module.copy_location(
    #         ast_module.Assign(
    #             targets=[node.target],
    #             value=ast_module.BinOp(left=ast_module.Name(ctx=ast_module.Load(),
    #                                                         id=node.target.id),
    #                                    right=node.value,
    #                                    op=node.op)),
    #         node)

    #     return new_node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        new_node = ast_module.copy_location(
            _simple_call(args=[node.value,
                               ast_module.Str(s=node.attr),
                               self.ast_wrap('handle_array_methods',
                                             [node.value, ast_module.Str(s=node.attr)])],
                         func=ast_module.Name(ctx=ast_module.Load(),
                                              id='getattr')),
            node)
        return new_node
