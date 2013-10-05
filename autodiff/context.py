import logging
import meta
import ast as ast_module
import numpy as np
import theano
import theano.tensor as T


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
        self.s_vars = dict() # symbolic map
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
                                                            id='__Context')),
            args=args)

        return wrapped

    def transform(self, f):
        f_ast = get_ast(f)
        transformed_ast = self.visit(f_ast)
        return ast_module.fix_missing_locations(transformed_ast)

    def recompile(self, f):
        ast = self.transform(f)
        new_globals = f.func_globals.copy()
        new_globals.update({'__Context' : self})
        return compile_func(ast, new_globals, '<Context-AST>')


class TheanoTransformer(ASTTransformer):

    def __init__(self, watcher):
        super(TheanoTransformer, self).__init__()
        self.watcher = watcher

    # ** --------------------------------------------------------
    # ** Direct Manipulation (Methods)

    def shadow(self, x):
        """
        Given a numerical variable x, return an equivalent Theano shared variable
        and store the relationship in self.s_vars. Otherwise return x.
        """
        if id(x) in self.watcher._noshadow:
            return x

        if not isinstance(x, (int, float, np.ndarray)):
            return x

        # take special care with small ints, because CPython caches them.
        if isinstance(x, int) and -5 <= x <= 256:
            x = np.int_(x)

        elif isinstance(x, float):
            x = np.float_(x)

        if getattr(x, 'dtype', None) == bool:
            logger.info('Warning: Theano has no bool type; upgrading to int8.')
            x = x.astype('int8')

        if id(x) in self.watcher.s_vars:
            return self.watcher.s_vars[id(x)]
        else:
            self.watcher._nogc.append(x)
            sym_x = theano.shared(x)
            self.watcher.s_vars[id(x)] = theano.shared(x)
            return sym_x

    def handle_functions(self, func):
        """
        Given some function for, return another function.

        Generally used to exchange NumPy functions for Theano equivalents.
        """

        # ** ------------------------
        # if the function has a _theano_fn attribute, return that fn
        if hasattr(func, '_theano_fn'):
            func = func._theano_fn

        # ** ------------------------
        # handle casting functions
        elif func.__name__ in ['bool', 'bool_', 'bool8']:
            logger.info('Warning: Theano has no bool type; upgrading to int8.')
            return lambda x : T.neq(x, 0)
        elif func.__name__ in T.basic._cast_mapping.keys():
            return lambda x : T.cast(x, dtype=func.__name__)
        elif func.__name__ == 'float':
            return lambda x : T.cast(x, dtype=theano.config.floatX)
        elif func.__name__ == 'int':
            dtype = 'int' + theano.config.floatX[-2:]
            return lambda x : T.cast(x, dtype=dtype)

        # ** ------------------------
        # handle range/xrange
        elif func.__name__ in ('range', 'xrange'):
            return lambda *args : func(*(unshadow(a) for a in args))

        # ** ------------------------
        # handle numpy functions
        elif ((getattr(func, '__module__', None)
               and func.__module__.startswith('numpy'))
               or isinstance(func, np.ufunc)):

            # else:
                # get the theano version
            func = getattr(T, func.__name__, func)

        # ** ------------------------
        # handle array methods that suddenly have tensor instances
        elif isvar(getattr(func, '__self__', None)):
            return self.handle_array_methods(func.__self__, func.__name__)

        # ** ------------------------
        # handle random numbers
        elif ('method random of mtrand.RandomState' in str(func)
              or 'method random_sample of mtrand.RandomState' in str(func)):
            def rand_u(shape):
                return global_randomstreams.uniform( low=0, high=1, size=shape)
            return rand_u

        return func

    def handle_array_methods(self, var, method_name):
        if not isvar(var):
            return getattr(var, method_name)

        if method_name == 'reshape':
            def reshape(*args, **kwargs):
                if not isinstance(args[0], (list, tuple)):
                    args = [args]
                return var.reshape(*args, **kwargs)
            return reshape
        elif method_name == 'swapaxes':
            def swapaxes(*args, **kwargs):
                axis1, axis2 = (unshadow(a) for a in args)
                dims = range(var.ndim)
                dims[axis1], dims[axis2] = dims[axis2], dims[axis1]
                return var.dimshuffle(*dims)
            return swapaxes
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
                        raise NotImplementedError('Unsupported dtype: {0}'.format(dtype))
                if dtype == 'bool':
                    dtype == 'int8'
                    logger.info('Warning: Theano has no bool type; upgrading to int8.')

                return var.astype(dtype)
            return astype
        else:
            return getattr(var, method_name)

    # ** --------------------------------------------------------
    # ** AST Manipulation (Node Visitors)


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

    def visit_AugAssign(self, node):
        """
        Inplace assigns circumvent shadowing, so we change them into regular assigns.
        """
        self.generic_visit(node)
        new_node = ast_module.copy_location(
            ast_module.Assign(
                targets=[node.target],
                value=ast_module.BinOp(left=ast_module.Name(ctx=ast_module.Load(),
                                                            id=node.target.id),
                                       right=node.value,
                                       op=node.op)),
            node)

        return new_node

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