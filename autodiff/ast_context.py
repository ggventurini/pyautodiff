import logging
import meta
import ast as ast_module
import numpy as np
import theano
import theano.tensor as T


logger = logging.getLogger('pyautodiff')


# XXX FIXME This will not do - seed must be exposed.
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
global_randomstreams = RandomStreams(seed=123)


def istensor(x):
    tensortypes = (theano.tensor.TensorConstant,
                   theano.tensor.TensorVariable)
    return isinstance(x, tensortypes)


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


def unshadow(x):
    if isvar(x):
        try:
            return x.eval()
        except:
            return x
    else:
        return x


class TheanoTransformer(ast_module.NodeTransformer):

    def __init__(self):
        super(TheanoTransformer, self).__init__()
        self.smap = dict()

    def ast_wrap(self, node, method_name):
        wrapped = ast_module.Call(
            args=[node],
            func=ast_module.Attribute(
                attr=method_name,
                ctx=ast_module.Load(),
                value=ast_module.Name(
                    ctx=ast_module.Load(), id='__TT')),
            keywords=[],
            kwargs=None,
            starargs=None)
        return wrapped

    def getvar(self, var):
        return self.smap.get(id(var), var)

    def shadow(self, x):
        """
        Given a numerical variable x, return an equivalent Theano shared variable
        and store the relationship in self.smap. Otherwise return x.
        """
        print 'in shadow'
        print x, id(x)
        if not isinstance(x, (int, float, np.ndarray)):
            return x

        # take special care with small ints, because CPYthon caches them.
        # This makes it impossible to tell one from the other.
        if isinstance(x, int) and -5 <= x <= 256:
            print 'in int'
            print x, id(x)
            x2 = np.asarray(x)
            x = x2
            print x, id(x)
            print 'done with int'

        elif isinstance(x, float):
            x = np.float_(x)

        if getattr(x, 'dtype', None) == bool:
            logger.info('Warning: Theano has no bool type; upgrading to int8.')
            x = x.astype('int8')

        sym_x = theano.shared(x)
        return self.smap.setdefault(id(x), sym_x)


    def handle_functions(self, func):
        """
        Given some function for, return another function.

        Generally used to exchange NumPy functions for Theano equivalents.
        """

        # if the function has a _theano_fn attribute, return that fn
        if hasattr(func, '_theano_fn'):
            func = func._theano_fn

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

        # handle range/xrange
        elif func.__name__ in ('range', 'xrange'):
            return lambda *args : func(*(unshadow(a) for a in args))

        # handle numpy functions
        elif ((getattr(func, '__module__', None)
               and func.__module__.startswith('numpy'))
               or isinstance(func, np.ufunc)):

            # if func.__name__ == 'reshape':
                # return func
                # def f(x, shape):
                    # print shape
                    # return T.reshape(x, shape)#[unshadow(s) for s in shape])
                # return f

            # else:
                # get the theano version
            func = getattr(T, func.__name__, func)

        # handle random numbers
        elif ('method random of mtrand.RandomState' in str(func)
              or 'method random_sample of mtrand.RandomState' in str(func)):
            def rand_u(shape):
                return global_randomstreams.uniform( low=0, high=1, size=shape)
            return rand_u

        return func

    def visit_Num(self, node):
        # return self.ast_wrap(node, 'shadow')
        # don't make changes because these are typically function arguments
        return node

    def visit_Name(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast_module.Load):
            node = self.ast_wrap(node, 'shadow')
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        node.func = self.ast_wrap(node.func, 'handle_functions')
        return node

    def visit_Compare(self, node):
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

        new_node = ast_module.Call(args=[node.left] + node.comparators,
                                   func=ast_module.Attribute(
                                       attr=theano_op,
                                       ctx=ast_module.Load(),
                                       value=ast_module.Name(
                                           ctx=ast_module.Load(), id='T')),
                                   keywords=[],
                                   kwargs=None,
                                   starargs=None)

        return new_node

    def transform(self, f):
        self.smap.clear()
        ast = self.visit(get_ast(f))
        ast = ast_module.fix_missing_locations(ast)
        new_globals = globals()
        new_globals.update({'__TT' : self})
        new_f = meta.decompiler.compile_func(
            ast, '<TheanoTransformer-AST>', new_globals)
        return new_f

