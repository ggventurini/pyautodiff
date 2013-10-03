import logging
import meta
import ast as ast_module
import numpy as np
import theano
import theano.tensor as T


logger = logging.getLogger('pyautodiff')


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
    meta.asttools.print_ast(ast)


def print_source(ast):
    if hasattr(ast, 'func_code'):
        ast = get_ast(ast)
    meta.asttools.python_source(ast)


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
                    ctx=ast_module.Load(), id='TT')),
            keywords=[],
            kwargs=None,
            starargs=None)
        return wrapped

    def getvar(self, var):
        return self.smap.get(id(var), var)

    def shadow(self, x):
        if not isinstance(x, (int, float, np.ndarray)):
            return x

        # take special care with small ints, because CPYthon caches them.
        # This makes it impossible to tell one from the other.
        if isinstance(x, int) and -5 <= x <= 256:
            x = np.int_(x)
        elif isinstance(x, float):
            x = np.float_(x)

        if getattr(x, 'dtype', None) == bool:
            logger.info('Warning: Theano has no bool type; upgrading to int8.')
            x = x.astype('int8')

        sym_x = theano.shared(x)

        return self.smap.setdefault(id(x), sym_x)


    def handle_functions(self, func):
        # if the function has a _theano_fn attribute, return that fn
        if hasattr(func, '_theano_fn'):
            return func._theano_fn

        # if it is a numpy function, try to get the theano version
        elif ((getattr(func, '__module__', None)
            and func.__module__.startswith('numpy'))
            or isinstance(func, np.ufunc)):
                return getattr(T, func.__name__, func)

        else:
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

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node = self.ast_wrap(node, 'handle_functions')
        return node

    def transform(f):
        ast = self.visit(get_ast(f))
        ast = ast_module.fix_missing_locations(ast)
        new_globals = globals()
        new_globals.update({'TT' : self})
        new_f = meta.decompiler.compile_func(
            ast, '<TheanoTransformer-AST>', new_globals)
        return new_f


def f1(x):
    z = np.dot(x.sum(), 4)
    return z + np.ones((2,4))

t = TheanoTransformer()
f2 = t.test_run(f1)

a = np.ones((3,4))
f2(a)

