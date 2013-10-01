import logging
import meta
import ast

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
    if isinstance(func_def, ast.Lambda):
        func_def = ast.FunctionDef(name='<lambda>', args=func_def.args,
                                   body=[ast.Return(func_def.body)],
                                   decorator_list=[])
    assert isinstance(func_def, ast.FunctionDef)
    return func_def

def print_ast(ast):
    meta.asttools.print_ast(ast)

class TheanoTransformer(ast.NodeTransformer):
    def __init__(self):
        super(TheanoTransformer, self).__init__()
        self.smap = dict()

    def ast_wrap_node(self, node, method):
        wrapped = ast.Call(args=[node],
                           func=ast.Attribute(attr=method,
                                              ctx=ast.Load(),
                                              value=ast.Name(ctx=ast.Load(),
                                                             id='self')),
                           keywords=[],
                           kwargs=None,
                           starargs=None)
        return wrapped

    def ensure_shadow(self, x):
        if isvar(x):
            return x
        else:
            return self.shadow(x)

    def shadow(self, x):
        # take special care with small ints, because CPYthon caches them.
        # This makes it impossible to tell one from the other.
        if isinstance(x, int) and -5 <= x <= 256:
            x = np.int_(x)

        if getattr(x, 'dtype', None) == bool:
            logger.info('Warning: Theano has no bool type; upgrading to int8.')
            x = x.astype('int8')

        sym_x = theano.shared(x)

        return self.smap.setdefault(id(x), sym_x)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            node = ast.copy_location(self.ast_wrap_node(node, 'ensure_shadow'), node)
        return node

    def test_run(self, f):
        a = get_ast(f)
        self.visit(a)
        a = ast.fix_missing_locations(a)
        new_globals = globals()
        new_globals.update({'self' : self})
        new_f = meta.decompiler.compile_func(a, '<TheanoTransformer>', new_globals)
        return new_f
