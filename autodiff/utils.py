import gc
import opcode
import inspect
import numpy as np

from autodiff.compat import OrderedDict, getcallargs

#import theano
#from theano.sandbox.cuda import cuda_ndarray
#cuda_ndarray = cuda_ndarray.cuda_ndarray


def _int(i):
    """
    CPython caches small ints (-5 <= i <= 256), so they share ids and can not
    be differentiated. NumPy int classes do not get cached, so two variables
    with the same _int value will have different ids.
    """
    return np.int_(i)


def orderedcallargs(fn, *args, **kwargs):
    """
    Returns an OrderedDictionary containing the names and values of a
    function's arguments. The arguments are ordered according to the function's
    argspec:
        1. named arguments
        2. variable positional argument
        3. variable keyword argument
    """
    callargs = getcallargs(fn, *args, **kwargs)
    argspec = inspect.getargspec(fn)

    o_callargs = OrderedDict()
    for argname in argspec.args:
        o_callargs[argname] = callargs[argname]

    if argspec.varargs:
        o_callargs[argspec.varargs] = callargs[argspec.varargs]

    if argspec.keywords:
        o_callargs[argspec.keywords] = callargs[argspec.keywords]

    return o_callargs


def as_seq(x, seq_type=None):
    """
    If x is not a sequence, returns it as one. The seq_type argument allows the
    output type to be specified (defaults to list). If x is a sequence and
    seq_type is provided, then x is converted to seq_type.

    Arguments
    ---------
    x : seq or object

    seq_type : output sequence type
        If None, then if x is already a sequence, no change is made. If x
        is not a sequence, a list is returned.
    """
    if x is None:
        x = []
    elif not isinstance(x, (list, tuple, set, frozenset)):
        x = [x]
    if seq_type is not None:
        x = seq_type(x)
    return x


def itercode(code):
    """Return a generator of byte-offset, opcode, and argument
    from a byte-code-string
    """
    i = 0
    extended_arg = 0
    n = len(code)
    while i < n:
        c = code[i]
        num = i
        op = ord(c)
        i = i + 1
        oparg = None
        if op >= opcode.HAVE_ARGUMENT:
            oparg = ord(code[i]) + ord(code[i + 1]) * 256 + extended_arg
            extended_arg = 0
            i = i + 2
            if op == opcode.EXTENDED_ARG:
                extended_arg = oparg * 65536L

        delta = yield num, op, oparg
        if delta is not None:
            abs_rel, dst = delta
            assert abs_rel == 'abs' or abs_rel == 'rel'
            i = dst if abs_rel == 'abs' else i + dst


def flat_from_doc(doc):
    """Iterate over the elements of a nested document in a consistent order,
    unpacking dictionaries, lists, and tuples.

    Returns a list.
    """
    rval = []
    if type(doc) in (list, tuple):
        for d_i in doc:
            rval.extend(flat_from_doc(d_i))
    elif type(doc) == dict:
        for k, v in doc.items():
            if not isinstance(k, basestring):
                # -- if keys are tuples containing ndarrays, should
                #    they be traversed also?  What about number keys
                #    where to draw line?
                raise NotImplementedError(
                    'potential ambiguity in non-string keys', k)
            rval.extend(flat_from_doc(v))
    else:
        rval.append(doc)
    return rval


def doc_from_flat(doc, flat, pos):
    """Iterate over a nested document, building a clone from the elements of
    flat

    Returns object with same type as doc, and the position of the next unused
    element in flat.
    """
    if type(doc) in (list, tuple):
        rval = []
        for d_i in doc:
            d_i_clone, pos = doc_from_flat(d_i, flat, pos)
            rval.append(d_i_clone)
        rval = type(doc)(rval)

    elif type(doc) == dict:
        rval = {}
        for k, v in doc.items():
            v_clone, pos = doc_from_flat(v, flat, pos)
            rval[k] = v_clone

    else:
        rval = flat[pos]
        pos += 1

    return rval, pos


# -- picklable decorated function
class post_collect(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **kwargs)
        finally:
            gc.collect()
            #mem_info = cuda_ndarray.mem_info()
            #om = cuda_ndarray.outstanding_mallocs()
            #print 'Post-gc: %s %s' % (mem_info, om)
