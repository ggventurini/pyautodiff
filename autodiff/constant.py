import numpy as np

# _constants is a set of object ids that should be treated as constants and not
# shadowed with symbolic variables
_constants = set()


def clear_constants():
    _constants.clear()


def is_constant(val):
    return id(val) in _constants


def Constant(val):
    """
    Returns a new and unique variable to be used as a constant. Also adds the
    id of that variable to the _constants set, so that Context objects can tell
    if a variable should be treated as constant, or not.  Constants are not
    shadowed by symbolic variables, and provide a way to safely use functions
    like range or iterators.

    This should not work:

        @function
        def fn1(x):
            a = 3
            for i in range(a):
                ...

        @function
        def fn2(x, a):
            x.sum(axis=a)
            ...

    But this should:

        @function
        def fn1(x):
            a = 3
            for i in range(Constant(a)):
                ...

        @function
        def fn2(x, a):
            x.sum(axis=Constant(a))
            ...

    Note that this *does* work:
        @function
        def fn1(x):
            for i in range(3):
                ...
    Because the '3' is loaded as a constant int, which is not shadowed by
    default.
    """
    if isinstance(val, int):
        new_val = np.int_(val)
    elif isinstance(val, float):
        new_val = np.float_(val)
    elif isinstance(val, bool):
        new_val = np.bool_(val)
    else:
        new_val = np.array(val)
    _constants.add(id(new_val))
    return new_val
