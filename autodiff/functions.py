def tag(obj, tag):
    """
    Tags an object with a certain keyword. By default, all symbolic objects are
    associated with the id of the Python object they shadow in a Context's
    svars dict.  By calling this function on an object and providing a
    (hashable) tag, users can more easily access the symbolic representation of
    any objects that might only be created during function execution.

    Example:

        @function
        def fn(x):
            y = tag(x + 2, 'y')
            z = y * 3
            return z

        fn.s_vars['y'] # returns the symbolic version of y

    """
    return obj

def escape(obj):
    return obj

def escaped_call(fn, *args, **kwargs):
    return f(*args, **kwargs)
