import theano
import theano.tensor as tt
from autodiff.context import Context


class Gradient(object):
    def __init__(self, fn):
        self._fn = fn
        self._grad_fn = None

    @property
    def fn(self):
        return self._fn

    @property
    def grad_fn(self):
        return self._grad_fn

    def __call__(self, *args, **kwargs):
        # TODO: convert small ints to arrays to allow tracking, but
        # watch out for functions that require int arguments
        if self.grad_fn is None:
            ctxt = Context()
            result = ctxt.call(self.fn, args, kwargs)

            try:
                s_args = [ctxt.svars[id(a)] for a in args]
                s_kwargs = [ctxt.svars[id(v)] for v in kwargs.values()]
                s_result = ctxt.svars[id(result)]
            except KeyError:
                print 'ERROR: PyAD was unable to trace the requested variable.'
                raise
            except:
                raise
            grad = tt.grad(s_result, s_args + s_kwargs)
            if len(grad) == 1:
                grad = grad[0]

            self._grad_fn = theano.function([s_args + s_kwargs], grad)
            self._sym_grad = grad

        all_args = args + tuple(kwargs.values())
        try:
            grad_result = self.grad_fn(*all_args)
        except:
            self._grad_fn = None
            raise
        return grad_result
