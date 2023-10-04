from functools import wraps, update_wrapper


## I want to rename this to avoid confusion with OpenMP offload
## but I can't think of a better name for now
class OffloadDecorator:
    """Stateful decorator for functions with both Python and CPP implementations.
    Allows for quick switching of function call with a cleaner interface for
    replacing prototype implementations with compiled code.
    """

    def __init__(self, func, offload_func, use_offload):
        update_wrapper(self, func)
        self.func = func
        self.offload_func = offload_func
        self.use_offload = use_offload

    def __call__(self, *args, **kwargs):
        if self.use_offload:
            return self.offload_func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

    @property
    def use_offload(self):
        return self._use_offload

    @use_offload.setter
    def use_offload(self, val: bool):
        if isinstance(val, bool):
            self._use_offload = val
        else:
            raise TypeError("Value must be True or False.")


def offload(offload_func, use_offload=True):
    """Decorator factory for compiled versions of function calls."""

    def _offload(func):
        return OffloadDecorator(func, offload_func, use_offload)

    return _offload


def return_tuple(func):
    """Decorator for wrapping return types of indexing util calls."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs).t

    return wrapper
