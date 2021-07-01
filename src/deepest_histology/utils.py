import inspect
import logging
from typing import Callable, Any
from functools import wraps, cached_property


def log_defaults(func):
    """Decorator which logs used default values of parameters of a function."""
    @wraps(func)
    def default_logged(*args, **kwargs):
        # find unset kwargs with default values
        params = inspect.signature(func).parameters
        remaining_keys = list(params)[len(args):]
        params_with_defaults = [param
                                for k in set(remaining_keys) - set(kwargs)
                                if (param := params[k]).default != inspect.Parameter.empty]
        # log them
        for param in params_with_defaults:
            logging.getLogger(func.__module__).debug(f'using default value {param}')

        # call wrapped function
        return func(*args, **kwargs)

    return default_logged


class Lazy:
    """A wrapper which constructs the underlying object only when it is needed."""
    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory

    @cached_property
    def _val(self):
        return self._factory()

    def __getattr__(self, k):
        return getattr(self._val, k)

    def __setattr__(self, k, v):
        if k == '_factory':
            super().__setattr__(k, v)
        else:
            setattr(self._val, k, v)

    def __getitem__(self, k):
        return self._val[k]

    def __setitem__(self, k, v):
        self._val[k] = v