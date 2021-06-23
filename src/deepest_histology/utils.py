import inspect
import logging
from functools import wraps


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
            logging.getLogger(func.__module__).info(f'using default value {param}')

        # call wrapped function
        return func(*args, **kwargs)

    return default_logged