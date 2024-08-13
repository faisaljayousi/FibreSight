import functools


def log_method(func):
    """Decorator to log method calls."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        instance = args[0]
        instance.logger.info(f"Calling method: {func.__name__}")
        result = func(*args, **kwargs)
        instance.logger.info(f"Method {func.__name__} completed")
        return result

    return wrapper
