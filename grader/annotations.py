import functools

def generate_custom_comparer(equality_fn):
    assert callable(equality_fn), 'equality_fn must be a function'

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.equality_fn = equality_fn
        return wrapper
    return decorator


def generate_class(trials_per_instance, **class_kwargs):
    assert type(trials_per_instance) == int, 'trials_per_instance must be an int'
    for k_name, k_val in class_kwargs.items():
        assert callable(k_val), f'[Debug] class_builder variable {k_name} has to be a function'

    def __gen_class_params__():
        params = {}
        for k_name, k_fn in class_kwargs.items():
            params[k_name] = k_fn()
        return params

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.trials_per_instance = trials_per_instance
        wrapper.class_kwargs = class_kwargs
        wrapper.gen_class_params = __gen_class_params__
        return wrapper
    return decorator


def generate_test_case(trials=2500, **fn_kwargs):
    assert type(trials) == int, 'trials must be an int'
    assert trials > 0, 'trials must be positive'
    for k_name, k_fn in fn_kwargs.items():
        assert callable(k_fn), f'[Debug] test_case_builder variable {k_name} has to be a function'

    def __gen_fn_params__():
        params = {}
        for k_name, k_fn in fn_kwargs.items():
            params[k_name] = k_fn()
        return params

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.max_trials = trials
        wrapper.fn_kwargs = fn_kwargs
        wrapper.gen_fn_params = __gen_fn_params__
        return wrapper
    return decorator


def no_test_cases():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.no_test_cases = True
        return wrapper
    return decorator