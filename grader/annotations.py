import functools

def generate_custom_comparer(equality_fn):
    def decorator(func):
        assert callable(equality_fn), f'[Debug] Error while annotating function "{func.__name__}" [{equality_fn} must be a function]'
        # TODO: Check that the parameter count matches the one used in grader
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.equality_fn = equality_fn
        return wrapper
    return decorator


def generate_class(trials_per_instance, **class_kwargs):

    def __gen_class_params__():
        params = {}
        for k_name, k_fn in class_kwargs.items():
            params[k_name] = k_fn()
        return params

    def decorator(func):
        assert type(trials_per_instance) == int, f'[Debug] Error while annotating class function "{func.__name__}" [{trials_per_instance} must be an int]'
        for k_name, k_val in class_kwargs.items():
            assert callable(k_val), f'[Debug] Error while annotating class function "{func.__name__}" ["{k_name}" has to be a function, did you forget to include lambda?]'
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.trials_per_instance = trials_per_instance
        wrapper.class_kwargs = class_kwargs
        wrapper.gen_class_params = __gen_class_params__
        return wrapper
    return decorator


def generate_test_case(trials=2500, **fn_kwargs):

    def __gen_fn_params__():
        params = {}
        for k_name, k_fn in fn_kwargs.items():
            params[k_name] = k_fn()
        return params

    def decorator(func):
        assert type(trials) == int, f'[Debug] Error while annotating function "{func.__name__}" [{trials} must be an int]'
        assert trials > 0, f'[Debug] Error while annotating function "{func.__name__}" [{trials} must be greater than 0]'
        for k_name, k_fn in fn_kwargs.items():
            assert callable(k_fn), f'[Debug] Error while annotating function "{func.__name__}" ["{k_name}" has to be a function, did you forget to include lambda?]'
        
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