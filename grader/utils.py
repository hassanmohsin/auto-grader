import numpy as np
import inspect

# Functions that don't require @no_test_cases to be ignored
IGNORED_FUNCTIONS = {'__init__', 'draw'}


class Colors:
    # Terminal Colors (https://chrisyeh96.github.io/2020/03/28/terminal-colors.html)
    T_RED = '\033[91m'
    T_GREEN = '\033[92m'
    T_YELLOW = '\033[93m'
    T_BLUE = '\033[94m'
    T_MAGENTA = '\033[95m'
    T_CYAN = '\033[96m'

    T_RESET = '\033[0m'
    T_LIGHT_GREEN = '\033[92;1m'
    T_DARK_GREEN = '\033[92;2m'
    T_DARK_RED = '\033[91;2m'
    T_DARK_YELLOW = '\033[93;2m'


def compare_solutions(real_output, student_output):
    """Compares the given outputs and determines if they are equal."""
    # If we expect a 1D array and the student passes a 1D list, convert the student's list to a 1D array
    expect_1d_array = type(real_output) is np.ndarray and len(real_output.shape) == 1
    student_has_1d_list = type(student_output) is list and all([not isinstance(item, (list, np.ndarray)) for item in student_output])
    if expect_1d_array and student_has_1d_list:
        student_output = np.array(student_output, dtype=real_output.dtype)

    # Check that the types match
    if type(real_output) != type(student_output):
        # raise StudentCodeException(f'Real solution for problem has type {type(real_solution)} but student solution has type {type(student_solution)}\n')
        return False

    # Compare the two solutions
    if type(real_output) is np.ndarray:
        return np.array_equal(real_output, student_output)
    elif type(real_output) is bool:
        return real_output == student_output
    elif type(real_output) is int or type(real_output) is float or type(real_output) is np.int32 or type(real_output) is np.float32:
        return abs(real_output - student_output) < 0.001
    elif type(real_output) is list:
        return np.array_equal(real_output, student_output)
    elif type(real_output) is tuple:
        L = []
        for real_item, student_item in zip(real_output, student_output):
            L.append(compare_solutions(real_item, student_item))
        return all(L)
    elif type(real_output) is dict or type(real_output) is set:
        return real_output == student_output
    elif real_output is None:
        return student_output is None
    else:
        raise Exception(f'[Debug] Cannot grade type {type(real_output)}')


def get_module_functions(module):
    """Gets all the functions in the module and all class constructors.

    Ignores class constructors and functions named draw() without requiring annotation.
    """
    # First get all module functions that are not inside a class
    fns = dict(inspect.getmembers(module, inspect.isfunction))

    # Get the classes if any and extract their functions
    classes = inspect.getmembers(module, inspect.isclass)
    constructors = {}

    # print(f"[Debug] Inspected classes = {classes}")
    for class_name, class_obj in classes:
        for orig_fn_name, fn in inspect.getmembers(class_obj, inspect.isfunction):
            # Combine class name with function name to use as key for dictionary
            fn_name = f'{class_name}.{orig_fn_name}()'
            if orig_fn_name in IGNORED_FUNCTIONS:
                print(f'[Debug] Not grading {fn_name} as they are in the ignore set')
                continue
            assert fn_name not in fns.keys(), f'Two functions have the same name, {fn_name}'
            fns[fn_name] = fn
            constructors[fn_name] = class_obj

    return fns, constructors
