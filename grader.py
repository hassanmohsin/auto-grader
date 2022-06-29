# Author: Jose G. Perez
# Requires a Linux machine for the timeout functionality
# TODO: Requires you to clean-up Colab Notebooks by manually removing colab-exclusive imports and getting rid of duplicate class definitions/functions
# Can only grade standalone functions and class functions
# Cannot grade class initializers (__init__ functions)
# TA's, if you want students to see their mistakes more easily implement the __str__() function for your classes
# TODO: Only parse classes/functions while ignoring loose code
import sys
import argparse
import pathlib
import inspect
import subprocess
import traceback
import copy
import errno
import os
import signal
import functools

import pandas as pd
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer

from threading import Thread
import functools

import wrapt_timeout_decorator

TIMEOUT_S = 0.1  # How many seconds should we allow student functions to run before terminating them


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


def generate_test_case(trials=5000, **fn_kwargs):
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

class StudentCodeException(Exception):
    pass

class StudentTimeoutException(StudentCodeException):
    pass

def compare_solutions(real_solution, student_solution):
    # If we expect a 1D array and the student passes a 1D list, convert the student's list to a 1D array
    expect_1d_array = type(real_solution) is np.ndarray and len(real_solution.shape) == 1
    student_has_1d_list = type(student_solution) is list and all([not isinstance(item, (list, np.ndarray)) for item in student_solution])
    if expect_1d_array and student_has_1d_list:
        student_solution = np.array(student_solution, dtype=real_solution.dtype)

    # Check that the types match
    if type(real_solution) != type(student_solution):
        raise StudentCodeException(f'Real solution for problem has type {type(real_solution)} but student solution has type {type(student_solution)}\n')

    # Compare the two solutions
    if type(real_solution) is np.ndarray:
        return np.array_equal(real_solution, student_solution)
    elif type(real_solution) is bool:
        return real_solution == student_solution
    elif type(real_solution) is int or type(real_solution) is float or type(real_solution) is np.int32 or type(real_solution) is np.float32:
        return abs(real_solution - student_solution) < 0.001
    elif type(real_solution) is list:
        return np.array_equal(real_solution, student_solution)
    elif type(real_solution) is tuple:
        L = []
        for real_item, student_item in zip(real_solution, student_solution):
            L.append(compare_solutions(real_item, student_item))
        return all(L)
    elif type(real_solution) is dict or type(real_solution) is set:
        return real_solution == student_solution
    else:
        raise Exception(f'[Debug] Cannot grade type {type(real_solution)} for problem {funct_name}')


def __try_student_fn__(student_fn, *args, **kwargs):
    try:
        return wrapt_timeout_decorator.timeout(TIMEOUT_S, use_signals=True, timeout_exception=StudentTimeoutException)(student_fn)(*args, **kwargs)
    except StudentTimeoutException as ex:
        return ex
    except OSError as ex:
        return ex
    except Exception as ex:
        return ex


def __create_class_instance__(constructor_fn, class_init_params):
    if class_init_params:
        assert type(class_init_params) == dict, 'class_init_params must be kwargs dictionary'
        return constructor_fn(**class_init_params)
    else:
        return constructor_fn()


if __name__ == '__main__':
    # Parse arguments from the command-line
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=str, required=True)
    parser.add_argument("--student_dir", type=str, required=True)
    parser.add_argument("--max_grade", type=int, required=True)
    parser.add_argument("--students", nargs='+', required=False)
    args = parser.parse_args()

    # Import the solution file
    solution_file = pathlib.Path(args.solution)
    if not solution_file.exists():
        raise Exception(f'solution file {solution_file} does not exist')

    # Save the current working directory
    original_cwd = os.getcwd()
    # Add the directory where the solution is to the path so we can easily import it
    sys.path.append(str(solution_file.parent.resolve()))
    # Change the working directory to the solution folder so it can open files
    os.chdir(solution_file.parent)
    # Import the solution file
    module = __import__(solution_file.stem)
    # Change the working directory back to its original state
    os.chdir(original_cwd)

    # Get the functions of the solution
    solution_functs = inspect.getmembers(module, inspect.isfunction)

    # Get the classes if any and extract their functions
    classes = inspect.getmembers(module, inspect.isclass)
    classes_function_dict = {}
    print(f"[Debug] Inspected classes = {classes}")
    for class_name, class_obj in classes:
        for funct_name, funct in inspect.getmembers(class_obj, inspect.isfunction):
            solution_functs.append((funct_name, funct))
            classes_function_dict[funct_name] = class_obj

    # Find the functions we do not want to grade
    to_remove = []
    for funct_name, funct in solution_functs:
        # Check for @grader.no_test_cases annotation and skip them
        if hasattr(funct, 'no_test_cases'):
            to_remove.append((funct_name, funct))
        else:
            assert hasattr(funct, 'gen_fn_params'), f'[Debug] Not grading {funct_name} due to lack of annotation. Did you forget to annotate it?'

    # Remove functions we do not want to grade
    for t in to_remove:
        solution_functs.remove(t)
        print(f'[Debug] Not grading {t}')

    # Create a dataframe to summarize the scores
    print(f"[Debug] Grading {len(solution_functs)} functions")
    df = pd.DataFrame(columns=['student'] + [fname for (fname, _) in solution_functs])

    # Check the integrity of the student code directory
    student_dir = pathlib.Path(args.student_dir)
    if not student_dir.exists():
        raise Exception(f'student_dir {student_dir} does not exist')

    # Add student directory to the path so we can import student modules more easily later
    sys.path.append(str(student_dir.resolve()))

    # Convert Jupyter notebooks into regular code files
    for fpath in student_dir.glob("*.ipynb"):
        # If the file has already been converted skip it (don't convert it again!)
        if fpath.with_suffix(".py").exists():
            continue

        print(f"[Auto-Grader] Converting jupyter notebook {fpath.stem} into regular code")
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script', str(fpath)])

        # Rename from .txt to .py
        output_fpath = fpath.with_suffix(".txt")
        if not output_fpath.exists():
            raise Exception(f'Converted Jupyter file {output_fpath} could not be found')
        output_fpath.rename(output_fpath.with_suffix('.py'))

    # Override some library functions
    # Specifically, do not open matplotlib plots and override colab's module with our own
    # import matplotlib.pyplot
    # sys.modules['matplotlib.pyplot'].show = lambda: None
    # sys.modules['google.colab'] = grader_colab

    # Go through all the student's code files.
    # !!!!THIS ONLY WORKS for files downloaded with the Blackboard bulk function!!!!
    student_files = list(student_dir.glob("*.py"))

    # If the student parameter is passed, grade only the last attempt of each of these students
    if args.students:
        temp_student_files = []
        for student_name in args.students:
            # Look for all attempts
            attempts = sorted([att_path for att_path in student_files if att_path.stem.split('_')[1] == student_name])
            # Get the final attempt and use that for grading
            temp_student_files.append(attempts[-1])

        student_files = temp_student_files

    for fpath in student_files:
        split_filename = fpath.stem.split('_')
        student_name = split_filename[1]

        # Look for all the attempts and sort them based on the date
        if len(split_filename) > 3:
            date = split_filename[3]
            other_attempt_dates = sorted([att_path.stem.split('_')[3] for att_path in student_files if att_path.stem.split('_')[1] == student_name])
            attempt_number = other_attempt_dates.index(date)+1
        else:
            other_attempt_dates = []
            attempt_number = 1
        print(f'[Auto-Grader] Grading {student_name} attempt {attempt_number} / {len(other_attempt_dates)}')

        with open(student_dir / f'{student_name}_attempt{attempt_number}.bbtxt', 'w') as feedback:
            # Silence output
            orig_stdout = sys.stdout
            sys.stdout = None

            # Import and replace the current source
            import_exception = None
            try:
                student_module = __import__(fpath.stem)
            except Exception as ex:
                feedback.write(f'[AutoGrader] Got exception {repr(ex)} when importing file, unable to grade | {sys.exc_info()[2]}\n')
                feedback.write(traceback.format_exc())
                import_exception = ex
            finally:
                # Recover output
                sys.stdout = orig_stdout

            if import_exception:
                print(f"\t Student received 0% as we were unable to import the file")
                continue

            # Create dictionary of student classes and student class functions
            classes = inspect.getmembers(student_module, inspect.isclass)
            student_classes_function_dict = {}
            for class_name, class_obj in classes:
                for funct_name, funct in inspect.getmembers(class_obj, inspect.isfunction):
                    student_classes_function_dict[funct_name] = class_obj

            # Go through all the functions in the solution
            scores = {'student': f'{student_name}_attempt_{attempt_number}'}
            with tqdm(total=len(solution_functs)) as p_bar:
                for funct_name, funct in solution_functs:
                    # print(f'[Debug] Grading {funct_name}')
                    feedback.write(f'***** [AutoGrader] Grading {funct_name}\n')
                    p_bar.update()
                    p_bar.desc = funct_name

                    # Check if the student actually has the function
                    try:
                        if funct_name in classes_function_dict:
                            student_fn = getattr(student_classes_function_dict[funct_name], funct_name)
                        else:
                            student_fn = getattr(student_module, funct_name)
                    except AttributeError:
                        feedback.write(f'Did not find {funct_name} on the student code, assigning a grade of 0 to this problem\n')
                        scores[funct_name] = 0
                        continue

                    # Grade the question by comparing all test cases
                    is_class_fn = hasattr(funct, 'gen_class_params')
                    has_custom_equality_fn = hasattr(funct, 'equality_fn')

                    log_freq = funct.max_trials // 3
                    log_next_failed_case = True

                    p_bar.total += funct.max_trials
                    results = []
                    for trial_idx in range(funct.max_trials):
                        p_bar.update()
                        p_bar.postfix = f'Test Case {trial_idx}/{funct.max_trials}'

                        if trial_idx % log_freq == 0:
                            log_next_failed_case = True

                        # Generate function parameters
                        real_test_case = funct.gen_fn_params()
                        student_test_case = real_test_case

                        # Create a new class instance every X iterations of the function we are testing when testing class functions
                        if is_class_fn and trial_idx % funct.trials_per_instance == 0:
                            student_test_case = copy.deepcopy(real_test_case)

                            # IMPORTANT! Both classes must be initialized with the same parameters
                            # class_init_params = funct.meta['class_init_param_fn']()
                            class_init_params = funct.gen_class_params()
                            solution_instance = __create_class_instance__(constructor_fn=classes_function_dict[funct_name], class_init_params=class_init_params)

                            sys.stdout = None
                            student_instance = __try_student_fn__(__create_class_instance__, constructor_fn=student_classes_function_dict[funct_name], class_init_params=class_init_params)
                            sys.stdout = orig_stdout

                        p_bar.postfix += ' | Running Fn'
                        # Check if we need to run the functions with an instance of a class or not
                        if funct_name in classes_function_dict:
                            start_time = timer()
                            real_solution = funct(solution_instance, **real_test_case)
                            end_time = timer()

                            sys.stdout = None
                            student_solution = __try_student_fn__(student_fn, student_instance, **student_test_case)
                            sys.stdout = orig_stdout
                        else:
                            real_solution = funct(**real_test_case)

                            sys.stdout = None
                            student_solution = __try_student_fn__(student_fn, **student_test_case)
                            sys.stdout = orig_stdout

                        # Check for student exceptions. If one test case has an exception, most likely the whole function is incorrect so we set a score of 0
                        if isinstance(student_solution, Exception):
                            if isinstance(student_solution, StudentTimeoutException):
                                feedback.write(f'\t[Test Case #{trial_idx+1}/{funct.max_trials}] Your solution took too long so it was terminated. As a reference, the TA function took {end_time - start_time:.10f}s, yours timed out with exception "{student_solution}"\n')
                                results.append(False)
                                continue
                            else:
                                feedback.write(f'Got exception {student_solution} when running function {funct_name}. Assigning a grade of 0 \n')
                                scores[funct_name] = 0
                                results.append(False)
                                break

                        # Compare answers between solution and student
                        p_bar.postfix += ' | Checking equality'
                        try:
                            # Determine which comparison function we will be using
                            if is_class_fn and has_custom_equality_fn:
                                r = funct.equality_fn(solution_instance, student_instance, real_solution, student_solution)
                            elif has_custom_equality_fn:
                                r = funct.equality_fn(real_solution, student_solution)
                            else:
                                r = compare_solutions(real_solution, student_solution)
                            results.append(r)

                            # Log failed test cases every 1/3 of the trials
                            p_bar.postfix += ' | Logging'
                            if not r and log_next_failed_case:
                                log_next_failed_case = False
                                solution_param_str = '' if real_test_case is None else real_test_case
                                student_param_str = '' if student_test_case is None else student_test_case

                                # solution_output_str = custom_grader_builder.equality_fn if has_custom_equality_fn else real_solution
                                # student_output_str = custom_grader_builder.equality_fn if has_custom_equality_fn else student_solution

                                feedback.write(f'Test Case #{trial_idx+1} Failed\n')
                                feedback.write(f'\t The Solution Outputs -> {funct_name}({solution_param_str}) = {real_solution}\n')
                                feedback.write(f'\tYour Solution Outputs -> {funct_name}({student_param_str}) = {student_solution} \n')

                                # Use our str function to print the student class
                                if is_class_fn:
                                    str_funct = solution_instance.__class__.__str__
                                    feedback.write(f'\tTA Class = \n{solution_instance}\n')
                                    feedback.write(f'\tYour Class = \n{str_funct(student_instance)}\n')
                        except Exception as ex:
                            feedback.write(f'Got exception {ex} when trying to grade function {funct_name} through comparing solutions with a custom equality function. Assigning a grade of 0 \n')
                            scores[funct_name] = 0
                            results.append(False)
                            break

                    # Score test cases
                    passed_cases = np.sum(results)
                    scores[funct_name] = (passed_cases / funct.max_trials)

                    feedback.write(f'\tPassed {passed_cases}/{funct.max_trials} test cases \n')
                    feedback.write(f'\tGrade = {scores[funct_name]*100:.0f}% \n')

            # Summarize
            feedback.write(f'\n ** Summary of all problem scores = \n\n \t{scores}\n\n')
            df = pd.concat([df, pd.DataFrame([scores])], ignore_index=True)

            # Final score
            final_score = 0
            if len(scores.keys()) > 0:
                for problem in scores.keys():
                    if problem == 'student':
                        continue
                    final_score += scores[problem]
                final_score /= len(scores.keys()) - 1

            feedback.write(f'Final grade = {final_score * args.max_grade:.2f}/{args.max_grade}\n')
            print(f"\t Student received {final_score * args.max_grade:.2f}/{args.max_grade}")

    print("Creating summary file with scores per problem")
    print(df)
    df.to_excel(solution_file.with_suffix('.xlsx'), index=False)

    print("Finished!")
    exit()
