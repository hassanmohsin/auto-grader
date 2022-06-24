# Author: Jose G. Perez
# Requires a Linux machine for the timeout functionality
# TODO: Only parse classes/functions while ignoring loose code
# TODO: Make it one main decorator
# TODO: Refactor grading as a class to separate the process into easily read functions
# TODO: Change parameter generation from single function to separate functions?
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

TIMEOUT_S = 0.1 # How many seconds should we allow student functions to run before terminating them

def generate_class_test_cases(param_gen_fn = lambda: None, equality_fn=None, class_init_param_fn=None, trials=5000):
    if type(param_gen_fn) is list:
        test_cases = param_gen_fn
    elif callable(param_gen_fn):
        test_cases = []
        for _ in range(trials):
            test_cases.append(param_gen_fn())
    else:
        raise Exception('param_gen_fn must be a list of test cases or a function')
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.meta = {'test_cases': test_cases, 
                        'equality_fn': equality_fn, 
                        'class_init_param_fn': class_init_param_fn, 
                        'param_count': len(inspect.signature(func).parameters.values()),
                        }
        return wrapper
    return decorator

def no_test_cases():
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.meta = {'test_cases': None}
        return wrapper
    return decorator

def compare_solutions(real_solution, student_solution):
    if type(real_solution) is np.ndarray:
        return np.array_equal(real_solution, student_solution)
    elif type(real_solution) is bool:
        return real_solution == student_solution
    elif type(real_solution) is int or type(real_solution) is float or type(real_solution) is np.int32 or type(real_solution) is np.float32:
        return abs(real_solution - student_solution) < .001
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

@wrapt_timeout_decorator.timeout(TIMEOUT_S, use_signals=True)
def try_student_fn(student_fn, class_instnc, args):
    try:
        return run_fn(student_fn, class_instnc, args)
    except OSError as ex:
        return ex
    except Exception as ex:
        return ex

def run_fn(fn, class_instnc, args):
    try:
        if class_instnc:
            if type(args) is tuple:
                return fn(class_instnc, *args)
            elif args is None:
                return fn(class_instnc)
            else:
                return fn(class_instnc, args)
        else:
            if type(args) is tuple:
                return fn(*args)
            elif args is None:
                return fn()
            else:
                return fn(args)
    except TimeoutError as ex:
        return ex

def create_class_instance(constructor_fn, class_init_params):
    if class_init_params:
        return constructor_fn(class_init_params)
    else:
        return constructor_fn()

if __name__ == '__main__':
    # Parse arguments from the command-line
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=str, required=True)
    parser.add_argument("--student_dir", type=str, required=True)
    parser.add_argument("--max_grade", type=int, required=True)
    args = parser.parse_args()

    # Import the solution file
    solution_file = pathlib.Path(args.solution)
    if not solution_file.exists():
        raise Exception(f'solution file {solution_file} does not exist')
    module = __import__(solution_file.stem)

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
        try:
            funct.meta.keys()
            # print('[Debug]', funct_name, funct.meta.keys())
        except AttributeError:
            to_remove.append((funct_name, funct))
            raise Exception(f'[Debug] Not grading {funct_name} due to lack of annotation. Did you forget to annotate it?')

        # Check for @grader.no_test_cases annotation and skip them
        if funct.meta['test_cases'] is None:
            to_remove.append((funct_name, funct))

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
            scores = { 'student': f'{student_name}_attempt_{attempt_number}' }
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
                    p_bar.total += len(funct.meta["test_cases"])
                    results = []
                    for idx, real_test_case in enumerate(funct.meta['test_cases']):
                        p_bar.update()
                        p_bar.postfix = f'Test Case {idx}/{len(funct.meta["test_cases"])}'
                        student_test_case = real_test_case

                        # Create a new class instance every 100 iterations of the function we are testing when testing class functions
                        if funct_name in classes_function_dict and idx % 100 == 0:
                            student_test_case = copy.deepcopy(real_test_case)

                            # IMPORTANT! Both classes must be initialized with the same parameters
                            class_init_params = funct.meta['class_init_param_fn']()
                            solution_instance = create_class_instance(constructor_fn = classes_function_dict[funct_name], class_init_params = class_init_params)
                            
                            sys.stdout = None
                            student_instance = try_student_fn(create_class_instance, None, args = (student_classes_function_dict[funct_name], class_init_params))
                            sys.stdout = orig_stdout

                        p_bar.postfix += ' | Running Fn'
                        # Check if we need to run the functions with an instance of a class or not
                        if funct_name in classes_function_dict:
                            start_time = timer()
                            real_solution = run_fn(funct, solution_instance, real_test_case)
                            end_time = timer()

                            sys.stdout = None
                            student_solution = try_student_fn(student_fn, student_instance, student_test_case)
                            sys.stdout = orig_stdout
                        else:
                            real_solution = run_fn(funct, None, real_test_case)
                            
                            sys.stdout = None
                            student_solution = try_student_fn(student_fn, None, student_test_case)
                            sys.stdout = orig_stdout

                        # Check for student exceptions. If one test case has an exception, most likely the whole function is incorrect so we set a score of 0
                        if isinstance(student_solution, Exception):
                            if isinstance(student_solution, TimeoutError):
                                feedback.write(f'\t[Test Case #{idx}/{len(funct.meta["test_cases"])}] Your solution took too long so it was terminated. As a reference, the TA function took {end_time - start_time:.10f}s, yours timed out with exception {student_solution} \n')
                                results.append(False)
                            else:
                                feedback.write(f'Got exception {student_solution} when running function {funct_name}. Assigning a grade of 0 \n')
                                scores[funct_name] = 0
                                break

                        p_bar.postfix += ' | Checking equality'
                        # Determine if we want to compare class instances instead of return types
                        if funct_name in classes_function_dict and funct.meta['equality_fn']:
                            r = funct.meta['equality_fn'](solution_instance, student_instance, real_solution, student_solution)

                        # Determine if we want to check the equality of a specific parameter index #
                        elif type(funct.meta['equality_fn']) is int:
                            if type(real_test_case) is tuple:
                                param_idx = funct.meta['equality_fn']
                                real_solution = real_test_case[param_idx]
                                student_solution = student_test_case[param_idx]
                            else:
                                real_solution = real_test_case
                                student_solution = student_test_case

                        # Convert types when it makes sense
                        elif type(real_solution) is np.ndarray and type(student_solution) is list and len(real_solution.shape) == 1:
                            student_solution = np.array(student_solution, dtype=real_solution.dtype)

                        # Make sure the types match for both solutions
                        elif type(real_solution) != type(student_solution):
                            feedback.write(f'Real solution for problem {funct_name} has type {type(real_solution)} but student solution has type {type(student_solution)}, assigning a grade of 0 to this problem\n')
                            scores[funct_name] = 0
                            results = []
                            break

                        # Compare answers
                        elif callable(funct.meta['equality_fn']):
                            r = funct.meta['equality_fn'](real_solution, student_solution)
                        else:
                            r = compare_solutions(real_solution, student_solution)

                        # Log failed test cases every once in a while
                        # p_bar.postfix += ' | Logging'
                        # if not r and idx % 1000 == 0:
                        #     feedback.write(f'Failed Case: Solution -> {funct_name}({real_test_case}) = {real_solution} | Student -> {funct_name}({student_test_case}) = {student_solution} \n')

                        #     # TODO: This only works for lab 1
                        #     if funct_name in classes_function_dict:
                        #         feedback.write(f'My   Sudoku = {solution_instance.S} \nYour Sudoku ={student_instance.S}\n')

                        results.append(r)

                    # Score test cases
                    if len(results) > 0:
                        passed_cases = np.sum(results)
                        scores[funct_name] = (passed_cases / len(results))
                    else:
                        passed_cases = 0
                        scores[funct_name] = 0
                
                    feedback.write(f'\tPassed {passed_cases}/{len(results)} test cases \n')
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