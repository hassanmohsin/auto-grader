# ███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
#
# ▄████████    ▄████████         ▄████████ ███    █▄      ███      ▄██████▄     ▄██████▄     ▄████████    ▄████████ ████████▄     ▄████████    ▄████████
# ███    ███   ███    ███        ███    ███ ███    ███ ▀█████████▄ ███    ███   ███    ███   ███    ███   ███    ███ ███   ▀███   ███    ███   ███    ███
# ███    █▀    ███    █▀         ███    ███ ███    ███    ▀███▀▀██ ███    ███   ███    █▀    ███    ███   ███    ███ ███    ███   ███    █▀    ███    ███
# ███          ███               ███    ███ ███    ███     ███   ▀ ███    ███  ▄███         ▄███▄▄▄▄██▀   ███    ███ ███    ███  ▄███▄▄▄      ▄███▄▄▄▄██▀
# ███        ▀███████████      ▀███████████ ███    ███     ███     ███    ███ ▀▀███ ████▄  ▀▀███▀▀▀▀▀   ▀███████████ ███    ███ ▀▀███▀▀▀     ▀▀███▀▀▀▀▀
# ███    █▄           ███        ███    ███ ███    ███     ███     ███    ███   ███    ███ ▀███████████   ███    ███ ███    ███   ███    █▄  ▀███████████
# ███    ███    ▄█    ███        ███    ███ ███    ███     ███     ███    ███   ███    ███   ███    ███   ███    ███ ███   ▄███   ███    ███   ███    ███
# ████████▀   ▄████████▀         ███    █▀  ████████▀     ▄████▀    ▀██████▀    ████████▀    ███    ███   ███    █▀  ████████▀    ██████████   ███    ███
#                                                                                         ███    ███                                        ███    ███
#
# ██████████ by Jose G. Perez <DeveloperJose> ███████████████████████████████████████████████████████████████████████████████████████████████████████████
# ██████████████ Last Modified: 06/30/2022 ██████████████████████████████████████████████████████████████████████████████████████████████████████████████
# Requires a Linux machine for the timeout functionality. TODO: Figure out if we can get it to work on Windows
# TODO: Requires you to clean-up Colab Notebooks by getting rid of duplicate class definitions/functions and making sure it can run as a .py file
# Can only grade standalone functions and class functions
# Cannot grade class initializers (__init__ functions)
# If you want students to see their mistakes more easily implement the __str__() and __repr__() function for your classes
# TODO: Only parse classes/functions while ignoring loose code (it seems to not be easily possible)
# TODO: Allow each problem to have its own max_score
# TODO: Allow each problem to set its weight relative to the final score
# TODO: Allow each problem to pass timeout_s instead of defining a single one
# TODO: It is possible to grade students in separate threads, but might not be worth
import sys
import argparse
import pathlib
import inspect
import subprocess
import traceback
import copy
import os
import functools

import pandas as pd
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer


import wrapt_timeout_decorator

TIMEOUT_S = 0.1  # How many seconds should we allow student functions to run before terminating them
USE_MULTIPROCESSING = False

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


class StudentTimeoutException(Exception):
    """Exception that is raised when a student function takes too long to run and is timed out."""
    pass


class SilenceOutput:
    """Silences stdout. Useful to ignore student print() statements."""

    def __init__(self):
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout


class StudentCode:
    def __init__(self, student_dir: pathlib.Path, student_fpath: pathlib.Path, all_student_files: list):
        self.dir = student_dir
        self.fpath = student_fpath
        self.all_student_files = all_student_files
        assert self.dir.exists(), f'Student directory {student_dir} could not be found'
        assert self.fpath.exists(), f'Student file {student_fpath} could not be found'
        assert len(all_student_files) > 0, 'Student files must be positive'

        # Create directory to store feedback files
        self.feedback_dir = student_dir / 'feedback'
        if not self.feedback_dir.exists():
            self.feedback_dir.mkdir()

    def write_feedback(self, msg):
        """Writes feedback to the student output file."""
        assert hasattr(self, 'feedback'), f'You must use the with statement when using this class'
        self.feedback.write(f'{msg}\n')

    def import_module(self):
        """Imports the student module, its functions, and the class constructors."""
        import_exception = None
        # Change current working directory to the solution directory for importing
        original_cwd = os.getcwd()
        os.chdir(self.fpath.parent)
        with SilenceOutput():
            try:
                self.module = __import__(self.fpath.stem)
                self.fns, self.constructors = Grader.get_module_functions(self.module)
            except Exception as ex:
                self.write_feedback(f'[AutoGrader] Could not import module due to exception {ex}')
                self.write_feedback(traceback.format_exc())
                import_exception = ex

        # Change the working directory back to its original state
        os.chdir(original_cwd)
        return import_exception

    def has_fn(self, fn_name):
        """Determines if the given function name is in the student code"""
        return fn_name in self.fns.keys()

    def run_fn(self, fn_name, *args, **kwargs):
        """Runs the provided function given the function name."""
        assert hasattr(self, 'fns'), 'You must call import_module() first before calling run_fn'

        if fn_name not in self.fns.keys():
            raise Exception(f'Did not find {fn_name} in the student code')

        with SilenceOutput():
            fn = self.fns[fn_name]
            return self.__run_fn_timeout__(fn, *args, **kwargs)

    def create_class_instance(self, fn_name, **constructor_kwargs):
        """Creates an instance of the class needed to run the provided function."""
        if fn_name not in self.constructors.keys():
            raise Exception(f'Function {fn_name} was expected to be inside a class but it was not')

        constructor_fn = self.constructors[fn_name]
        with SilenceOutput():
            if constructor_kwargs:
                return constructor_fn(**constructor_kwargs)
            else:
                return constructor_fn()

    def __run_fn_timeout__(self, fn, *args, **kwargs):
        """Runs the given function with a time limit."""
        try:
            return wrapt_timeout_decorator.timeout(TIMEOUT_S, use_signals=True, timeout_exception=StudentTimeoutException)(fn)(*args, **kwargs)
        except StudentTimeoutException as ex:
            self.tb = traceback.format_exc()
            return ex
        except OSError as ex:
            self.tb = traceback.format_exc()
            return ex
        except Exception as ex:
            self.tb = traceback.format_exc()
            return ex

    def __enter__(self):
        # Determine if it's a Blackboard bulk download file in the format
        # {assignment name}_{student username}_attempt_{date}_{filename}
        split_filename = self.fpath.stem.split('_')
        is_bulk_blackboard = len(split_filename) >= 4

        # If it's a Blackboard file, sort all attempts based on date and assign this attempt a simple number
        # for the feedback output file. Otherwise make the feedback output file the same name as the attempt file
        if is_bulk_blackboard:
            student_name = split_filename[1]
            date = split_filename[3]

            # Gather and then sort all Blackboard attempts based on date
            other_attempt_dates = []
            for att_path in self.all_student_files:
                split_att_path = att_path.stem.split('_')
                if len(split_att_path) < 4 or split_att_path[1] != student_name:
                    continue
                other_attempt_dates.append(split_att_path[3])
            other_attempt_dates = sorted(other_attempt_dates)

            attempt_number = other_attempt_dates.index(date)+1

            self.feedback_filename = f'{student_name}_attempt{attempt_number}.bbtxt'
            print(f'[Auto-Grader] Grading {student_name} blackboard attempt {attempt_number} / {len(other_attempt_dates)}')
        else:
            self.feedback_filename = f'{self.fpath}.bbtxt'
            print(f'[Auto-Grader] Grading {self.fpath} attempt')

        # Open the feedback file
        self.feedback = open(self.feedback_dir / self.feedback_filename, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.write_feedback(f'[AutoGrader] Got {exc_type} {exc_val}, unable to proceed with grading')
            self.write_feedback(exc_tb)

        print(f"\t*StudentCode() has exited | ExceptionType={exc_type} | Exception={exc_val}")
        self.feedback.close()


class Grader:
    def __init__(self, solution_file, student_dir, max_grade, students):
        self.solution_file = pathlib.Path(solution_file)
        self.student_dir = pathlib.Path(student_dir)
        self.max_grade = max_grade
        self.students = students

        assert self.solution_file.exists(), f'Solution file {self.solution_file} does not exist'
        assert self.student_dir.exists(), f'Student directory {self.student_dir} does not exist'
        assert isinstance(self.max_grade, (int, float)), f'Max grade has to be int or float, cannot be {self.max_grade}'

        # This is where the summary excel file will be saved
        self.summary_path = pathlib.Path(self.student_dir / 'summary.xlsx')

        # Add directories to the path so we can import the modules more easily later
        # and for the student's code to find files with open() as well
        sys.path.append(str(student_dir.resolve()))
        sys.path.append(str(self.solution_file.parent.resolve()))

        # print(f'[Debug] sys.path = {sys.path}')

    @staticmethod
    def get_module_functions(module):
        """Gets all the functions in the module and all class constructors."""
        # First get all module functions that are not inside a class
        fns = dict(inspect.getmembers(module, inspect.isfunction))

        # Get the classes if any and extract their functions
        classes = inspect.getmembers(module, inspect.isclass)
        constructors = {}

        # print(f"[Debug] Inspected classes = {classes}")
        for class_name, class_obj in classes:
            for fn_name, fn in inspect.getmembers(class_obj, inspect.isfunction):
                assert fn_name not in fns.keys(), f'Two functions have the same name, {fn_name}'
                fns[fn_name] = fn
                constructors[fn_name] = class_obj

        return fns, constructors

    def create_class_instance(self, fn_name, **constructor_kwargs):
        """Create an instance of a class from the solution module for a given function."""
        constructor_fn = self.solution_constructors[fn_name]
        if constructor_kwargs:
            return constructor_fn(**constructor_kwargs)
        else:
            return constructor_fn()

    def override_libraries(self):
        """
        Override some library functions that we don't want to run in the grader.

        For example, we usually don't want to run calls to draw() functions in matplotlib.
        You can also use this to provide a fake/dummy module that the student code will use,
            a good example of such a case is to provide a google.colab module locally.
        """
        import matplotlib.pyplot
        import empty_module

        sys.modules['matplotlib.pyplot'].show = lambda: None
        sys.modules['google.colab'] = empty_module

    def import_solution(self):
        """Imports the solution file, its functions, and its classes"""
        print('[Debug] Importing solution module')
        # Change current working directory to the solution directory for importing
        original_cwd = os.getcwd()
        os.chdir(self.solution_file.parent)

        # Import the solution file
        self.solution_module = __import__(self.solution_file.stem)

        # Change the working directory back to its original state
        os.chdir(original_cwd)

        # Import functions
        self.solution_fns, self.solution_constructors = self.get_module_functions(self.solution_module)

        # Check that we didn't forget to annotate a function
        # and also remove functions we do not want to grade
        to_remove = []
        for fn_name, fn in self.solution_fns.items():
            # Check for @grader.no_test_cases annotation and skip them
            if hasattr(fn, 'no_test_cases'):
                to_remove.append(fn_name)
            else:
                assert hasattr(fn, 'gen_fn_params'), f'[Debug] Not grading {fn_name} due to lack of annotation. Did you forget to annotate it?'

        for t in to_remove:
            del self.solution_fns[t]
            print(f'[Debug] Not grading {t}')

        # Create dataframe where we will keep track of student scores
        print(f"[Debug] Grading {len(self.solution_fns)} functions | {self.solution_fns.keys()}")
        self.df = pd.DataFrame(columns=['student'] + [fname for (fname) in self.solution_fns.keys()])

    def convert_student_jupyter_to_py(self):
        """Converts all student Jupyter notebooks into regular code files"""
        for fpath in self.student_dir.glob("*.ipynb"):
            # If the file has already been converted skip it (don't convert it again!)
            if fpath.with_suffix(".py").exists():
                continue

            print(f"[Auto-Grader] Converting jupyter notebook {fpath.stem} into regular code")
            subprocess.call(['jupyter', 'nbconvert', '--to', 'script', str(fpath)])

            # Rename from .txt to .py
            output_fpath = fpath.with_suffix(".txt")
            assert output_fpath.exists(), f'[Debug] Converted Jupyter file {output_fpath} could not be found'
            output_fpath.rename(output_fpath.with_suffix('.py'))

    def grade_all_students(self):
        """Grades all students in the given student directory"""
        # Find all the student code files
        all_student_files = list(self.student_dir.glob("*.py"))

        # If the --student parameter is passed in the command-line, grade only those students
        if self.students:
            temp_student_files = []
            for student_name in args.students:
                # Look for all Blackboard attempts
                attempts = []
                for att_path in all_student_files:
                    att_path_split = att_path.stem.split('_')
                    if len(att_path_split) >= 4 and att_path_split[1] == student_name:
                        attempts.append(att_path)

                # If we find Blackboard attempts, grade those
                # If not, try to open the parameter as a file with
                if len(attempts) > 0:
                    temp_student_files.extend(sorted(attempts))
                else:
                    path = pathlib.Path(student_name)
                    assert path.exists(), f'Could not find {student_name} blackboard attempts or as a file'
                    temp_student_files.append(path)

            all_student_files = temp_student_files

        # Grade the students
        for fpath in all_student_files:
            with StudentCode(self.student_dir, fpath, all_student_files) as stu_code:
                import_exception = stu_code.import_module()
                if import_exception:
                    print(f'\t*Could not import student module due to exception "{import_exception}"')
                else:
                    self.grade_one_student(stu_code)

        # Create summary file
        print("Creating summary file with scores per problem")
        print(self.df)
        self.df.to_excel(self.summary_path, index=False)

    def grade_one_student(self, stu_code: StudentCode):
        """Grades all functions of a given student."""
        # We will keep track of all problem grades in this dictionary
        scores = {'student': stu_code.feedback_filename}
        total_score = 0

        # Open a progress bar for visualization in the command-line
        with tqdm(total=len(self.solution_fns.keys())) as p_bar:
            # Go through all functions in the solution
            for fn_name, sol_fn in self.solution_fns.items():
                p_bar.update()
                p_bar.desc = f'Grading {fn_name}'
                stu_code.write_feedback(f'******************** [AutoGrader] Grading {fn_name} ********************')

                # Check if the student actually has the function
                if not stu_code.has_fn(fn_name):
                    stu_code.write_feedback(f'Did not find {fn_name} in the student code, assigning a grade of 0 to this problem')
                    scores[fn_name] = 0
                    continue

                # Grade the function
                scores[fn_name] = self.grade_one_function(fn_name, sol_fn, p_bar, stu_code)
                total_score += scores[fn_name]

        # Summarize scores
        stu_code.write_feedback(f'\n ** Summary of all problem scores = \n\n \t{scores}\n')

        # Update dataframe
        self.df = pd.concat([self.df, pd.DataFrame([scores])], ignore_index=True)

        # Compute final score out as an integer between 0 and 1
        # TODO: Currently all problems are evenly weighted, perhaps we should allow the annotation to set the weight
        final_score = total_score / len(self.solution_fns.keys())

        # Convert the final score to the scale passed by the user
        final_score = final_score * self.max_grade

        # Log final scores
        stu_code.write_feedback(f'Final grade = {final_score:.2f}/{self.max_grade}')
        print(f'\t*Student received {final_score:.2f}/{self.max_grade}')

    def grade_one_function(self, fn_name, sol_fn, p_bar: tqdm, stu_code: StudentCode):
        """Runs all test cases and grades the provided function for one specific student."""
        # Gather the metadata provided by the decorators
        is_class_fn = hasattr(sol_fn, 'gen_class_params')
        has_custom_equality_fn = hasattr(sol_fn, 'equality_fn')
        n_trials: int = sol_fn.max_trials

        # Log detailed failed cases every 1/3rd of the trials
        log_freq = n_trials // 3
        log_next_failed_case = True

        # Extend progress bar and keep track of test case results
        p_bar.total += n_trials
        test_case_results = []

        # How many exceptions we will allow before stopping the trials
        n_exception_patience = 5

        for trial_idx in range(n_trials):
            header = f'Test Case #{trial_idx+1}/{n_trials}'
            p_bar.update()
            p_bar.postfix = header

            # Check if we need to log the next failed test case
            if trial_idx % log_freq == 0:
                log_next_failed_case = True

            # Generate function parameters
            sol_params = sol_fn.gen_fn_params()
            stu_params = sol_params

            # Create a new class instance every X iterations of the function we are testing when testing class functions
            if is_class_fn and trial_idx % sol_fn.trials_per_instance == 0:
                p_bar.postfix = f'{header} | Creating classes'
                # Decouple parameter references
                # TODO: Determine if this needs to be done every single time or only for classes
                stu_params = copy.deepcopy(sol_params)

                # IMPORTANT! Both classes must be initialized with the same parameters!
                class_init_params = sol_fn.gen_class_params()
                sol_instnc = self.create_class_instance(fn_name, **class_init_params)

                try:
                    stu_instnc = stu_code.create_class_instance(fn_name, **class_init_params)
                except Exception as ex:
                    test_case_results.append(False)

                    stu_code.write_feedback(f'[{header}] Got exception [{ex}] when creating class {sol_instnc.__class__.__name__}. Stopping grading function early')
                    stu_code.write_feedback(f'\t{traceback.format_exc()}\n')
                    break

            # Check if we need to run the functions with an instance of a class or not
            # We also time the solution in seconds for logging later in case of student code timeouts
            p_bar.postfix = f'{header} | Running fns'
            if is_class_fn:
                start_t = timer()
                sol_output = sol_fn(sol_instnc, **sol_params)
                end_t = timer()

                stu_output = stu_code.run_fn(fn_name, stu_instnc, **stu_params)
            else:
                start_t = timer()
                sol_output = sol_fn(**sol_params)
                end_t = timer()

                stu_output = stu_code.run_fn(fn_name, **stu_params)

            # Check for exceptions
            if isinstance(stu_output, Exception):
                n_exception_patience -= 1
                test_case_results.append(False)

                if isinstance(stu_output, StudentTimeoutException):
                    stu_code.write_feedback(f'[{header}] Your code took too long to run so it was timed out and stopped. Remaining exceptions allowed = {n_exception_patience}')
                    stu_code.write_feedback(f'\tAs a reference, the solution took {end_t - start_t:.6f}s to run. The time limit is {TIMEOUT_S}s \n')
                else:
                    stu_code.write_feedback(f'[{header}] Got exception "{stu_output}" when running function {fn_name}({stu_params}). Remaining exceptions allowed = {n_exception_patience}')
                    stu_code.write_feedback(f'\t{stu_code.tb}\n')

                # Stop running test cases when patience runs out. If patience remains, continue to the next test case
                if n_exception_patience <= 0:
                    stu_code.write_feedback(f'# Stopping grading function early due to repeated exceptions')
                    p_bar.n = p_bar.total
                    break
                else:
                    continue

            # Compare answers between the solution and the student
            p_bar.postfix = f'{header} | Checking equality'
            try:
                # Determine which comparison function we will be using
                if is_class_fn and has_custom_equality_fn:
                    has_passed_test = sol_fn.equality_fn(sol_instnc, stu_instnc, sol_output, stu_output)
                elif has_custom_equality_fn:
                    has_passed_test = sol_fn.equality_fn(sol_output, stu_output)
                else:
                    has_passed_test = compare_solutions(sol_output, stu_output)
                # Add pass/fail result to the total list
                test_case_results.append(has_passed_test)
            except Exception as ex:
                stu_code.write_feedback(f'### Got exception {ex} when grading {fn_name} when trying to compare outputs. Stopping grading early')
                stu_code.write_feedback(traceback.format_exc())
                test_case_results.append(False)
                p_bar.n = p_bar.total
                break

            # Log failed test cases after a certain amount of iterations
            p_bar.postfix = f'{header} | Logging'
            if not has_passed_test and log_next_failed_case:
                log_next_failed_case = False

                stu_code.write_feedback(f'{header} failed')
                stu_code.write_feedback(f'\t The Solution Outputs -> {fn_name}({sol_params})={sol_output}')
                stu_code.write_feedback(f'\tYour Solution Outputs -> {fn_name}({stu_params})={stu_output}\n')

                # Use our str function to print the student class
                if is_class_fn:
                    str_funct = sol_instnc.__class__.__str__
                    stu_code.write_feedback(f'\tSolution Class = \n{sol_instnc}')
                    stu_code.write_feedback(f'\tYour Class = \n{str_funct(stu_instnc)}\n')

        # Once all trials are completed, we compute the score between 0 and 1
        n_passed_cases = np.sum(test_case_results)
        total_score = n_passed_cases / n_trials

        stu_code.write_feedback(f'###>>> Passed {n_passed_cases}/{n_trials} test cases')
        stu_code.write_feedback(f'###>>> Grade for function "{fn_name}" = {total_score*100:.0f} / 100\n')
        return total_score


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


if __name__ == '__main__':
    # Parse arguments from the command-line
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=str, required=True)
    parser.add_argument("--student_dir", type=str, required=True)
    parser.add_argument("--max_grade", type=int, required=True)
    parser.add_argument("--students", nargs='+', required=False)
    args = parser.parse_args()

    print("""
        ███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████

        ▄████████    ▄████████         ▄████████ ███    █▄      ███      ▄██████▄     ▄██████▄     ▄████████    ▄████████ ████████▄     ▄████████    ▄████████ 
        ███    ███   ███    ███        ███    ███ ███    ███ ▀█████████▄ ███    ███   ███    ███   ███    ███   ███    ███ ███   ▀███   ███    ███   ███    ███ 
        ███    █▀    ███    █▀         ███    ███ ███    ███    ▀███▀▀██ ███    ███   ███    █▀    ███    ███   ███    ███ ███    ███   ███    █▀    ███    ███ 
        ███          ███               ███    ███ ███    ███     ███   ▀ ███    ███  ▄███         ▄███▄▄▄▄██▀   ███    ███ ███    ███  ▄███▄▄▄      ▄███▄▄▄▄██▀ 
        ███        ▀███████████      ▀███████████ ███    ███     ███     ███    ███ ▀▀███ ████▄  ▀▀███▀▀▀▀▀   ▀███████████ ███    ███ ▀▀███▀▀▀     ▀▀███▀▀▀▀▀   
        ███    █▄           ███        ███    ███ ███    ███     ███     ███    ███   ███    ███ ▀███████████   ███    ███ ███    ███   ███    █▄  ▀███████████ 
        ███    ███    ▄█    ███        ███    ███ ███    ███     ███     ███    ███   ███    ███   ███    ███   ███    ███ ███   ▄███   ███    ███   ███    ███ 
        ████████▀   ▄████████▀         ███    █▀  ████████▀     ▄████▀    ▀██████▀    ████████▀    ███    ███   ███    █▀  ████████▀    ██████████   ███    ███ 
                                                                                                ███    ███                                        ███    ███ 

        ██████████ by Jose G. Perez <DeveloperJose> ███████████████████████████████████████████████████████████████████████████████████████████████████████████
    """)

    # Grading Pipeline:
    # 1. Prepare grader
    # 2. Override some libraries we don't want to run such as draw() or Colab's file.upload()
    # 3. Import the solution file functions and classes
    # 4. Convert jupyter .ipynb notebooks to .py files
    # 5. Grade all student code files
    print(f'Grading {args.solution} with student code located at {args.student_dir}')
    grader = Grader(pathlib.Path(args.solution), pathlib.Path(args.student_dir), args.max_grade, args.students)
    grader.override_libraries()
    grader.import_solution()
    grader.convert_student_jupyter_to_py()
    grader.grade_all_students()

    print("Finished grading!")
