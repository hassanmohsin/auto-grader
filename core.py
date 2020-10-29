import pathlib
import sys

import big_o


class PersistentLocals(object):
    def __init__(self, func):
        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event == 'return':
                self._locals = frame.f_locals.copy()

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals


def auto_grade_lab(student_code_dir, lab_str, problem_dict):
    # We will store the scores in a table per problem
    csv_table = open(f'{lab_str}.csv', 'w')

    if type(student_code_dir) == str:
        student_code_dir = pathlib.Path(student_code_dir)

    for fpath in student_code_dir.glob("*.py"):
        # Import and replace the current lab source
        exec(f"lab_source = None")

        import_exception = False
        module_path = str(fpath).replace('\\', '.').replace('.py', '')

        # Silence print()
        orig_stdout = sys.stdout
        sys.stdout = None
        try:
            module = __import__(module_path)
            exec(f"from data import {fpath.stem} as lab_source")
        except Exception as ex:
            import_exception = ex

        # Redirect print() to a file
        f = open(lab_str + "_" + fpath.stem + ".bbtxt", 'w')
        sys.stdout = f

        print(">> AutoGrader v1.1 by Jose G. Perez (TA)")
        print(f">> Assignment: {lab_str}")
        print(f">> Student: {fpath.stem}")

        if import_exception:
            print(">> There was an exception thrown by a cell while trying to import. Run all your cells next time to make sure no exceptions are thrown.")
            print(">> Exception: ", repr(import_exception))

        totals = []
        for idx, (problem_str, (problem_max_points, problem_grade_func)) in enumerate(problem_dict.items()):
            print(f"------------------------- [Problem {idx + 1}] {problem_str} ({problem_max_points}pts)")
            exec(f"totals.append(problem_grade_func(lab_source))")
            print()

        print("------------------------- Overall Grades")
        for idx, (problem_str, (problem_max_points, problem_grade_func)) in enumerate(problem_dict.items()):
            print(f"--==> Problem {idx + 1}={totals[idx]}/{problem_max_points}pts")

        print(f"--==> Total={sum(totals)}/100pts")

        sys.stdout = orig_stdout
        f.close()

        csv_table.write(f'{fpath.stem},')
        for total in totals:
            csv_table.write(f'{total},')
        csv_table.write('\n')

    csv_table.close()


def auto_grade_constant_space(max_space_points, func, *params):
    try:
        # Find out how many lists are in the local space of the function
        ob = PersistentLocals(func)
        ob(*params)

        local_vars = ob.locals
        local_list_vars = [var for var_name, var in local_vars.items() if type(var) == list]

        # Find out how many lists are in the input
        input_list_vars = [var for var in params if type(var) == list]

        # If there number of lists in the input and the local space is the same, it's constant space
        is_constant_space = len(local_list_vars) == len(input_list_vars)

        # Give full-points if the space complexity is constant, half otherwise
        space_complexity_grade = 0
        if is_constant_space:
            space_complexity_grade = max_space_points
            print(f"=> Automatic Space Analysis = {space_complexity_grade}/{max_space_points}pts, Constant Space [{len(input_list_vars)} list in the input, {len(local_list_vars)} list in the function]")
        else:
            space_complexity_grade = max_space_points / 2
            print(f"=> Automatic Space Analysis = {space_complexity_grade}/{max_space_points}pts, NOT Constant Space [{len(input_list_vars)} lists in the input, {len(local_list_vars)} lists in the function]")

        return space_complexity_grade
    except Exception as ex:
        print(f"An exception was thrown by the student's lab: {repr(ex)}. This might affect the final score.")
        print(f"=> Automatic Space Analysis = 0/{max_space_points}pts, could not automatically measure space complexity due to an exception")
        return 0


def auto_grade_time_complexity(measure_func, data_gen_func, max_time_points, correct_time_type):
    try:
        # Approximate the big-o for the function
        best_time, other_times = big_o.big_o(measure_func, data_gen_func, min_n=10, max_n=2500, n_measures=10,
                                             n_repeats=1, n_timings=1, classes=[big_o.complexities.Constant, big_o.complexities.Linear, big_o.complexities.Quadratic, big_o.complexities.Exponential])

        # Give full-points if the time complexity is what we expected
        time_complexity_grade = 0
        if type(best_time) is correct_time_type:
            time_complexity_grade = max_time_points
        else:
            # If the time complexity is not exactly what we expected, compare the residuals to see if we were close
            best_time_coeff = 0
            correct_time_coeff = 0
            for ctype, coeff in other_times.items():
                if type(ctype) == type(best_time):
                    best_time_coeff = coeff
                if type(ctype) == correct_time_type:
                    correct_time_coeff = coeff

            # If the measured time is close to the correct time, count it as correct
            if abs(best_time_coeff - correct_time_coeff) <= 1e-10:
                best_time = correct_time_type.__name__
                time_complexity_grade = max_time_points
            else:
                time_complexity_grade = max_time_points / 2

        # Let the student know what we measured and their time complexity grade
        print(f"=> Automatic Time Analysis = {time_complexity_grade}/{max_time_points}pts [Expected='{correct_time_type.__name__}', Measured='{best_time}']")
        return time_complexity_grade
    except Exception as ex:
        print(f"An exception was thrown by the student's lab: {repr(ex)}. This might affect the final score.")
        print(f"=> Automatic Time Analysis = 0/{max_time_points}pts [Expected='{correct_time_type.__name__}' but could not measure due to an exception]")
        return 0


def auto_grade_test_cases(problem, description, max_points, max_test_case_points, test_case_gen_func=lambda: []):
    print(f'--==> Problem: {problem} ({max_points}pts)')
    print(f'--==> Description: {description}')

    try:
        # Run the tests and generate their boolean values
        test_cases = test_case_gen_func()

        # Count which ones passed and which ones failed
        passed_tests = [idx + 1 for idx, passed_test in enumerate(test_cases) if passed_test]
        failed_tests = [idx + 1 for idx, passed_test in enumerate(test_cases) if not passed_test]

        # Assign points equally per test case
        points_per_test = max_test_case_points / len(test_cases)
        total_points = points_per_test * len(passed_tests)

        # Let the student know which test cases failed and their test case grade
        print(f"=> TA Test Cases: {total_points:.2f}/{max_test_case_points}, passed {len(passed_tests)} out of {len(test_cases)} test cases.")
        if len(failed_tests) > 0:
            print(f"Failed Test Cases: {failed_tests}. If you wish to know what exactly failed just email me these numbers and I'll send you the test cases you request.")

        return total_points

    except Exception as ex:
        print(f"An exception was thrown by the student's lab: {repr(ex)}. This might affect the final score.")
        print(f"=> TA Test Cases: 0/{max_test_case_points}pts, passed 0 of the test cases due to an exception.")
        return 0


def silent_print(*str):
    old_stdout = sys.stdout
    sys.stdout = None
    print(str)
    sys.stdout = old_stdout


# Test automatic time complexity grading
if __name__ == '__main__':
    def gen_n(n):
        return int(n)


    def constant_time(n):
        return 0


    def linear_time(n):
        for i in range(n):
            silent_print(n)


    def quadratic_time(n):
        for i in range(n):
            for j in range(n):
                silent_print(i, j)


    print("Should all be constant")
    auto_grade_time_complexity(measure_func=constant_time, data_gen_func=gen_n, max_time_points=15, correct_time_type=big_o.complexities.Constant)

    print("Should all be linear")
    auto_grade_time_complexity(measure_func=linear_time, data_gen_func=gen_n, max_time_points=15, correct_time_type=big_o.complexities.Linear)
    auto_grade_time_complexity(measure_func=linear_time, data_gen_func=gen_n, max_time_points=15, correct_time_type=big_o.complexities.Quadratic)
    auto_grade_time_complexity(measure_func=linear_time, data_gen_func=gen_n, max_time_points=15, correct_time_type=big_o.complexities.Constant)

    print("Should be quadratic")
    auto_grade_time_complexity(measure_func=quadratic_time, data_gen_func=gen_n, max_time_points=15, correct_time_type=big_o.complexities.Quadratic)
