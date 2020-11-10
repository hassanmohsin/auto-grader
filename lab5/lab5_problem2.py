import inspect

from core import *


def grade_problem(lab5_source):
    # ======================================================= TEST-CASES
    def heap_test():
        # Get all the classes inside the student's code
        classes_list = inspect.getmembers(lab5_source, inspect.isclass)

        for class_name, class_obj in classes_list:
            if 'heap' in class_name.lower():
                print(f"=> Found a heap class named '{class_name}'")
                return [True]

        print(f"=> Did not find a heap class inside the module")
        return [False]

    def heap_sort_tests():
        dict1 = {
            "password123": 10,
            "12345": 8,
            "abCdFEggA.1.xkal;s": 1,
            "password": 100,
            "burst": 2,
        }
        test1 = grade_test_case_helper(test_number=1, source_func=lab5_source.heap_sort, source_params=[dict1, 1], expected_value=["password"], extra_info='')
        test2 = grade_test_case_helper(test_number=2, source_func=lab5_source.heap_sort, source_params=[dict1, 2], expected_value=["password", "password123"], extra_info='')
        test3 = grade_test_case_helper(test_number=3, source_func=lab5_source.heap_sort, source_params=[dict1, 5], expected_value=["password", "password123", "12345", "burst", "abCdFEggA.1.xkal;s"], extra_info='')

        dict2 = {
            "X": 10,
            "B": 10,
            "D": 10,
            "C": 10,
            "A": 10
        }
        test4 = grade_test_case_helper(test_number=4, source_func=lab5_source.heap_sort, source_params=[dict2, 1], expected_value=["A"], extra_info='')
        test5 = grade_test_case_helper(test_number=5, source_func=lab5_source.heap_sort, source_params=[dict2, 2], expected_value=["A", "B"], extra_info='')
        test6 = grade_test_case_helper(test_number=6, source_func=lab5_source.heap_sort, source_params=[dict2, 5], expected_value=["A", "B", "C", "D", "X"], extra_info='')

        return [test1, test2, test3, test4, test5, test6]

    # ======================================================= AUTO-GRADING
    problem2_1 = auto_grade_test_cases(problem='Use a heap',
                                       description='The student must use a heap for this problem',
                                       max_points=10,
                                       max_test_case_points=10,
                                       test_case_gen_func=heap_test)

    problem2_2 = auto_grade_test_cases(problem='heap_sort',
                                       description='The student just has to provide a solution to the problem, no time or space complexity requirements',
                                       max_points=40,
                                       max_test_case_points=40,
                                       test_case_gen_func=heap_sort_tests)
    return problem2_1 + problem2_2
