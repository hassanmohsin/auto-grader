from core import *


def grade_problem(lab5_source):
    # ======================================================= TEST-CASES
    def heap_test():
        return [True]

    def heap_sort_tests():
        return [True]

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