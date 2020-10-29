from core import *


def grade_problem(lab4_source):
    # ======================================================= TEST-CASES
    def get_tests():
        return [True]

    # ======================================================= AUTO-GRADING

    return auto_grade_test_cases(problem='Average of Levels',
                                 description='The student just has to provide a solution to the problem, no time or space complexity requirements',
                                 max_points=30,
                                 max_test_case_points=30,
                                 test_case_gen_func=get_tests)
