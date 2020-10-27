from core import *


def grade_problem(lab3_source):
    # ======================================================= TEST-CASES
    def test_one_edit_away():
        return [True]

    # ======================================================= AUTO-GRADING
    problem3_1 = grade(problem='One Edit Away',
                       description='The student just has to provide a solution to the problem, no time or space complexity requirements',
                       max_points=20,
                       max_test_case_points=20,
                       max_time_pts=0,
                       test_case_func=test_one_edit_away)

    return problem3_1
