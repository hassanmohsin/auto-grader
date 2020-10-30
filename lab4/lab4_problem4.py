from core import *


def grade_problem(lab4_source):
    # ======================================================= TEST-CASES
    def get_tests():
        from lab4_helper import AVLTree as TA_AVLTree
        tree1 = TA_AVLTree()
        tree1.insertList([3, 9, 20, 15, 7])
        test1 = grade_test_case_helper(test_number=1, source_func=lab4_source.average_of_levels, source_params=[tree1.getRoot()], expected_value=[9.0, 11.5, 11.0], extra_info=f"Tree={str(tree1)}")

        return [test1]

    # ======================================================= AUTO-GRADING

    return auto_grade_test_cases(problem='Average of Levels',
                                 description='The student just has to provide a solution to the problem, no time or space complexity requirements',
                                 max_points=30,
                                 max_test_case_points=30,
                                 test_case_gen_func=get_tests)
