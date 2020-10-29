from core import *


def grade_problem(lab4_source):
    # ======================================================= TEST-CASES
    def range_sum(root, min_val, max_val):
        try:
            return lab4_source.range_sum(root, min_val, max_val)
        except Exception as ex:
            print(f"An exception was thrown by range_sum(root, {min_val}, {max_val}): {str(ex)}")
            return False

    def get_tests():
        from lab4_helper import AVLTree as TA_AVLTree
        tree1 = TA_AVLTree()
        tree1.insertList([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        test1 = range_sum(tree1.getRoot(), min_val=-5, max_val=-1) == 0
        test2 = range_sum(tree1.getRoot(), min_val=5, max_val=10) == 45

        tree2 = TA_AVLTree()
        tree2.insertList([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 10])

        tree3 = TA_AVLTree()
        tree3.insertList([-5, -4, -3, -2, -2, -1, 5, 10])
        return [True]

    # ======================================================= AUTO-GRADING
    return auto_grade_test_cases(problem='Range Sum',
                                 description='The student just has to provide a solution to the problem, no time or space complexity requirements',
                                 max_points=30,
                                 max_test_case_points=30,
                                 test_case_gen_func=get_tests)
