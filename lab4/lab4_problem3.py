from core import *


def grade_problem(lab4_source):
    # ======================================================= TEST-CASES
    def get_tests():
        from lab4_helper import AVLTree as TA_AVLTree
        tree1 = TA_AVLTree()
        tree1.insertList([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        test1 = grade_test_case_helper(test_number=1, source_func=lab4_source.is_univalued, source_params=[tree1.getRoot()], expected_value=False, extra_info=f"Tree={str(tree1)}")

        tree2 = TA_AVLTree()
        tree2.insertList([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        test2 = grade_test_case_helper(test_number=2, source_func=lab4_source.is_univalued, source_params=[tree2.getRoot()], expected_value=True, extra_info=f"Tree={str(tree2)}")

        tree3 = TA_AVLTree()
        tree3.insertList([3])
        test3 = grade_test_case_helper(test_number=3, source_func=lab4_source.is_univalued, source_params=[tree3.getRoot()], expected_value=True, extra_info=f"Tree={str(tree3)}")

        tree4 = TA_AVLTree()
        tree4.insertList([-1, -1, -1])
        test4 = grade_test_case_helper(test_number=4, source_func=lab4_source.is_univalued, source_params=[tree4.getRoot()], expected_value=True, extra_info=f"Tree={str(tree4)}")

        tree5 = TA_AVLTree()
        tree5.insertList([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3.1])
        test5 = grade_test_case_helper(test_number=5, source_func=lab4_source.is_univalued, source_params=[tree5.getRoot()], expected_value=False, extra_info=f"Tree={str(tree5)}")

        return [test1, test2, test3, test4, test5]

    # ======================================================= AUTO-GRADING

    return auto_grade_test_cases(problem='Univalued Tree',
                                 description='The student just has to provide a solution to the problem, no time or space complexity requirements',
                                 max_points=30,
                                 max_test_case_points=30,
                                 test_case_gen_func=get_tests)
