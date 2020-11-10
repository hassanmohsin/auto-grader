from core import *


def grade_problem(lab4_source):
    # ======================================================= TEST-CASES
    def get_tests():
        # sum([x for x in [0.5, 1, 1.5, 2, 2.5, 3, 3.14, 3.15, 4, 4.28, 5, 6.28] if x>=3 and x<=4])
        from lab4_helper import AVLTree as TA_AVLTree
        tree1 = TA_AVLTree()
        tree1.insertList([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        test1 = grade_test_case_helper(test_number=1, source_func=lab4_source.range_sum, source_params=[tree1.getRoot(), -5, -1], expected_value=0, extra_info=f"Tree={str(tree1)}")
        test2 = grade_test_case_helper(test_number=2, source_func=lab4_source.range_sum, source_params=[tree1.getRoot(), 5, 10], expected_value=45, extra_info=f"Tree={str(tree1)}")
        test3 = grade_test_case_helper(test_number=3, source_func=lab4_source.range_sum, source_params=[tree1.getRoot(), 5, 9], expected_value=35, extra_info=f"Tree={str(tree1)}")
        test4 = grade_test_case_helper(test_number=4, source_func=lab4_source.range_sum, source_params=[tree1.getRoot(), 0, 3], expected_value=6, extra_info=f"Tree={str(tree1)}")
        test5 = grade_test_case_helper(test_number=5, source_func=lab4_source.range_sum, source_params=[tree1.getRoot(), -10, 100], expected_value=55, extra_info=f"Tree={str(tree1)}")
        test6 = grade_test_case_helper(test_number=6, source_func=lab4_source.range_sum, source_params=[tree1.getRoot(), -3.2, 3], expected_value=6, extra_info=f"Tree={str(tree1)}")

        tree2 = TA_AVLTree()
        tree2.insertList([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 10])
        test7 = grade_test_case_helper(test_number=7, source_func=lab4_source.range_sum, source_params=[tree2.getRoot(), -5, -1], expected_value=0, extra_info=f"Tree={str(tree2)}")
        test8 = grade_test_case_helper(test_number=8, source_func=lab4_source.range_sum, source_params=[tree2.getRoot(), 1, 2], expected_value=19, extra_info=f"Tree={str(tree2)}")
        test9 = grade_test_case_helper(test_number=9, source_func=lab4_source.range_sum, source_params=[tree2.getRoot(), 1, 10], expected_value=36, extra_info=f"Tree={str(tree2)}")
        test10 = grade_test_case_helper(test_number=10, source_func=lab4_source.range_sum, source_params=[tree2.getRoot(), 1, 1], expected_value=1, extra_info=f"Tree={str(tree2)}")

        tree3 = TA_AVLTree()
        tree3.insertList([-5, -4, -3, -2, -2, -1, 5, 10])
        test11 = grade_test_case_helper(test_number=11, source_func=lab4_source.range_sum, source_params=[tree3.getRoot(), 0, 10], expected_value=15, extra_info=f"Tree={str(tree3)}")
        test12 = grade_test_case_helper(test_number=12, source_func=lab4_source.range_sum, source_params=[tree3.getRoot(), -5, 10], expected_value=-2, extra_info=f"Tree={str(tree3)}")
        test13 = grade_test_case_helper(test_number=13, source_func=lab4_source.range_sum, source_params=[tree3.getRoot(), -5, -2], expected_value=-16, extra_info=f"Tree={str(tree3)}")

        tree4 = TA_AVLTree()
        tree4.insertList([0.5, 1, 1.5, 2, 2.5, 3, 3.14, 3.15, 4, 4.28, 5, 6.28])
        test14 = grade_test_case_helper(test_number=14, source_func=lab4_source.range_sum, source_params=[tree4.getRoot(), 0, 2], expected_value=5, extra_info=f"Tree={str(tree4)}")
        test15 = grade_test_case_helper(test_number=15, source_func=lab4_source.range_sum, source_params=[tree4.getRoot(), 3, 4], expected_value=13.29, extra_info=f"Tree={str(tree4)}")
        test16 = grade_test_case_helper(test_number=16, source_func=lab4_source.range_sum, source_params=[tree4.getRoot(), 3.1, 3.5], expected_value=6.29, extra_info=f"Tree={str(tree4)}")

        return [test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12, test13, test14, test15, test16]

    # ======================================================= AUTO-GRADING
    return auto_grade_test_cases(problem='Range Sum',
                                 description='The student just has to provide a solution to the problem, no time or space complexity requirements',
                                 max_points=30,
                                 max_test_case_points=30,
                                 test_case_gen_func=get_tests)
