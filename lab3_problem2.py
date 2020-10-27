from core import *


def grade_problem(lab3_source):
    # ======================================================= TEST CASES
    def circular_shift_1(nums, k):
        try:
            return lab3_source.circular_shift_1(nums, k)
        except Exception as Ex:
            print("")
            return False

    def circular_shift_2(nums, k):
        try:
            return lab3_source.circular_shift_2(nums, k)
        except:
            return False

    def test_circular_shift_1():
        test1 = circular_shift_1([5, 3, 1, 7, 9], 2) == [7, 9, 5, 3, 1]
        test2 = circular_shift_1([5, 3, 1, 7, 9], 7) == [7, 9, 5, 3, 1]
        test3 = circular_shift_1([5, 3, 1, 7, 9], 12) == [7, 9, 5, 3, 1]

        test4 = circular_shift_1([5, 3, 1, 7, 9], 0) == [5, 3, 1, 7, 9]
        test5 = circular_shift_1([5, 3, 1, 7, 9], 5) == [5, 3, 1, 7, 9]

        test6 = circular_shift_1([5, 3, 1, 7, 9], 1) == [9, 5, 3, 1, 7]
        test7 = circular_shift_1([5, 3, 1, 7, 9], 3) == [1, 7, 9, 5, 3]
        test8 = circular_shift_1([5, 3, 1, 7, 9], 4) == [3, 1, 7, 9, 5]
        test9 = circular_shift_1([5, 3, 1, 7, 9], 6) == [9, 5, 3, 1, 7]
        test10 = circular_shift_1([5, 3, 1, 7, 9], 8) == [1, 7, 9, 5, 3]
        return [test1, test2, test4, test5, test6, test7, test8, test9, test10]

    def test_circular_shift_2():
        test1 = circular_shift_2([5, 3, 1, 7, 9], 2) == [7, 9, 5, 3, 1]
        test2 = circular_shift_2([5, 3, 1, 7, 9], 7) == [7, 9, 5, 3, 1]
        test3 = circular_shift_2([5, 3, 1, 7, 9], 12) == [7, 9, 5, 3, 1]

        test4 = circular_shift_2([5, 3, 1, 7, 9], 0) == [5, 3, 1, 7, 9]
        test5 = circular_shift_2([5, 3, 1, 7, 9], 5) == [5, 3, 1, 7, 9]

        test6 = circular_shift_2([5, 3, 1, 7, 9], 1) == [9, 5, 3, 1, 7]
        test7 = circular_shift_2([5, 3, 1, 7, 9], 3) == [1, 7, 9, 5, 3]
        test8 = circular_shift_2([5, 3, 1, 7, 9], 4) == [3, 1, 7, 9, 5]
        test9 = circular_shift_2([5, 3, 1, 7, 9], 6) == [9, 5, 3, 1, 7]
        test10 = circular_shift_2([5, 3, 1, 7, 9], 8) == [1, 7, 9, 5, 3]
        return [test1, test2, test3, test4, test5, test6, test7, test8, test9, test10]

    # ======================================================= AUTO-GRADING
    problem2_1 = grade(problem='Circular Shift 1',
                       description='The student must solve the problem. No time or space complexity requirements',
                       max_points=10,
                       max_test_case_points=10,
                       max_time_pts=0,
                       test_case_func=test_circular_shift_1)

    problem2_2 = grade(problem='Circular Shift 2',
                       description='The student must solve the problem in linear time and constant space',
                       max_points=30,
                       max_test_case_points=15,
                       max_time_pts=0,
                       test_case_func=test_circular_shift_2)

    return problem2_1 + problem2_2
