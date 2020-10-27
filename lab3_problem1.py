from core import *


def grade_problem(lab3_source):
    # ======================================================= TEST CASES
    def get_item_tests():
        arr = lab3_source.ArrayList(5)
        arr_internal = lab3_source.Array(10)
        arr_internal.data = [1, 2, 3, 4, 5]
        arr.data = arr_internal
        arr.curr_size = 5

        test1 = (arr[0] == 1) and (arr[4] == 5)
        test2 = (arr[1] == 2) and (arr[2] == 3)
        test3 = False
        try:
            x = arr[-1]
        except IndexError:
            test3 = True

        test4 = False
        try:
            x = arr[10]
        except IndexError:
            test4 = True

        return [test1, test2, test3, test4]

    def set_item_tests():
        arr = lab3_source.ArrayList(10)
        arr_internal = lab3_source.Array(10)
        arr_internal.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        arr.curr_size = 10

        arr[0] = 1738
        test1 = arr.data[0] == 1738

        arr[3] = -5
        test2 = arr.data[3] == -5

        arr[9] = 692
        test3 = arr.data[9] == 692

        test4 = False
        try:
            arr[-1] = 5
        except IndexError:
            test4 = True

        test5 = False
        try:
            arr[15] = 5
        except IndexError:
            test5 = True

        return [test1, test2, test3, test4, test5]

    def append_tests():
        arr = lab3_source.ArrayList(10)
        for i in range(9):
            arr.append(i)

        test1 = True
        for i in range(9):
            if arr.data[i] != i:
                test1 = False

        test2 = arr.max_size == 10 and arr.curr_size == 9

        arr.append(-13)
        arr.append(-13)
        test3 = arr.data[10] == -13

        test4 = arr.max_size > 10
        return [test1, test2, test3, test4]

    def prepend_tests():
        arr = lab3_source.ArrayList(10)
        arr_internal = lab3_source.Array(10)
        arr_internal.data = [1, 2, 3, 4, 5, 6, 7, 8, None, None]
        arr.curr_size = 8

        arr.prepend(1738)
        test1 = arr[0] == 1738

        arr.prepend(1739)
        test2 = arr[0] == 1739

        arr.prepend(-5)
        test3 = arr[0] == -5

        test4 = arr.max_size > 10

        return [test1, test2, test3, test4]

    def insert_tests():
        arr = lab3_source.ArrayList(10)
        for i in range(8):
            arr.append(i)

        arr.insert(0, -5)
        test1 = arr[0] == -5
        test2 = arr.curr_size == 9

        arr.insert(8, -1738)
        test3 = arr[8] == -1738

        arr.insert(5, -5)
        arr.insert(5, -10)
        arr.insert(5, -15)
        test4 = arr.max_size > 10

        test5 = False
        try:
            arr.insert(-1, 5)
        except IndexError:
            test5 = True

        test6 = False
        try:
            arr.insert(1500, 5)
        except IndexError:
            test6 = True

        return [test1, test2, test3, test4, test5, test6]

    def remove_tests():
        arr = lab3_source.ArrayList(10)
        arr.append(1)
        arr.append(2)
        arr.append(5)
        arr.append(4)
        arr.append(5)
        arr.append(5)
        arr.append(7)
        arr.append(8)

        arr.remove(1)
        test1 = arr[0] == 2 and arr.curr_size == 7

        arr.remove(8)
        test2 = arr[arr.curr_size - 1] == 7 and arr.curr_size == 6

        arr.remove(4)
        test3 = 4 not in arr.data.data and arr.curr_size == 5

        arr.remove(5)
        test4 = arr[2] == 5 and arr.curr_size == 4

        test5 = False
        try:
            arr.remove(1738)
        except ValueError:
            test5 = True

        return [test1, test2, test3, test4, test5]

    def delete_tests():
        arr = lab3_source.ArrayList(10)
        arr.append(0)
        arr.append(1)
        arr.append(2)
        arr.append(3)
        arr.append(4)
        arr.append(5)
        arr.append(6)
        arr.append(7)

        arr.delete(0)
        test1 = arr[0] == 1 and arr.curr_size == 7

        arr.delete(7)
        test2 = arr[arr.curr_size - 1] == 6 and arr.curr_size == 6

        arr.delete(5)
        test3 = arr.curr_size == 5

        test4 = False
        try:
            arr.delete(-1)
        except IndexError:
            test4 = True

        test5 = False
        try:
            arr.delete(1000)
        except IndexError:
            test5 = True

        return [test1, test2, test3, test4, test5]

    def contains_tests():
        arr = lab3_source.ArrayList(10)
        for i in range(8):
            arr.append(i)

        test1 = 0 in arr
        test2 = 5 in arr
        test3 = not (-100 in arr)
        test4 = not (30 in arr)
        return [test1, test2, test3, test4]

    # ======================================================= AUTO-GRADING
    problem1_1 = grade(problem='Using Array class internally',
                       description='The student must use the provided Array class for internal data representation and not just Python lists',
                       max_points=4,
                       max_test_case_points=4,
                       max_time_pts=0,
                       test_case_func=lambda: [type(lab3_source.ArrayList().data) is lab3_source.Array])

    problem1_2 = grade(problem='__getitem__',
                       description='The student must correctly implement the given method in O(1) time',
                       max_points=4,
                       max_test_case_points=2,
                       max_time_pts=0,
                       test_case_func=get_item_tests)

    problem1_3 = grade(problem='__setitem__',
                       description='The student must correctly implement the given method in O(1) time',
                       max_points=4,
                       max_test_case_points=2,
                       max_time_pts=0,
                       test_case_func=set_item_tests)

    problem1_4 = grade(problem='__len__',
                       description='The given __len__ method was complete, the student is given full-points automatically',
                       max_points=4,
                       max_test_case_points=4,
                       max_time_pts=0,
                       test_case_func=lambda: [True])

    problem1_5 = grade(problem='append',
                       description='The student must correctly implement the given method in the given time and space complexity constraints',
                       max_points=4,
                       max_test_case_points=4,
                       max_time_pts=0,
                       test_case_func=append_tests)

    problem1_6 = grade(problem='prepend',
                       description='The student must correctly implement the given method in the given time and space complexity constraints',
                       max_points=4,
                       max_test_case_points=4,
                       max_time_pts=0,
                       test_case_func=prepend_tests)

    problem1_7 = grade(problem='insert',
                       description='The student must correctly implement the given method in the given time and space complexity constraints',
                       max_points=4,
                       max_test_case_points=4,
                       max_time_pts=0,
                       test_case_func=insert_tests)

    problem1_8 = grade(problem='remove',
                       description='The student must correctly implement the given method in the given time and space complexity constraints',
                       max_points=4,
                       max_test_case_points=4,
                       max_time_pts=0,
                       test_case_func=remove_tests)

    problem1_9 = grade(problem='delete',
                       description='The student must correctly implement the given method in the given time and space complexity constraints',
                       max_points=4,
                       max_test_case_points=4,
                       max_time_pts=0,
                       test_case_func=delete_tests)

    problem1_10 = grade(problem='__contains__',
                        description='The student must correctly implement the given method in the given time and space complexity constraints',
                        max_points=4,
                        max_test_case_points=4,
                        max_time_pts=0,
                        test_case_func=contains_tests)

    return problem1_1 + problem1_2 + problem1_3 + problem1_4 + problem1_5 + problem1_6 + problem1_7 + problem1_8 + problem1_9 + problem1_10
