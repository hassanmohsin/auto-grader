import numpy as np

from core import *


def grade_problem(lab3_source):
    # ======================================================= TEST-CASES
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

    # ======================================================= TIME-COMPLEXITY
    def gen_test_array_list(n: np.int64):
        n = int(n)

        arr = lab3_source.ArrayList(n)
        for i in range(n - 2):
            arr.append(i)
        return arr

    def measure_get_item(lst):
        lst.__getitem__(0)

    def measure_set_item(lst):
        lst[0] = 5

    def measure_append_constant(lst):
        lst.append(5)

    def measure_append_linear(lst):
        lst.append(15)
        lst.append(20)
        lst.append(20)

    def measure_prepend_constant(lst):
        lst.prepend(0)

    def measure_prepend_linear(lst):
        lst.prepend(0)
        lst.prepend(5)
        lst.prepend(10)

    def measure_insert_constant(lst):
        lst.insert(0, -5)
        lst.insert(lst.curr_size - 1, -5)

    def measure_insert_linear(lst):
        lst.insert(1, 5)

    def measure_remove_constant(lst):
        lst.remove(lst[0])

    def measure_remove_linear(lst):
        lst.remove(lst[lst.curr_size - 1])

    def measure_delete_constant(lst):
        lst.delete(0)
        lst.delete(lst.curr_size - 1)

    def measure_delete_linear(lst):
        lst.delete(int(lst.curr_size / 2))

    def measure_contains(lst):
        var = lst[lst.curr_size - 1] in lst

    # ======================================================= AUTO-GRADING
    problem1_1 = auto_grade_test_cases(problem='Using Array class internally',
                                       description='The student must use the provided Array class for internal data representation and not just Python lists',
                                       max_points=4,
                                       max_test_case_points=4,
                                       test_case_gen_func=lambda: [type(lab3_source.ArrayList().data) is lab3_source.Array])
    print(f"==> {problem1_1}/4pts")
    print()

    problem1_2 = auto_grade_test_cases(problem='__getitem__',
                                       description='The student must correctly implement the given method in O(1) time',
                                       max_points=4,
                                       max_test_case_points=2,
                                       test_case_gen_func=get_item_tests) + auto_grade_time_complexity(measure_func=measure_get_item,
                                                                                                       data_gen_func=gen_test_array_list,
                                                                                                       max_time_points=2,
                                                                                                       correct_time_type=big_o.complexities.Constant)
    print(f"==> {problem1_2}/4pts")
    print()

    problem1_3 = auto_grade_test_cases(problem='__setitem__',
                                       description='The student must correctly implement the given method in O(1) time',
                                       max_points=4,
                                       max_test_case_points=2,
                                       test_case_gen_func=set_item_tests) + auto_grade_time_complexity(measure_func=measure_set_item,
                                                                                                       data_gen_func=gen_test_array_list,
                                                                                                       max_time_points=2,
                                                                                                       correct_time_type=big_o.complexities.Constant)
    print(f"==> {problem1_3}/4pts")
    print()

    problem1_4 = auto_grade_test_cases(problem='__len__',
                                       description='The given __len__ method was already perfect, every student is given full-points automatically',
                                       max_points=4,
                                       max_test_case_points=4,
                                       test_case_gen_func=lambda: [True])
    print(f"==> {problem1_4}/4pts")
    print()

    problem1_5 = auto_grade_test_cases(problem='append',
                                       description='The student must correctly implement the given method in the given time and space complexity constraints',
                                       max_points=4,
                                       max_test_case_points=2,
                                       test_case_gen_func=append_tests) + auto_grade_time_complexity(measure_func=measure_append_constant,
                                                                                                     data_gen_func=gen_test_array_list,
                                                                                                     max_time_points=1,
                                                                                                     correct_time_type=big_o.complexities.Constant) + auto_grade_time_complexity(measure_func=measure_append_linear,
                                                                                                                                                                                 data_gen_func=gen_test_array_list,
                                                                                                                                                                                 max_time_points=1,
                                                                                                                                                                                 correct_time_type=big_o.complexities.Linear)
    print(f"==> {problem1_5}/4pts")
    print()

    problem1_6 = auto_grade_test_cases(problem='prepend',
                                       description='The student must correctly implement the given method in the given time and space complexity constraints',
                                       max_points=4,
                                       max_test_case_points=2,
                                       test_case_gen_func=prepend_tests) + auto_grade_time_complexity(measure_func=measure_prepend_constant,
                                                                                                      data_gen_func=gen_test_array_list,
                                                                                                      max_time_points=1,
                                                                                                      correct_time_type=big_o.complexities.Constant) + auto_grade_time_complexity(measure_func=measure_prepend_linear,
                                                                                                                                                                                  data_gen_func=gen_test_array_list,
                                                                                                                                                                                  max_time_points=1,
                                                                                                                                                                                  correct_time_type=big_o.complexities.Linear)
    print(f"==> {problem1_6}/4pts")
    problem1_7 = auto_grade_test_cases(problem='insert',
                                       description='The student must correctly implement the given method in the given time and space complexity constraints',
                                       max_points=4,
                                       max_test_case_points=2,
                                       test_case_gen_func=insert_tests) + auto_grade_time_complexity(measure_func=measure_insert_constant,
                                                                                                     data_gen_func=gen_test_array_list,
                                                                                                     max_time_points=1,
                                                                                                     correct_time_type=big_o.complexities.Constant) + auto_grade_time_complexity(measure_func=measure_insert_linear,
                                                                                                                                                                                 data_gen_func=gen_test_array_list,
                                                                                                                                                                                 max_time_points=1,
                                                                                                                                                                                 correct_time_type=big_o.complexities.Linear)
    print(f"==> {problem1_7}/4pts")
    print()

    problem1_8 = auto_grade_test_cases(problem='remove',
                                       description='The student must correctly implement the given method in the given time and space complexity constraints',
                                       max_points=4,
                                       max_test_case_points=2,
                                       test_case_gen_func=remove_tests) + auto_grade_time_complexity(measure_func=measure_remove_constant,
                                                                                                     data_gen_func=gen_test_array_list,
                                                                                                     max_time_points=1,
                                                                                                     correct_time_type=big_o.complexities.Constant) + auto_grade_time_complexity(measure_func=measure_remove_linear,
                                                                                                                                                                                 data_gen_func=gen_test_array_list,
                                                                                                                                                                                 max_time_points=1,
                                                                                                                                                                                 correct_time_type=big_o.complexities.Linear)
    print(f"==> {problem1_8}/4pts")
    print()

    problem1_9 = auto_grade_test_cases(problem='delete',
                                       description='The student must correctly implement the given method in the given time and space complexity constraints',
                                       max_points=4,
                                       max_test_case_points=2,
                                       test_case_gen_func=delete_tests) + auto_grade_time_complexity(measure_func=measure_delete_constant,
                                                                                                     data_gen_func=gen_test_array_list,
                                                                                                     max_time_points=1,
                                                                                                     correct_time_type=big_o.complexities.Constant) + auto_grade_time_complexity(measure_func=measure_delete_linear,
                                                                                                                                                                                 data_gen_func=gen_test_array_list,
                                                                                                                                                                                 max_time_points=1,
                                                                                                                                                                                 correct_time_type=big_o.complexities.Linear)
    print(f"==> {problem1_9}/4pts")
    print()

    problem1_10 = auto_grade_test_cases(problem='__contains__',
                                        description='The student must correctly implement the given method in the given time and space complexity constraints',
                                        max_points=4,
                                        max_test_case_points=2,
                                        test_case_gen_func=contains_tests) + auto_grade_time_complexity(measure_func=measure_contains,
                                                                                                        data_gen_func=gen_test_array_list,
                                                                                                        max_time_points=2,
                                                                                                        correct_time_type=big_o.complexities.Linear)
    print(f"==> {problem1_10}/4pts")
    print()
    return problem1_1 + problem1_2 + problem1_3 + problem1_4 + problem1_5 + problem1_6 + problem1_7 + problem1_8 + problem1_9 + problem1_10
