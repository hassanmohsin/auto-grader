from core import *


def grade_problem(lab5_source):
    # ======================================================= TEST-CASES
    def get_tests():
        return [True]

    def put_tests():
        return [True]

    def size_tests():
        return [True]

    def max_capacity_tests():
        return [True]

    # ======================================================= TIME-COMPLEXITY
    def gen_test_lru_cache(n):
        n = int(n)

        lru = lab5_source.LRUCache(n)
        return lru

    def measure_get(lru):
        lru.get(0)

    def measure_put(lru):
        pass

    def measure_size(lru):
        pass

    def measure_max_capacity(lru):
        pass

    # ======================================================= AUTO-GRADING

    problem1_1 = auto_grade_test_cases(problem='get',
                                 description='The student must correctly implement the given method in O(1) time',
                                 max_points=12.5,
                                 max_test_case_points=6.25,
                                 test_case_gen_func=get_tests) + auto_grade_time_complexity(measure_func=measure_get,
                                                                                                       data_gen_func=gen_test_lru_cache,
                                                                                                       max_time_points=6.25,
                                                                                                       correct_time_type=big_o.complexities.Constant)
    problem1_2 = auto_grade_test_cases(problem='put',
                                 description='The student must correctly implement the given method in O(1) time',
                                 max_points=12.5,
                                 max_test_case_points=6.25,
                                 test_case_gen_func=get_tests) + auto_grade_time_complexity(measure_func=measure_put,
                                                                                                       data_gen_func=gen_test_lru_cache,
                                                                                                       max_time_points=6.25,
                                                                                                       correct_time_type=big_o.complexities.Constant)
    problem1_3 = auto_grade_test_cases(problem='size',
                                 description='The student must correctly implement the given method in O(1) time',
                                 max_points=12.5,
                                 max_test_case_points=6.25,
                                 test_case_gen_func=get_tests) + auto_grade_time_complexity(measure_func=measure_size,
                                                                                                       data_gen_func=gen_test_lru_cache,
                                                                                                       max_time_points=6.25,
                                                                                                       correct_time_type=big_o.complexities.Constant)
    problem1_4 = auto_grade_test_cases(problem='max_capacity',
                                 description='The student must correctly implement the given method in O(1) time',
                                 max_points=12.5,
                                 max_test_case_points=6.25,
                                 test_case_gen_func=get_tests) + auto_grade_time_complexity(measure_func=measure_max_capacity,
                                                                                                       data_gen_func=gen_test_lru_cache,
                                                                                                       max_time_points=6.25,
                                                                                                       correct_time_type=big_o.complexities.Constant)

    return problem1_1_ + problem1_2 + problem1_3 + problem1_4
