from core import *


def grade_problem(lab5_source):
    # ======================================================= TEST-CASES
    def get_and_put_tests():
        lru = lab5_source.LRUCache(5)
        lru.put("k1", 5)
        lru.put(20, -3.14)
        lru.put("k2", 0)

        test1 = grade_test_case_helper(test_number=1, source_func=lru.get, source_params=["k1"], expected_value=5, extra_info='Called put("k1", 5)')
        test2 = grade_test_case_helper(test_number=2, source_func=lru.get, source_params=[20], expected_value=-3.14, extra_info='Called put(20, -3.14)')
        test3 = grade_test_case_helper(test_number=3, source_func=lru.get, source_params=["k2"], expected_value=0, extra_info='Called put("k2", 0)')
        test4 = grade_test_case_helper(test_number=4, source_func=lru.get, source_params=["notvalidkey"], expected_value=None, extra_info='')

        # Test the key replace
        lru.put("k1", 17)
        test5 = grade_test_case_helper(test_number=5, source_func=lru.get, source_params=["k1"], expected_value=17, extra_info='Called put("k1", 17) to replace the key k1')

        # Test the LRU delete
        lru.put(20, 15)
        lru.put("k3", 15)
        lru.put("k4", 50)
        lru.put("k5", -5)
        lru.get(20)
        lru.get("k3")
        lru.get("k4")
        lru.get("k5")

        test6 = grade_test_case_helper(test_number=6, source_func=lru.get, source_params=["k2"], expected_value=None, extra_info='Updated all keys except k2, so k2 should be removed as the least recently accessed key')

        return [test1, test2, test3, test4, test5, test6]

    def size_tests():
        lru = lab5_source.LRUCache(5)
        test1 = grade_test_case_helper(test_number=1, source_func=lru.size, source_params=[], expected_value=0, extra_info='')
        lru.put("k1", 1)
        test2 = grade_test_case_helper(test_number=2, source_func=lru.size, source_params=[], expected_value=1, extra_info='')
        lru.put("k1", 10)
        test3 = grade_test_case_helper(test_number=3, source_func=lru.size, source_params=[], expected_value=1, extra_info='')
        lru.put("k2", 2)
        lru.put("k3", 3)
        lru.put("k4", 4)
        lru.put("k5", 5)
        lru.put("k6", 6)
        test4 = grade_test_case_helper(test_number=4, source_func=lru.size, source_params=[], expected_value=5, extra_info='')

        return [test1, test2, test3, test4]

    def max_capacity_tests():
        lru = lab5_source.LRUCache(5)
        test1 = grade_test_case_helper(test_number=1, source_func=lru.max_capacity, source_params=[], expected_value=5, extra_info='')
        lru = lab5_source.LRUCache(100)
        test2 = grade_test_case_helper(test_number=2, source_func=lru.max_capacity, source_params=[], expected_value=100, extra_info='')

        return [test1, test2]

    # ======================================================= TIME-COMPLEXITY
    def gen_test_lru_cache(n):
        n = int(n)

        lru = lab5_source.LRUCache(n)
        for i in range(n):
            lru.put("k" + str(i), i)
        return lru

    def measure_get(lru):
        lru.get(0)

    def measure_put(lru):
        lru.put("k" + str(lru.size()+5), 3.14)
        lru.put("k" + str(lru.size()+4), 3.14)

    def measure_size(lru):
        lru.size()

    def measure_max_capacity(lru):
        lru.max_capacity()

    # ======================================================= AUTO-GRADING

    problem1_1 = auto_grade_test_cases(problem='get & put',
                                       description='The student must correctly implement the given methods in O(1) time',
                                       max_points=12.5 + 12.5,
                                       max_test_case_points=6.25 + 6.25,
                                       test_case_gen_func=get_and_put_tests) + auto_grade_time_complexity(measure_func=measure_get,
                                                                                                  data_gen_func=gen_test_lru_cache,
                                                                                                  max_time_points=6.25,
                                                                                                  correct_time_type=big_o.complexities.Constant) + auto_grade_time_complexity(measure_func=measure_put,
                                                                                                                                                                              data_gen_func=gen_test_lru_cache,
                                                                                                                                                                              max_time_points=6.25,
                                                                                                                                                                              correct_time_type=big_o.complexities.Constant)

    print()
    problem1_2 = auto_grade_test_cases(problem='size',
                                       description='The student must correctly implement the given method in O(1) time',
                                       max_points=12.5,
                                       max_test_case_points=6.25,
                                       test_case_gen_func=size_tests) + auto_grade_time_complexity(measure_func=measure_size,
                                                                                                  data_gen_func=gen_test_lru_cache,
                                                                                                  max_time_points=6.25,
                                                                                                  correct_time_type=big_o.complexities.Constant)
    print()
    problem1_3 = auto_grade_test_cases(problem='max_capacity',
                                       description='The student must correctly implement the given method in O(1) time',
                                       max_points=12.5,
                                       max_test_case_points=6.25,
                                       test_case_gen_func=max_capacity_tests) + auto_grade_time_complexity(measure_func=measure_max_capacity,
                                                                                                  data_gen_func=gen_test_lru_cache,
                                                                                                  max_time_points=6.25,
                                                                                                  correct_time_type=big_o.complexities.Constant)
    print()
    return problem1_1 + problem1_2 + problem1_3
