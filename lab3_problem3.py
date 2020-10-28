from core import *


def grade_problem(lab3_source):
    # ======================================================= TEST-CASES
    def test_one_edit_away():
        test1 = lab3_source.one_edit_away('AAA', 'AA')
        test2 = lab3_source.one_edit_away('ABA', 'ACA')
        test3 = lab3_source.one_edit_away('ABA', 'AA')
        test4 = lab3_source.one_edit_away('pale', 'ale')
        test5 = lab3_source.one_edit_away('pale', 'ple')
        test6 = lab3_source.one_edit_away('pales', 'pale')
        test7 = lab3_source.one_edit_away('pale', 'bale')

        test8 = not lab3_source.one_edit_away('pale', 'bae')
        test9 = not lab3_source.one_edit_away('AA', 'AAAA')
        test10 = not lab3_source.one_edit_away('AB', 'BC')
        test11 = not lab3_source.one_edit_away('ABAB', 'BBAA')
        test12 = not lab3_source.one_edit_away('', 'AB')

        test13 = lab3_source.one_edit_away('', 'A')
        test14 = lab3_source.one_edit_away('A', '')

        test15 = lab3_source.one_edit_away('', '')
        test16 = lab3_source.one_edit_away('ABC', 'ABC')
        test17 = lab3_source.one_edit_away('123', '123')

        test18 = lab3_source.one_edit_away('12', '2')
        test19 = lab3_source.one_edit_away('123', '13')
        test20 = lab3_source.one_edit_away('12', '123')
        return [test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12, test13, test14, test15, test16, test17, test18, test19, test20]

    # ======================================================= AUTO-GRADING
    problem3_1 = auto_grade_test_cases(problem='One Edit Away',
                                       description='The student just has to provide a solution to the problem, no time or space complexity requirements',
                                       max_points=20,
                                       max_test_case_points=20,
                                       test_case_gen_func=test_one_edit_away)
    print(f"==> {problem3_1}/20pts")
    print()

    return problem3_1
