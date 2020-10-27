def grade_test_cases(max_points, test_cases):
    passed_tests = 0
    failed_test_indices = []
    for idx, case in enumerate(test_cases):
        if case is True:
            passed_tests += 1
        else:
            failed_test_indices.append(idx + 1)

    total_tests = len(test_cases)
    points_per_test = max_points / total_tests
    total_points = points_per_test * passed_tests

    print(f"=> TA Test Cases: {total_points:.2f}/{max_points}, passed {passed_tests} out of {total_tests} test cases.")
    if len(failed_test_indices) > 0:
        print(f"Failed Test Cases: {failed_test_indices}. If you wish to know what exactly failed just email me these numbers and I'll send you the test cases you request.")
    return total_points


def grade(problem, description, max_points, max_test_case_points, max_time_pts=0, test_case_func=lambda: []):
    print(f'--==> Problem: {problem} ({max_points}pts)')
    print(f'--==> Description: {description}')

    gr = 0
    try:
        test_cases = test_case_func()
        gr = grade_test_cases(max_test_case_points, test_cases)
    except Exception as Ex:
        print(f"An exception was thrown by the student's lab: {repr(Ex)}. This will affect the final score.")
        # Ex.with_traceback()
        print(f"=> TA Test Cases: 0/{max_test_case_points}pts, passed 0 of the test cases due to an exception.")

    if max_time_pts > 0:
        correct_time_str = input("Is the student's time/space complexity correct? If not, what is the student's complexity?")
        added_pts = 0
        if correct_time_str.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
            added_pts += max_time_pts
        else:
            added_pts += (max_time_pts / 2)

        gr += added_pts
        print(f"=> Time & Space Complexity: {added_pts}/{max_time_pts}pts")

    print(f'=> Grade: {gr}/{max_points}pts')
    print()

    return gr
