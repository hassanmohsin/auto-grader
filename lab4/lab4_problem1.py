import inspect

from core import *

lookup_table = ['avl', 'red', 'black', 'sg', 'scapegoat', 'aa', 'splay', 'treap', 'redblack']


def grade_problem(lab4_source):
    # ======================================================= TEST-CASES
    def get_tests():
        # Get all the classes inside the student's code
        classes_list = inspect.getmembers(lab4_source, inspect.isclass)

        for class_name, class_obj in classes_list:
            # If we find the name of a self-balancing tree, assume it works
            for possible_str in lookup_table:
                if possible_str in class_name.lower():
                    print(f"=> Found a self-balancing BST class named {class_name}")
                    return [True]

            # If this class has 'tree'/'bst'/'node' anywhere in its name, it's probably a regular BST
            # Check if it has a balance function
            if 'tree' in class_name.lower() or 'bst' in class_name.lower() or 'node' in class_name.lower():
                print(f"Found a potential self-balancing binary search tree class named '{class_name}', looking for a balance method")
                functions_list = inspect.getmembers(class_obj, inspect.isfunction)
                for function_name, function_obj in functions_list:
                    if 'balance' in function_name.lower():
                        print(f"=> Found a balance method named '{function_name}' therefore the class is probably a self-balancing BST")
                        return [True]

                print(f"Did not find a balance method inside the class")

        print(f"=> Did not find a self-balancing binary search tree inside the module")
        return [False]

    # ======================================================= AUTO-GRADING

    return auto_grade_test_cases(problem='Self-Balancing Binary Search Tree',
                                 description='The student must implement a self-balancing BST',
                                 max_points=10,
                                 max_test_case_points=10,
                                 test_case_gen_func=get_tests)
