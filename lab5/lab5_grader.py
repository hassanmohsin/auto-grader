import core
import lab5_problem1
import lab5_problem2

core.auto_grade_lab(student_code_dir=r'data', lab_str='Lab5', problem_dict={
    'Least Recently Used Cache': (50, lab5_problem1.grade_problem),
    'HeapSort Passwords': (50, lab5_problem2.grade_problem),
})