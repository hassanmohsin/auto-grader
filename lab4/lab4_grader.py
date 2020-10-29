import core
import lab4_problem1
import lab4_problem2

core.auto_grade_lab(student_code_dir=r'data', lab_str='Lab4', problem_dict={
    'Self-Balancing Binary Search Tree': (10, lab4_problem1.grade_problem),
    'Range Sum': (30, lab4_problem2.grade_problem),
})
