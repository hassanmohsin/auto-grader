import core
import lab4_problem1
import lab4_problem2
import lab4_problem3
import lab4_problem4

core.auto_grade_lab(student_code_dir=r'data', lab_str='Lab4', problem_dict={
    'Self-Balancing Binary Search Tree': (10, lab4_problem1.grade_problem),
    'Range Sum': (30, lab4_problem2.grade_problem),
    'Univalued Tree': (30, lab4_problem3.grade_problem),
    'Average of Levels': (30, lab4_problem4.grade_problem),
})
