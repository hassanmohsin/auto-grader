import core
import lab3_problem1
import lab3_problem2
import lab3_problem3

core.auto_grade_lab(student_code_dir='data', lab_str='Lab3', problem_dict={
    'ArrayList': (40, lab3_problem1.grade_problem),
    'Circular Shift': (40, lab3_problem2.grade_problem),
    'One Edit Away': (20, lab3_problem3.grade_problem),
})