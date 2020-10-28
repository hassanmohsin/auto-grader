from pathlib import Path

import lab3_problem1
import lab3_problem2
import lab3_problem3

for fpath in Path("data/lab3/").glob("*.py"):
    # Import and replace the current lab 3 source
    lab3_source = None
    import_exception = False
    module_path = str(fpath).replace('\\', '.').replace('.py', '')
    try:
        module = __import__(module_path)
        exec(f"from data.lab3 import {fpath.stem} as lab3_source")
    except Exception as ex:
        import_exception = ex

    print("")
    print(">> AutoGrader v1.0 by Jose G. Perez (TA)")
    print(">> Assignment: Lab 3")
    print(">> Student: ", fpath.stem)

    if import_exception:
        print(">> There was an exception thrown by a cell while trying to import. Run all your cells next time to make sure no exceptions are thrown.")
        print(">> Exception: ", repr(import_exception))

    print("------------------------- [Problem 1] ArrayList (40pts)")
    problem1_total = lab3_problem1.grade_problem(lab3_source)
    print()

    print("------------------------- [Problem 2] Circular Shift (40pts)")
    problem2_total = lab3_problem2.grade_problem(lab3_source)
    print()

    print("------------------------- [Problem 3] One Edit Away (20pts)")
    problem3_total = lab3_problem3.grade_problem(lab3_source)
    print()

    print("------------------------- Overall Grades")
    print(f"--==> Problem 1={problem1_total}/40pts")
    print(f"--==> Problem 2={problem2_total}/40pts")
    print(f"--==> Problem 3={problem3_total}/20pts")
    print(f"--==> Total={problem1_total + problem2_total + problem3_total}/100pts")

    input("> Waiting to go to next student")
