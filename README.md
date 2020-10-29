# **Data Structures Lab Auto-Grader**
Automatically grades labs of the Fall 2020 semester of CS2302 - Data Structures with Dr. Aguirre. I do not expect it to be perfect, but it should save a considerable amount of grading time for the TAs.

It is structured to automatically grade problems based on 3 things:
1. Test cases
2. Running time complexity
3. Whether the space complexity is constant or not

I give a brief description on how each of these is graded automatically at the bottom of this README.

## Requirements
* Python 3.8.5 or higher
* pip install jupyter, big_o

## How To Use
0. Clone this repository from Github or download it
1. Go to **Blackboard's Full Grade Center** for the course
2. In the column for the lab source code, click the arrow to see the column options
3. Select **"Assignment File Download"** to download all the assigments in a zipfile. Extract them and keep only the .ipynb files
4. **Run "clean_bb_files.py"** in the same directory that you have the .ipynb files to generate .py files
5. **Place the student .py files** inside the correct directory. As an example, for lab 3 that would be inside **\lab3\data\** where \lab3\ is the folder in this repository containing the "lab3_grader.py" file. Make the folder if it doesn't exist
6. **Run the grader** which is inside the lab specific folder. For lab 3, that would be **\lab3\lab3_grader.py**. When it finishes, the Python program will let you know it's finished grading

## What Does The Auto-Grader Generate?
* A **.csv** file containing every student's username and grades for every problem in the lab. It contains no headers, so I would recommend just copying it over to your own gradebook
* A **.bbtxt** file for every student. This is just a regular text-file which was renamed. This is the output of the Auto-Grader which should be given back to the student as this is the program's feedback telling them what exactly they got wrong

## Auto-Grading Test Cases
The idea is pretty simple. Import the student's code and call their functions and classes with my own test cases. Then based on how many points the test cases are worth give them points based on how many test cases their code passed. The main functionality is in **core.py** as
> auto_grade_test_cases(...)

## Auto-Grading Running Time Complexity
We determine the running time complexity of a function by running it and timing it for different size inputs using the **big_o** Python package. This is not a perfect solution, but it seems to work well enough for the purposes of this class. The main functionality is in **core.py** as
> auto_grade_time_complexity(...)

## Auto-Grading Constant Space Complexity
We get the **locals()** table from inside the student's function using code from (https://github.com/pberkes/persistent_locals). Since we have all the variables in the local scope, we can check their types and see if they are lists. Thus, we can count how many lists are in the local scope of the function. If the number of lists in the local scope does not match the number of lists in the input, the function is considered to use extra space since most of the time we expect those solutions to modify lists in place and not create new ones. The main functionality is in **core.py** as
> auto_grade_constant_space(...)