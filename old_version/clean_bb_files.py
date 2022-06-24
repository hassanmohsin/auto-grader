import shutil
import subprocess
from pathlib import Path

# Given a directory containing all the Jupyter notebooks downloaded from Blackboard renames them and converts them to .py files
dir = Path(".")

# Rename the files to simple <student_email_name>.py
for fpath in dir.glob("*.ipynb"):
    username = fpath.stem.split("_")[1]
    print("Renaming [", fpath, "] to {", str(dir / username) + ".ipynb")
    fpath.rename(str(dir / username) + ".ipynb")

# Activate my personal Anaconda "dataStructs" Environment
# My IDE doesn't seem to find "activate" in the path so I put the full path
subprocess.call([r'C:\Users\xeroj\Anaconda3\Scripts\activate.bat', 'dataStructs'])
for fpath in dir.glob("*.ipynb"):
    subprocess.call(['jupyter', 'nbconvert', '--to', 'script', str(fpath)])

# Rename all the recently converted files from .txt to .py
# It seems to make new copies of the files, just delete the .ipynb files manually
for fpath in dir.glob("*.txt"):
    fpath.rename(str(dir / fpath.stem) + ".py")
