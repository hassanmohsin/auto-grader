from io import TextIOWrapper


def parse_py_file(input_fpath):
    """Parses .py files and extracts only classes and functions.

    Does not keep comments or empty lines.
        * This allows us to add "pass" to any empty code blocks easily
        * You can always look at the original files if you want to read the comments

    Limitations:
        * Does not currently support classes within classes (could be implemented in parse_class() function)
    """
    output_lines = []
    with open(input_fpath, 'r') as file:
        lines = file.readlines()
        while len(lines) > 0:
            line = lines.pop(0)

            # Stop if we reach the end of the file
            if not line:
                break

            # Allow top-level imports
            if line.find('import') == 0:
                output_lines.append(line)

            # Top-Level Classes
            if line.find('class') == 0:
                output_lines.append(line)

                class_spaces = __count_indentation__(line)
                output_lines.extend(__parse_class__(lines, class_spaces))

            # Top-Level Functions
            if line.find('def') == 0:
                output_lines.append(line)

                fn_spaces = __count_indentation__(line)
                output_lines.extend(__parse_function__(lines, fn_spaces))

    with open(input_fpath, 'w') as file:
        file.writelines(output_lines)


def __is_comment_or_empty__(line: str):
    """Checks if a line is a comment or empty regardless of indentation."""
    line = line.strip()  # Remove leading & trailing whitespaces from a line, ignoring indentation
    return len(line) == 0 or line[0] == '#' or line[0] == '\n'


def __count_indentation__(line: str):
    """Counts the number of leading spaces in a line."""
    return len(line) - len(line.lstrip(' '))


def __parse_class__(lines: list, class_spaces: int):
    output_lines = []

    while len(lines) > 0:
        line = lines.pop(0)
        n_spaces = __count_indentation__(line)

        # Look for functions
        if line.find('def', n_spaces) != -1:
            output_lines.append(line)

            # Find the indentation of the "def" line
            output_lines.extend(__parse_function__(lines, n_spaces))
        else:
            # It's not a function so check if we un-indented, if so, rewind file
            n_spaces = __count_indentation__(line)
            if n_spaces <= class_spaces:
                lines.insert(0, line)
                break

    # Empty class without code so add "pass" statement
    if len(output_lines) == 0:
        indentation = ' ' * (class_spaces+4)
        output_lines.append(f'{indentation}pass\n')

    return output_lines


def __parse_function__(lines: list, fn_spaces: int):
    output_lines = []

    while len(lines) > 0:
        line = lines.pop(0)
        n_spaces = __count_indentation__(line)

        # Ignore comments and empty lines
        if __is_comment_or_empty__(line):
            continue

        # Check if we un-indented and are outside of the function
        if n_spaces <= fn_spaces:
            lines.insert(0, line)
            break

        # Allow all other correctly indented lines
        output_lines.append(line)

    # Empty function without code so add "pass" statement
    if len(output_lines) == 0:
        indentation = ' ' * (fn_spaces+4)
        output_lines.append(f'{indentation}pass\n')

    return output_lines
