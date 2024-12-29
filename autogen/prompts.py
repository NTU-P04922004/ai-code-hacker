CODE_AGENT_SYSTEM_PROMPT = """\
You will be provided with a problem statement, and you need to create a Python solution for it.
In addition, you must follow the guidelines below:
## Guidelines
    a. Do not use any external libraries; stick to Python 3 standard library
    b. Handle input and output using standard input/output (stdin/stdout)
    c. Use helper functions to improve readability of the code.
    d. Use the `input()` function to take input from stdin and print the output to stdout.
    e. Do not add extra print statements otherwise it will fail the test cases.
    f. Make sure your code passes all potential test cases, including edge cases
    g. Follow the input/output format specified in the problem statement and the sample test cases.
    h. We will run the program by calling `python3 program.py` so make sure it outputs the correct results.
"""

INITIAL_PROMPT = """\
## Problem:
{problem_description}

## Sample Input: 
{sample_input}

## Expected Output: 
{sample_output}

Write the final solution in Python programming language to solve the problem.

"""

REFLECTION_PROMPT = """\
Reflect and provide critique on the following solution and test results.
Finally, give back the modified and updated solution.

## Solution
{solution}

## Test Results
{test_results}

"""
