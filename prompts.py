# Reference:
#   https://github.com/wandb/aihackercup/blob/main/one_shot_o1.py
#   https://arxiv.org/abs/2410.01428v1
#


# system_prompt = "You are an expert problem solver. Your task is creating the code to solve the problem at hand in python."

system_prompt = """You will be provided with a problem statement, and you need to create a Python3 solution for it. 
Write the final solution in Python3 programming language to solve the problem.

## Competition Guidelines:
    a. Do not use any external libraries; stick to Python 3 standard library
    b. Handle input and output using standard input/output (stdin/stdout)
    c. Use helper functions to improve readability of the code.
    c. Use the `input()` function to take input from stdin and print the output to stdout.
    d. Do not add extra print statements otherwise it will fail the test cases.
    e. Make sure your code passes all potential test cases, including edge cases
    f. Follow the input/output format specified in the problem statement and the sample test cases.
    g. We will run the program by calling `python3 program.py` so make sure it outputs the correct results.
"""

# prompt_template = """
# ## Problem: 
# {problem_description}

# ## Input: 
# {sample_input}

# ## Output: 
# {sample_output}

# Create a python program that returns the correct output for the given input. 

# ## Competition Guidelines:
#     a. Do not use any external libraries; stick to Python 3 standard library
#     b. Handle input and output using standard input/output (stdin/stdout)
#     c. Use helper functions to improve readability of the code.
#     c. Use the `input()` function to take input from stdin and print the output to stdout.
#     d. Do not add extra print statements otherwise it will fail the test cases.
#     e. Make sure your code passes all potential test cases, including edge cases
#     f. Follow the input/output format specified in the problem statement and the sample test cases.
#     g. We will run the program by calling `python3 program.py` so make sure it outputs the correct results.
# """

# system_prompt = """You are an expert problem solver. Your task is creating the code to solve the problem at hand in python.

# ## Competition Guidelines:
#     a. Do not use any external libraries; stick to Python 3 standard library
#     b. Handle input and output using standard input/output (stdin/stdout)
#     c. Use helper functions to improve readability of the code.
#     c. Use the `input()` function to take input from stdin and print the output to stdout.
#     d. Do not add extra print statements otherwise it will fail the test cases.
#     e. Make sure your code passes all potential test cases, including edge cases
#     f. Follow the input/output format specified in the problem statement and the sample test cases.
#     g. We will run the program by calling `python3 program.py` so make sure it outputs the correct results.

# """

# prompt_template = """You are an expert problem solver. Your task is creating the code to solve the problem at hand in python.

# ## Problem:
# {problem_description}

# ## Sample Input: 
# {sample_input}

# ## Expected Output: 
# {sample_output}

# ## Competition Guidelines:
#     a. Do not use any external libraries; stick to Python 3 standard library
#     b. Handle input and output using standard input/output (stdin/stdout)
#     c. Use helper functions to improve readability of the code.
#     c. Use the `input()` function to take input from stdin and print the output to stdout.
#     d. Do not add extra print statements otherwise it will fail the test cases.
#     e. Make sure your code passes all potential test cases, including edge cases
#     f. Follow the input/output format specified in the problem statement and the sample test cases.
#     g. We will run the program by calling `python3 program.py` so make sure it outputs the correct results.

# Create a python program that returns the correct output for the given input.
# """

prompt_template = """## Problem:
{problem_description}

## Sample Input: 
{sample_input}

## Expected Output: 
{sample_output}
"""

REFLECTION_PROMPT = """You have incorrectly answered the following programming problem. 
Your task is to reflect on the problem, your previous solution, the correct answer, and the test result.
Write the improved solution in Python3 programming language to solve the problem.

## Competition Guidelines:
    a. Do not use any external libraries; stick to Python 3 standard library
    b. Handle input and output using standard input/output (stdin/stdout)
    c. Use helper functions to improve readability of the code.
    c. Use the `input()` function to take input from stdin and print the output to stdout.
    d. Do not add extra print statements otherwise it will fail the test cases.
    e. Make sure your code passes all potential test cases, including edge cases
    f. Follow the input/output format specified in the problem statement and the sample test cases.
    g. We will run the program by calling `python3 program.py` so make sure it outputs the correct results.

{problem_description}

## Sample Input: 
{sample_input}

## Expected Output: 
{sample_output}

## Previous Solution
```python
{previous_solution}
```

## Test Results
{test_report}

"""

fix_code_prompt_template = """Add an entry point for the following python code:

```python
{code}
```
"""