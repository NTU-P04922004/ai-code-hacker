import argparse
import asyncio
import logging
import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from problem import Problem
from utils import check_correctness, run_python, extract_python_code_blocks, setup_logger


def run_and_test_code(code_path, input_path, output_path, gt_path, timeout=60):
    log = ""
    exit_code = 0
    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            run_python(code_path, input_path,
                       output_path, timeout=timeout)
        )

        program_output = output_path.read_text().strip()
        gt_output = gt_path.read_text().strip()
        if len(program_output) == 0:
            exit_code = 1
            log = "STATUS_ERR_NO_OUTPUT"
        else:
            correct = check_correctness(
                gt_output, program_output)

            if correct:
                log = "STATUS_SUCCESS"
            else:
                exit_code = 1
                failed_report = FAILED_REPORT_PROMPT.format(
                    gt_output=gt_output,
                    program_output=program_output
                )
                log = f"STATUS_FAILED\n{failed_report}"
    except asyncio.TimeoutError:
        exit_code = 1
        log = "STATUS_ERR_TIMEOUT"
    except Exception as e:
        exit_code = 1
        log = f"STATUS_ERR_EXECUTION\n{e}"
    return {
        "exit_code": exit_code, 
        "log": log
    }


class OneShotReflectionSolver:

    def __init__(self, model_name, num_reflections=3, temperature=0) -> None:
        self.model_name = model_name
        self.num_reflections = num_reflections
        self.temperature = temperature

        self.model = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=os.environ['GOOGLE_API_KEY'],
            temperature=self.temperature,
            convert_system_message_to_human=True
        )

        self.solving_chain = self._get_solving_chain()
        self.reflection_chain = self._get_reflection_chain()

    def _get_solving_chain(self):
        messages = [
            (
                "system",
                CODE_AGENT_SYSTEM_PROMPT
            ),
            (
                "human",
                INITIAL_PROMPT
            )
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt | self.model | StrOutputParser()

    def _get_reflection_chain(self):
        messages = [
            (
                "system",
                CODE_AGENT_SYSTEM_PROMPT
            ),
            (
                "human",
                REFLECTION_PROMPT
            )
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt | self.model | StrOutputParser()

    def _check_solution(self, solution, problem, generated_output_file, code_path):
        code = extract_python_code_blocks(solution)[0]

        logging.info("[Code]\n %s", code)

        # Save code
        code_path.write_text(code)

        test_report = run_and_test_code(
            code_path,
            problem.sample_input_path,
            generated_output_file,
            problem.sample_output_path,
            timeout=EXECUTION_TIMEOUT
        )

        return code, test_report

    def solve(self, problem_id: str, problem_title: str, data_dir: Path):

        target_dir = data_dir / problem_title
        problem = Problem.from_name(problem_id, target_dir)
        generated_output_file = target_dir / \
            Path(f"{problem_id}_generated.txt")
        code_path = target_dir / Path(f"{problem_id}_generated.py")

        solution = self.solving_chain.invoke({
            "problem_description": problem.problem_description,
            "sample_input": problem.get_sample_input(),
            "sample_output": problem.get_sample_output()
        })

        logging.info("[Solution]\n %s", solution)

        code, test_report = self._check_solution(solution, problem, generated_output_file, code_path)
        exit_code, test_results = test_report['exit_code'], test_report['log']
 
        logging.info("[Test Report]\n %s", test_report)

        if exit_code != 0:
            for i in range(self.num_reflections):
 
                logging.info("[Reflection #%d]\n", i + 1)

                solution = self.reflection_chain.invoke({
                    "solution": code,
                    "test_results": test_results
                })

                logging.info("[Solution]\n %s", solution)

                code, test_report = self._check_solution(solution, problem, generated_output_file, code_path)
                exit_code, test_results = test_report['exit_code'], test_report['log']

                logging.info("[Test Report]\n %s", test_report)

                if exit_code == 0:
                    break


EXECUTION_TIMEOUT = 60
NUM_REFLECTIONS = 3
MODEL_NAME = "models/gemini-1.5-flash"

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

FAILED_REPORT_PROMPT = """\
<expected>
{gt_output}
</expected>
---
<got>
{program_output}
</got>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangChain Coding Problem Hacker")
    parser.add_argument("problem_id", type=str,
                        help="The ID of the problem to solve.")
    parser.add_argument("problem_title", type=str,
                        help="The title of the problem to solve.")
    parser.add_argument("data_dir", type=Path,
                        help="Path to the data directory.")
    args = parser.parse_args()

    setup_logger(False)

    solver = OneShotReflectionSolver(MODEL_NAME, num_reflections=NUM_REFLECTIONS)
    solver.solve(
        args.problem_id,
        args.problem_title,
        args.data_dir
    )
