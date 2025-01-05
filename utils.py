import asyncio
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import List


def load_jsonl(file: Path) -> List[dict]:
    """Load a JSONL file"""
    with open(file, 'r') as f:
        return [json.loads(line) for line in f]


def setup_logger(debug=False, silence_openai=True):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[logging.StreamHandler()]
    )

    # silence openai logger
    if silence_openai:
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def maybe_remove_backticks(solution: str) -> str:
    "Remove backticks from the solution"
    solution = solution.strip()
    solution = re.sub(r'^```python\s*', '', solution)
    solution = re.sub(r'\s*```$', '', solution)
    return solution


def compare_lines_with_tolerance(expected: str, actual: str, tolerance: float = 1e-9) -> bool:
    """
    Compare two lines of output with a tolerance for floating point numbers.
    """
    expected_lines = expected.strip().split('\n')
    actual_lines = actual.strip().split('\n')

    if len(expected_lines) != len(actual_lines):
        return False

    for expected_line, actual_line in zip(expected_lines, actual_lines):
        expected_match = re.match(r"Case #\d+: (.+)", expected_line)
        actual_match = re.match(r"Case #\d+: (.+)", actual_line)

        if not expected_match or not actual_match:
            return False

        expected_values = expected_match.group(1).split()
        actual_values = actual_match.group(1).split()

        if len(expected_values) != len(actual_values):
            return False

        for expected_value, actual_value in zip(expected_values, actual_values):
            try:
                expected_float = float(expected_value)
                actual_float = float(actual_value)
                if not math.isclose(expected_float, actual_float, rel_tol=tolerance):
                    return False
            except ValueError:
                if expected_value != actual_value:
                    return False

    return True


def check_correctness(expected: str, actual: str) -> dict:
    "Check the solution against the expected output"
    return compare_lines_with_tolerance(expected, actual)


async def _run_subprocess(command: list, input_file: Path, output_file: Path, timeout: float):
    """Run a subprocess with the given command, input file, and output file.
    Parameters:
    command (list): The command to execute as a list of arguments.
    input_file (Path): The path to the input file.
    output_file (Path): The path to the output file.
    timeout (float): The maximum time to allow for the subprocess to run.

    Raises:
        RuntimeError: If the subprocess fails or times out."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=input_file.open("rb"),
            stdout=output_file.open("wb"),
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(
                f"Program execution timed out after {timeout} seconds")

        if process.returncode != 0:
            raise RuntimeError(f"Program execution failed: {stderr.decode()}")

        logging.debug(f"Output saved to {output_file}")
    except Exception as e:
        raise RuntimeError(f"Error running subprocess: {str(e)}")


async def run_python(program: Path, input_file: Path, output_file: Path, timeout: float = 10):
    """Run a Python program with the given input file and output file.

    Parameters:
        program (Path): The path to the Python program to execute.
        input_file (Path): The path to the input file.
        output_file (Path): The path to the output file.
        timeout (float): The maximum time to allow for the program to run.

    Raises:
        RuntimeError: If there is an error running the Python program.
    """
    await _run_subprocess([sys.executable, str(program)], input_file, output_file, timeout)


async def run_program(code: Path, input: Path, output: Path, timeout: float = 10, cpp_version: int = 11):
    """
    Run the program with the given code and input file. Write the output to the given output file.
    """
    try:
        if code.suffix == ".py":
            logging.debug(f"Running Python program: {code}")
            await run_python(code, input, output, timeout)
        else:
            raise ValueError(f"Unsupported file type: {code}")
    except Exception as e:
        raise e
    return


def extract_python_code_blocks(markdown_string: str) -> List[str]:
    # Define a regular expression to find Python code blocks
    # This regex looks for ```python to open and ``` to close, capturing content in between
    pattern = r"```python(.*?)```"

    # Use re.DOTALL to make '.' match newlines as well
    code_blocks = re.findall(pattern, markdown_string, re.DOTALL)

    # Strip leading and trailing whitespace (including newlines) from each code block
    return [block.strip() for block in code_blocks]
