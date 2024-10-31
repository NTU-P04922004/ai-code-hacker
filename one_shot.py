import asyncio
import re
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import wandb
import openai
import weave
import simple_parsing
import instructor
from pydantic import BaseModel, Field

from problem import Problem
from solution import ExtractedSolution, Solution
from utils import maybe_remove_backticks, check_correctness, setup_logger, run_program
from prompts import system_prompt, prompt_template, fix_code_prompt_template


HEAVY_MODEL = "gpt-4o"
LITE_MODEL = "gpt-4o-mini"
WANDB_API_KEY = "398c00452049342e69728cf600b142cd7cd0bf5b"
WEAVE_PROJECT_ID = "ai-code-hacker-test"
RANDOM_SEEDS = [
    754268, 738153, 130790, 752576, 21838, 910034,
    917598, 398248, 919872, 657143, 979112, 217509,
    783514, 81463, 473165, 710501, 134994, 55516,
    546141, 990185, 282349, 503905, 651841, 115429,
    616545, 683375, 798343, 628677, 875733, 890746,
    908384, 741350, 87535, 743941, 147246, 457002
]


os.environ["WANDB_API_KEY"] = WANDB_API_KEY
client = instructor.from_openai(openai.OpenAI())


class TestReport(BaseModel):
    status: str
    message: str

    @property
    def as_xml(self) -> str:
        return f"""
<test_report>
<status>{self.status}</status>
<message>{self.message}</message>
</test_report>
"""


@dataclass
class Args(simple_parsing.Serializable):
    problem_name: str = "cheeseburger_corollary_ch1"  # name of the problem to solve
    # path to the folder containing the problems
    folder_path: Path = Path("./dataset/2023/practice/")
    model: str = HEAVY_MODEL  # openai model to use
    secondary_model: str = LITE_MODEL  # openai model to use
    use_images: bool = False  # set to True to use images in the prompt
    save_output: bool = True  # set to True to save the output to a file
    debug: bool = False  # set to True to see the debug logs
    timeout: int = 60  # timeout for the code execution
    num_reflexion: int = 1
    best_of_n: int = 5


def extract_python_code_blocks(markdown_string):
    # Define a regular expression to find Python code blocks
    # This regex looks for ```python to open and ``` to close, capturing content in between
    pattern = r"```python(.*?)```"

    # Use re.DOTALL to make '.' match newlines as well
    code_blocks = re.findall(pattern, markdown_string, re.DOTALL)

    # Strip leading and trailing whitespace (including newlines) from each code block
    return [block.strip() for block in code_blocks]


@weave.op
def fix_python_code(
        code: str,
        model: str,
        seed: int = 42) -> str:
    logging.info("Fix python code solution")

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": fix_code_prompt_template.format(
                code=code
            )}
        ]}
    ]

    logging.info("Generating a fixed solution")
    out = call_model(
        messages=messages,
        model=model,
        response_model=None,
        temperature=0,
        seed=seed)
    out = out.choices[0].message.content

    logging.info("  Extracting the code from the previous generation...")
    source_code = extract_python_code_blocks(out)[0]
    return source_code


@weave.op
def call_model(messages, **kwargs):
    response_model = kwargs.pop("response_model", str)
    res = client.chat.completions.create(
        messages=messages,
        response_model=response_model,
        **kwargs
    )
    return res


@weave.op
def generate_code(
        problem: Problem,
        system_prompt: str,
        prompt_template: str,
        model: str,
        use_images: bool = False,
        seed: int = 42) -> str:
    logging.info(f"Generating code solution for: {problem.name}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_template.format(
                problem_description=problem.problem_description,
                sample_input=problem.get_sample_input(),
                sample_output=problem.get_sample_output(),
            )}
        ] + ([{"type": "image_url", "image_url": {"url": img}} for img in problem.images] if use_images else [])}
    ]

    # call model one first time to get the code
    logging.info("Generating initial analysis and solution")
    out = call_model(
        messages=messages,
        model=model,
        response_model=None,
        temperature=0.2,
        seed=seed)
    out = out.choices[0].message.content

    logging.info("  Extracting the code from the previous generation...")
    source_code = extract_python_code_blocks(out)[0]

    return Solution(
        source_code=source_code,
        solution_explanation="",  # solution.solution_explanation,
        problem_name=problem.name,
        problem_folder=problem.folder_path,
    )


@weave.op
def reflection(
        problem: Problem,
        previous_solution: str,
        test_report: str,
        system_prompt: str,
        prompt_template: str,
        model: str,
        seed: int = 42) -> str:

    logging.info(f"Generating code solution for: {problem.name}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_template.format(
                problem_description=problem.problem_description,
                sample_input=problem.get_sample_input(),
                sample_output=problem.get_sample_output(),
                previous_solution=previous_solution,
                test_report=test_report
            )}
        ]}
    ]

    logging.info("Generating reflection analysis and solution")
    out = call_model(
        messages=messages,
        model=model,
        response_model=None,
        temperature=0,
        seed=seed)
    out = out.choices[0].message.content

    logging.info("  Extracting the code from the previous generation...")
    source_code = extract_python_code_blocks(out)[0]

    return Solution(
        source_code=source_code,
        solution_explanation="",  # solution.solution_explanation,
        problem_name=problem.name,
        problem_folder=problem.folder_path,
    )


@weave.op
async def run_and_test(code: str, input_file: Path, output_file: Path, generated_output_file: Path, timeout: int = 30):
    """
    Run the program and test the output against the sample output.

    Args:
        code: The path to the code file.
        input_file: The path to the input file.
        output_file: The path to the ground truth output file.
        generated_output_file: The path to the file where the generated output will be saved.
        timeout: The timeout for the code execution.
    """
    try:
        await run_program(code, input_file, generated_output_file, timeout=timeout)
    except asyncio.TimeoutError:
        return TestReport(
            status="timeout",
            message=f"Took too long! Your program timed out after {timeout} seconds of execution."
        )
    except Exception as e:
        logging.error(f"Error running program: {e}")
        return TestReport(
            status="error", message=f"Program execution failed: {str(e)}"
        )

    if len(generated_output_file.read_text().strip()) == 0:
        return TestReport(
            status="error", message="The program has no outputs."
        )

    correct = check_correctness(
        output_file.read_text(), generated_output_file.read_text())
    if correct:
        return TestReport(
            status="passed", message="Yay! Your program ran successfully"
        )
    else:
        return TestReport(
            status="failed",
            message=f"<expected>\n{output_file.read_text()}</expected>\n---\n<got>\n{generated_output_file.read_text()}</got>",
        )


if __name__ == "__main__":
    args = simple_parsing.parse(Args)

    setup_logger(args.debug)
    wandb.login()
    weave.init(WEAVE_PROJECT_ID)

    problem = Problem.from_name(args.problem_name, args.folder_path)

    for i in range(1, args.best_of_n + 1):
        logging.info(f"> Trial {i}: Solving on sample input...")
        solution = generate_code(
            problem,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            model=args.model,
            use_images=args.use_images,
            seed=RANDOM_SEEDS[i - 1])

        code_file = solution.save_code()
        generated_output_file = problem.folder_path / \
            (problem.name + f"_generated_{i}.out")

        logging.info(
            "> Running and testing the solution on sample input/output...")
        run_and_test_result = asyncio.run(run_and_test(
            code_file,
            problem.sample_input_path,
            problem.sample_output_path,
            generated_output_file,
            timeout=args.timeout))

        logging.info(f"> Test sample output: {run_and_test_result}")

        if run_and_test_result.status == "error" and \
           run_and_test_result.message == "The program has no outputs.":
            solution.source_code = fix_python_code(
                solution.source_code,
                model=args.secondary_model  
            )
            code_file = solution.save_code()
            generated_output_file = problem.folder_path / (problem.name + f"_generated_{i}_fixed.out")
        
            logging.info(
                "> Running and testing the solution on sample input/output...")
            run_and_test_result = asyncio.run(run_and_test(
                code_file,
                problem.sample_input_path,
                problem.sample_output_path,
                generated_output_file,
                timeout=args.timeout))
            
            logging.info(f"> Test sample output: {run_and_test_result}")
