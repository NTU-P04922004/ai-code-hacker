import argparse
import asyncio
import os
from typing import List, Union, Dict, Callable
from pathlib import Path

from autogen import ConversableAgent
from autogen.coding import CodeBlock, CodeExecutor, CodeExtractor, CodeResult, MarkdownCodeExtractor

from problem import Problem
from prompts import CODE_AGENT_SYSTEM_PROMPT, FAILED_REPORT_PROMPT, INITIAL_PROMPT, REFLECTION_PROMPT
from utils import check_correctness, run_python


class PythonExecutor(CodeExecutor):

    @property
    def code_extractor(self) -> CodeExtractor:
        return MarkdownCodeExtractor()

    def __init__(
            self,
            input_path: Path,
            output_path: Path,
            gt_path: Path,
            code_path: Path,
            timeout: int = 120) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.gt_path = gt_path
        self.code_path = code_path
        self.timeout = timeout

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
        log = ""
        exitcode = 0
        code = code_blocks[0].code
        self.code_path.write_text(code)

        try:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                run_python(self.code_path, self.input_path,
                           self.output_path, timeout=self.timeout)
            )

            program_output = self.output_path.read_text().strip()
            gt_output = self.gt_path.read_text().strip()
            if len(program_output) == 0:
                exitcode = 1
                log += "STATUS_ERR_NO_OUTPUT"
            else:
                correct = check_correctness(
                    gt_output, program_output)

                if correct:
                    log += "STATUS_SUCCESS"
                else:
                    exitcode = 1
                    failed_report = FAILED_REPORT_PROMPT.format(
                        gt_output=gt_output,
                        program_output=program_output
                    )
                    log += f"STATUS_FAILED\n{failed_report}"
        except asyncio.TimeoutError:
            exitcode = 1
            log += "STATUS_ERR_TIMEOUT"
        return CodeResult(exit_code=exitcode, output=log)


def test_agent_message(
        recipient: ConversableAgent,
        messages: Union[str, Callable],
        sender: ConversableAgent,
        config: dict) -> Union[str, Dict]:
    return recipient.chat_messages_for_summary(sender)[-1]["content"]


def test_agent_summary(
        sender: ConversableAgent,
        recipient: ConversableAgent,
        summary_args: dict) -> str:
    test_results = sender.chat_messages[recipient][-1]["content"]
    solution = sender.chat_messages[recipient][-2]["content"]

    if "STATUS_SUCCESS" in test_results:
        return test_results

    return REFLECTION_PROMPT.format(
        solution=solution,
        test_results=test_results
    )


GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

GEMINI_CONFIG = {
    "config_list": [
        {
            "model": "gemini-1.5-flash",
            "api_key": GOOGLE_API_KEY,
            "api_type": "google"
        }
    ],
    "temperature": 0.1
}

NUM_REFLECTIONS = 5
EXECUTION_TIMEOUT = 60


def main(problem_id: str, problem_title: str, data_dir: Path):

    target_dir = data_dir / problem_title
    problem = Problem.from_name(problem_id, target_dir)
    generated_output_file = target_dir / Path(f"{problem_id}_generated.txt")
    code_path = target_dir / Path(f"{problem_id}_generated.py")

    executor = PythonExecutor(
        problem.sample_input_path,
        generated_output_file,
        problem.sample_output_path,
        code_path,
        timeout=EXECUTION_TIMEOUT
    )

    user_proxy = ConversableAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda x: x.get(
            "content", "") and "STATUS_SUCCESS" in x.get("content", "")
    )

    code_agent = ConversableAgent(
        "code_agent",
        human_input_mode="NEVER",
        code_execution_config=False,
        system_message=CODE_AGENT_SYSTEM_PROMPT,
        llm_config=GEMINI_CONFIG,
        is_termination_msg=lambda x: x.get(
            "content", "") and "STATUS_SUCCESS" in x.get("content", "")
    )

    test_agent = ConversableAgent(
        "test_agent",
        human_input_mode="NEVER",
        code_execution_config={"executor": executor}
    )

    nested_chats = [
        {
            "recipient": test_agent,
            "message": test_agent_message,
            "max_turns": 1,
            "summary_method": test_agent_summary
        }
    ]
    user_proxy.register_nested_chats(
        nested_chats,
        trigger=code_agent
    )

    response = user_proxy.initiate_chat(
        code_agent,
        message=INITIAL_PROMPT.format(
            problem_description=problem.problem_description,
            sample_input=problem.get_sample_input(),
            sample_output=problem.get_sample_output()
        ),
        clear_history=False,
        max_turns=1 + NUM_REFLECTIONS
    )

    print(response.chat_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoGen Coding Problem Hacker")
    parser.add_argument("problem_id", type=str,
                        help="The ID of the problem to solve.")
    parser.add_argument("problem_title", type=str,
                        help="The title of the problem to solve.")
    parser.add_argument("data_dir", type=Path,
                        help="Path to the data directory.")
    args = parser.parse_args()

    main(args.problem_id, args.problem_title, args.data_dir)
