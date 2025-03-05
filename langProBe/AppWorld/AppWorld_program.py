import json
import re
from typing import List, Tuple

from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .AppWorld_utils.app_world_server_manager import appworld_manager

import shortuuid
import dspy

import ast


def strip_comments(code_str):
    try:
        parsed_code = ast.parse(code_str)
        return ast.unparse(parsed_code)
    except:
        return code_str


class AppWorldReactSignature(dspy.Signature):
    """Your job is to achieve my day-to-day tasks completely autonomously.

    To do this, you will need to interact with app/s (e.g., spotify, venmo etc) using their associated APIs on my behalf. For this you will undertake a *multi-step interaction* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

    Here are three key APIs that you need to know to get more information:

    # To get a list of apps that are available to you.
    print(apis.api_docs.show_app_descriptions())

    # To get the list of apis under any app listed above, e.g. spotify
    print(apis.api_docs.show_api_descriptions(app_name='spotify'))

    # To get the specification of a particular api, e.g. spotify app's login api
    print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))

    Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that I will execute, to solve the task.

    **Key instructions**:
    (1) Remember you can use the variables in your code in subsequent code blocks.

    (2) Remember that the email addresses, access tokens and variables (e.g. spotify_password) in the example above are not valid anymore.

    (3) You can use the "supervisor" app to get information about my accounts and use the "phone" app to get information about friends and family.

    (4) Always look at API specifications (using apis.api_docs.show_api_doc) before calling an API.

    (5) Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.

    (6) Many APIs return items in "pages". Make sure to run through all the pages by looping over `page_index`.

    (7) Once you have completed the task, make sure to call apis.supervisor.complete_task(). If the task asked for some information, return it as the answer argument, i.e. call apis.supervisor.complete_task(answer=<answer>). Many tasks do not require an answer, so in those cases, just call apis.supervisor.complete_task() i.e. do not pass any argument.

    Using these APIs, now generate the next code step to solve the actual task."""

    instruction = dspy.InputField(description="Task to be completed", format=str)
    supervisor_name = dspy.InputField()
    supervisor_phone_number = dspy.InputField()
    supervisor_email = dspy.InputField()
    past_steps = dspy.InputField(
        description="Past code executions and their outputs", format=str
    )
    code = dspy.OutputField(description="Python code to be executed", format=str)


class AppWorldReact(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self, max_steps=40, module=None, add_few_shot=True):
        if module is None:
            self.module = dspy.Predict(signature=AppWorldReactSignature)
        else:
            self.module = module
        self.max_steps = max_steps
        self.add_few_shot = add_few_shot

        if self.add_few_shot:
            with open("langProBe/AppWorld/fewshot_trace.json", "r") as f:
                past_steps_trace = json.load(f)

            fewshotexample = dspy.Example(
                instruction="How many playlists do I have in Spotify?",
                supervisor_name="John Doe",
                supervisor_phone_number="0123456789",
                supervisor_email="johndoe@example.com",
                past_steps=self.format_trace(past_steps_trace[:-1]),
                code=past_steps_trace[-1][0],
            ).with_inputs(
                "instruction",
                "supervisor_name",
                "supervisor_phone_number",
                "supervisor_email",
                "past_steps",
            )

            self.module.demos.append(fewshotexample)

        self.known_apis = {}
        self.app_docs = {}

    def format_trace(self, trace: List[Tuple[str, str]]) -> str:
        # return "\n\n".join([f"Code:\n{code}\nOutput:\n{output}" for code, output in trace])
        if len(trace) == 0:
            return ""
        return "\n\n".join(
            [
                f"Step {idx+1}\nCode:\n{code}\n\nOutput:\n{output}"
                for idx, (code, output) in enumerate(trace)
            ]
        )

    def forward(self, task_id):
        output_experiment_name = shortuuid.uuid()
        with appworld_manager.acquire_server(output_experiment_name, task_id) as server:
            task = server.request.show_task(task_id)
            supervisor_name = (
                task["supervisor"]["first_name"] + " " + task["supervisor"]["last_name"]
            )
            supervisor_email = task["supervisor"]["email"]
            supervisor_phone = task["supervisor"]["phone_number"]

            trace = []
            for i in range(self.max_steps):
                while True:
                    try:
                        module_gen = self.module(
                            instruction=task["instruction"],
                            supervisor_name=supervisor_name,
                            supervisor_phone_number=supervisor_phone,
                            supervisor_email=supervisor_email,
                            past_steps=self.format_trace(trace),
                        )
                        break
                    except ValueError as ve:
                        # TODO: This is a hack to deal with the fact that the model runs out of context window. Need better ways to resolve this
                        if (
                            ve.args[0]
                            == "Expected dict_keys(['code']) but got dict_keys([])"
                        ):
                            if len(trace) >= 1:
                                trace = trace[1:]
                            else:
                                raise ve
                        else:
                            raise ve
                gen_code = module_gen.code
                if "os.path.expanduser" in strip_comments(gen_code):
                    gen_code_output = (
                        "Usage of the following os.path.expanduser is not allowed."
                    )
                else:
                    gen_code_output = server.request.execute(
                        task_id=task_id, code=gen_code
                    )
                trace.append((gen_code, gen_code_output))

                if "Execution failed" in gen_code_output:
                    if re.search(
                        r"No API named '(.*)' found in the (.*) app.", gen_code_output
                    ):
                        app_name = re.search(
                            r"No API named '(.*)' found in the (.*) app.",
                            gen_code_output,
                        ).groups()[1]
                        extract_api_code = f"# Let me check the available APIs in the {app_name} app.\nprint(apis.api_docs.show_api_descriptions(app_name='{app_name}'))"
                        if app_name not in self.app_docs:
                            self.app_docs[app_name] = server.request.execute(
                                task_id=task_id, code=extract_api_code
                            )
                        if "Execution failed" not in self.app_docs[app_name]:
                            trace.append((extract_api_code, self.app_docs[app_name]))
                    elif (
                        "apis" in gen_code_output
                        and "KeyError" not in gen_code_output
                        and "TypeError" not in gen_code_output
                        and "NameError" not in gen_code_output
                    ):
                        extract_api = gen_code_output.split("apis.")[1].split("(")[0]
                        extract_api_code = f"# There was an issue with the way I invoked the API. Let me check the documentations to fix the issue.\nprint(apis.api_docs.show_api_doc(app_name='{extract_api.split('.')[0]}', api_name='{extract_api.split('.')[1]}'))"
                        if extract_api not in self.known_apis:
                            self.known_apis[extract_api] = server.request.execute(
                                task_id=task_id, code=extract_api_code
                            )
                        if "Execution failed" not in self.known_apis[extract_api]:
                            trace.append(
                                (extract_api_code, self.known_apis[extract_api])
                            )
                    elif re.search(
                        r"Usage of the following .* is not allowed", gen_code_output
                    ):
                        trace[-1] = (
                            gen_code,
                            gen_code_output
                            + " You must use the file system app to access the file system.",
                        )

                if server.request.task_completed(task_id=task_id):
                    break

            eval_report = server.request.evaluate(
                task_id=task_id, suppress_errors=True, report=False
            )
            return dspy.Prediction(
                trace=trace,
                experiment_name_to_eval=output_experiment_name,
                eval_report=eval_report,
            )
