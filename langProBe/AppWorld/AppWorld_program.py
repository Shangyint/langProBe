import re
from typing import List, Tuple
from .AppWorld_utils.app_world_server_manager import appworld_manager

import shortuuid
import dspy

class AppWorldReactSignature(dspy.Signature):
    """You are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information:

# 1. To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# 2. To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# 3. To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Your code will be executed in a Python environment that has access to the APIs of the apps, and if anything is printed, it will be shown to you. You can also print variables to see their values. However, if the code raises an exception, you will see the error message.

**Key instructions and disclaimers**:

1. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
2. You can use the variables from the previous code blocks in the subsequent code blocks. To view the variables, you can print them.
3. Make sure everything is working correctly before making any irreversible change.
4. The provided Python environment has access to its standard library and the APIs provided by the apps. Do not use any external libraries. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify.
5. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
6. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
7. Different apps may have different login credentials. Read the API documentation for the app to understand how to login.
8. For APIs that return results in "pages", make sure to consider all pages.
9. To obtain current data or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
10. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
11. You must make all decisions completely autonomously and not ask for any clarifications or confirmations from me or anyone else.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials (like login details), physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. Once you have completed the task, call `apis.supervisor.complete_task()`. If the task asks for some information, return it as the answer argument, i.e. call `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, just skip the answer argument or pass it as None.
15. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
16. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.
17. Do not mock any data or responses. All data should be fetched from the apps.
18. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change."""
    instruction = dspy.InputField(description="Task to be completed", format=str)
    supervisor_name = dspy.InputField()
    supervisor_phone_number = dspy.InputField()
    supervisor_email = dspy.InputField()
    past_steps = dspy.InputField(description="Past code executions and their outputs", format=str)
    code = dspy.OutputField(description="Python code to be executed", format=str)

class AppWorldReact(dspy.Module):
    def __init__(self, max_steps=40):
        self.module = dspy.Predict(signature=AppWorldReactSignature)
        self.max_steps = max_steps
        self.known_apis = {}
        self.app_docs = {}
    
    def format_trace(self, trace: List[Tuple[str, str]]) -> str:
        # return "\n\n".join([f"Code:\n{code}\nOutput:\n{output}" for code, output in trace])
        if len(trace) == 0:
            return "# No code executed"
        return "\n\n".join([f"Step {idx+1}\nCode:\n{code}\n\nOutput:\n{output}" for idx, (code, output) in enumerate(trace)])

    def forward(self, task_id):
        output_experiment_name = shortuuid.uuid()
        with appworld_manager.acquire_server(output_experiment_name, task_id) as server:
            task = server.request.show_task(task_id)
            supervisor_name = task['supervisor']['first_name'] + " " + task['supervisor']['last_name']
            supervisor_email = task['supervisor']['email']
            supervisor_phone = task['supervisor']['phone_number']

            print(f"Task: {task['instruction']}")
            print(f"Supervisor: {supervisor_name}")
            print(f"Supervisor email: {supervisor_email}")
            print(f"Supervisor phone: {supervisor_phone}")

            trace = []
            for i in range(self.max_steps):
                while True:
                    try:
                        module_gen = self.module(
                            instruction=task['instruction'],
                            supervisor_name=supervisor_name,
                            supervisor_phone_number=supervisor_phone,
                            supervisor_email=supervisor_email,
                            past_steps=self.format_trace(trace),
                        )
                        break
                    except ValueError as ve:
                        # TODO: This is a hack to deal with the fact that the model runs out of context window. Need better ways to resolve this
                        if ve.args[0] == "Expected dict_keys(['code']) but got dict_keys([])":
                            if len(trace) >= 1:
                                trace = trace[1:]                            
                            else:
                                raise ve
                        else:
                            raise ve
                gen_code = module_gen.code
                gen_code_output = server.request.execute(task_id=task_id, code=gen_code)
                trace.append((gen_code, gen_code_output))

                print(f"Step {i+1}:")
                print(f"Code:\n{gen_code}")
                print(f"Output:\n{gen_code_output}")

                if "Execution failed" in gen_code_output:
                    # if "Exception: No API named 'get_friends' found in the venmo app." in 
                    if re.search(r"No API named '(.*)' found in the (.*) app.", gen_code_output):
                        app_name = re.search(r"No API named '(.*)' found in the (.*) app.", gen_code_output).groups()[1]
                        extract_api_code = f"# Let me check the available APIs in the {app_name} app.\nprint(apis.api_docs.show_api_descriptions(app_name='{app_name}'))"
                        if app_name not in self.app_docs:
                            # self.known_apis[extract_api] = server.request.execute(task_id=task_id, code=extract_api_code)
                            self.app_docs[app_name] = server.request.execute(task_id=task_id, code=extract_api_code)
                        # if "Execution failed" not in self.known_apis[extract_api]:
                        if "Execution failed" not in self.app_docs[app_name]:
                            trace.append((extract_api_code, self.app_docs[app_name]))
                            print(f"Step {i+1} free pass:")
                            print(f"Code:\n{extract_api_code}")
                            print(f"Output:\n{self.app_docs[app_name]}")
                    elif "apis" in gen_code_output and "KeyError" not in gen_code_output and "TypeError" not in gen_code_output and "NameError" not in gen_code_output:
                        extract_api = gen_code_output.split("apis.")[1].split("(")[0]
                        # if extract_api not in known_apis:
                        extract_api_code = f"# There was an issue with the way I invoked the API. Let me check the documentations to fix the issue.\nprint(apis.api_docs.show_api_doc(app_name='{extract_api.split('.')[0]}', api_name='{extract_api.split('.')[1]}'))"
                        if extract_api not in self.known_apis:
                            self.known_apis[extract_api] = server.request.execute(task_id=task_id, code=extract_api_code)
                        if "Execution failed" not in self.known_apis[extract_api]:
                            trace.append((extract_api_code, self.known_apis[extract_api]))
                            print(f"Step {i+1} free pass:")
                            print(f"Code:\n{extract_api_code}")
                            print(f"Output:\n{self.known_apis[extract_api]}")

                if server.request.task_completed(task_id=task_id):
                    break
            
            eval_report = server.request.evaluate(task_id=task_id, suppress_errors=True, report=False)
            print(f"Evaluation report: {eval_report}")
            # raise ValueError("Task completed")
            return dspy.Prediction(trace=trace, experiment_name_to_eval=output_experiment_name, eval_report=eval_report)
