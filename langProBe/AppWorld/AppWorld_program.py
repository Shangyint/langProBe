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

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.

**Key instructions and disclaimers**:

1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. You can use the variables from the previous code blocks in the subsequent code blocks.
4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages.
10. To obtain current data or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. Once you have completed the task, call `apis.supervisor.complete_task()`. If the task asks for some information, return it as the answer argument, i.e. call `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, just skip the answer argument or pass it as None.
15. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
16. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.
17. You must make all decisions completely autonomously and not ask for any clarifications or confirmations from me or anyone else."""
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
    
    def format_trace(self, trace: List[Tuple[str, str]]) -> str:
        return "\n".join([f"Code: {code}\nOutput: {output}" for code, output in trace])

    def forward(self, task_id):
        output_experiment_name = shortuuid.uuid()
        with appworld_manager.acquire_server(output_experiment_name, task_id) as server:
            task = server.request.show_task(task_id)
            supervisor_name = task['supervisor']['first_name'] + " " + task['supervisor']['last_name']
            supervisor_email = task['supervisor']['email']
            supervisor_phone = task['supervisor']['phone_number']

            trace = []
            for i in range(self.max_steps):
                module_gen = self.module(
                    instruction=task['instruction'],
                    supervisor_name=supervisor_name,
                    supervisor_phone_number=supervisor_phone,
                    supervisor_email=supervisor_email,
                    past_steps=self.format_trace(trace),
                )
                gen_code = module_gen.code
                gen_code_output = server.request.execute(task_id=task_id, code=gen_code)
                trace.append((gen_code, gen_code_output))

                if server.request.task_completed(task_id=task_id):
                    break

            return dspy.Prediction(trace=trace, experiment_name_to_eval=output_experiment_name)
