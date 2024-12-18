from dataclasses import dataclass

import requests


@dataclass
class AppWorldInitializeArgs:
    task_id: str
    experiment_name: str
    # remote_apis_url: Optional[str] = None
    # remote_environment_url: Optional[str] = None
    # remote_docker: bool = False
    # max_interactions: int = 1000
    # max_api_calls_per_interaction: int = 1000
    # raise_on_unsafe_syntax: bool = True
    # null_patch_unsafe_execution: bool = True
    # load_ground_truth: bool = True
    # ground_truth_mode: Literal["fullminimal"] = "minimal"
    # raise_on_failure: bool = True
    # random_seed: int = 100
    # timeout_seconds: int = 100
    # show_api_response_schemas: bool = True
    # gc_threshold: int = 500000
    # raise_on_extra_parameters: bool = False
    # import_utils: bool = False
    # parse_datetimes: bool = False
    # allow_datetime_change: bool = False
    # add_login_shortcut: bool = False
    # munchify_response: bool = False


@dataclass
class ExecuteArgs:
    task_id: str
    code: str


@dataclass
class CloseArgs:
    task_id: str


# @dataclass
# class CloseAllArgs:
#     task_id: str


@dataclass
class EvaluateArgs:
    task_id: str
    suppress_errors: bool = False
    report: bool = False


@dataclass
class SaveLogsArgs:
    task_id: str


@dataclass
class SaveStateArgs:
    task_id: str
    state_id: str


@dataclass
class LoadStateArgs:
    task_id: str
    state_id: str


@dataclass
class TaskCompletedArgs:
    task_id: str


class AppWorldClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def _make_request(self, method, endpoint, data=None):
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method in ["POST", "PUT", "DELETE"]:
            response = requests.request(method, url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()
        return response.json()["output"]

    def index(self):
        return self._make_request("GET", "/")

    def initialize(self, experiment_name: str, task_id: str):
        args = AppWorldInitializeArgs(
            experiment_name=experiment_name, task_id=task_id
        ).__dict__
        return self._make_request("POST", "/initialize", data=args)

    def execute(self, task_id, code):
        data = {"task_id": task_id, "code": code}
        return self._make_request("POST", "/execute", data=data)

    def close(self, task_id):
        data = {"task_id": task_id}
        return self._make_request("POST", "/close", data=data)

    def close_all(self, task_id):
        data = {"task_id": task_id}
        return self._make_request("POST", "/close_all", data=data)

    def evaluate(self, task_id, suppress_errors=False, report=False):
        data = {
            "task_id": task_id,
            "suppress_errors": suppress_errors,
            "report": report,
        }
        return self._make_request("POST", "/evaluate", data=data)

    def save_logs(self, task_id):
        data = {"task_id": task_id}
        return self._make_request("POST", "/save_logs", data=data)

    def save_state(self, task_id, state_id):
        data = {"task_id": task_id, "state_id": state_id}
        return self._make_request("POST", "/save_state", data=data)

    def load_state(self, task_id, state_id):
        data = {"task_id": task_id, "state_id": state_id}
        return self._make_request("POST", "/load_state", data=data)

    def task_completed(self, task_id):
        data = {"task_id": task_id}
        return self._make_request("POST", "/task_completed", data=data)

    def show_task(self, task_id):
        return self._make_request("GET", f"/tasks/{task_id}")

    def api_docs(self):
        return self._make_request("GET", "/api_docs")

    def launch_playground(self):
        return self._make_request("GET", "/playground")
