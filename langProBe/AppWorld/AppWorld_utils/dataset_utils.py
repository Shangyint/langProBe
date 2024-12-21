import json
from collections import defaultdict

from .appworld_setup import get_appworld_root


# All of the following functions are adapted from AppWorld: https://github.com/StonyBrookNLP/appworld/blob/5115df1480bd52be412bac6d92a6552ef6b60703/src/appworld/task.py#L234
def remove_tag(task_id: str) -> str:
    return task_id.split(":")[0]

def task_id_to_generator_id(task_id: str) -> str:
    task_id = remove_tag(task_id)
    assert task_id.count("_") == 1
    return task_id.split("_")[0]

def task_id_to_number(task_id: str) -> int:
    task_id = remove_tag(task_id)
    assert task_id.count("_") == 1
    return int(task_id.split("_")[1])

def task_id_to_tag(task_id: str) -> str:
    if ":" not in task_id:
        return ""
    return task_id.split(":")[1].strip()

def _maybe_assure_num_tasks_per_scenario(
    task_ids: list[str], num_tasks_per_scenario: int | None = None
) -> None:
    if num_tasks_per_scenario is not None:
        generator_id_to_task_ids: dict[str, list[str]] = defaultdict(list)
        for task_id in task_ids:
            generator_id_to_task_ids[task_id_to_generator_id(task_id)].append(task_id)
        for generator_id, task_ids_ in generator_id_to_task_ids.items():
            task_numbers_expected = list(range(1, num_tasks_per_scenario + 1))
            task_numbers_found = sorted(task_id_to_number(task_id_) for task_id_ in task_ids_)
            if task_numbers_found != task_numbers_expected:
                raise Exception(
                    f"The task numbers for scenario {generator_id} are expected to be in "
                    f"{task_numbers_expected} but found {task_numbers_found}."
                )

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_task_ids(
    dataset_name: str = "train",
    difficulty: int | None = None,
    num_tasks_per_scenario: int | None = None,
    only_tagged: list[str] | None = None,
) -> list[str]:
    appworld_root = get_appworld_root()
    data_root = appworld_root / "data"
    datasets_directory = data_root / "datasets"
    tasks_directory = data_root / "tasks"

    if not tasks_directory.exists():
        raise Exception(f"The task directory ({tasks_directory}) doesn't exist.")

    assert datasets_directory is not None  # for mypy
    dataset_file_path = datasets_directory / (dataset_name + ".txt")
    if not dataset_file_path.exists():
        raise Exception(f"The dataset file ({dataset_file_path}) doesn't exist.")

    task_ids = [line.strip() for line in read_file(dataset_file_path).splitlines() if line.strip()]
    if only_tagged is not None:
        task_ids = [task_id for task_id in task_ids if task_id_to_tag(task_id) in only_tagged]
    task_ids = [remove_tag(task_id) for task_id in task_ids]
    if difficulty is not None:

        def get_task_difficulty(task_id: str) -> int:
            task_metadata_file_path = tasks_directory / task_id / "ground_truth" / "metadata.json"
            metadata = read_json(task_metadata_file_path)
            return metadata["difficulty"]

        task_ids = [task_id for task_id in task_ids if get_task_difficulty(task_id) == difficulty]
    if num_tasks_per_scenario is not None:
        task_ids = [
            task_id for task_id in task_ids if task_id_to_number(task_id) <= num_tasks_per_scenario
        ]
    _maybe_assure_num_tasks_per_scenario(task_ids, num_tasks_per_scenario)
    return task_ids
