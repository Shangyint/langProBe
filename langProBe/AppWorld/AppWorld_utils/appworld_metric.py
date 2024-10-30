from .app_world_server_manager import appworld_manager

import dspy
import os

def appworld_metric(example: dspy.Example, pred: dspy.Prediction, target: str = None):
    # task_id = example.task_id
    # eval_experiment_name = pred.experiment_name_to_eval
    # if not os.path.exists(os.path.join(os.environ['APPWORLD_ROOT'], "experiments", "outputs", eval_experiment_name, "tasks", task_id)):
    #     raise ValueError(f"Experiment {eval_experiment_name} with task_id {task_id} not found")
    # with appworld_manager.acquire_server(eval_experiment_name, task_id) as server:
    #     eval_report = server.request.evaluate(task_id=task_id, suppress_errors=True, report=False)
    #     print(eval_report)
    #     return eval_report['success']
    return pred.eval_report['success']
