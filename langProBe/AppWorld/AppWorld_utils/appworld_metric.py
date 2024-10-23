from .app_world_server_manager import appworld_manager

import dspy

def appworld_metric(example: dspy.Example, pred: dspy.Prediction, target: str = None):
    task_id = example.task_id
    eval_experiment_name = pred.experiment_name_to_eval
    with appworld_manager.acquire_server(eval_experiment_name, task_id) as server:
        eval_report = server.request.evaluate(task_id=task_id, suppress_errors=True, report=False)
        return eval_report['success']