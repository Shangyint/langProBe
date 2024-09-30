import dspy.teleprompt

from ..benchmark import EvaluateBench
from .swe_bench_verified_annotation_task_data import SWEBenchVerifiedAnnotationTaskBench
from .swe_bench_verifier_eval_validity_annotation_program import EvaluationValidityModule

import dspy
import json
import time
import os

expr_dir_results_name = f"langProBe/SweBenchVerifiedAnnotationTask/saved_outputs/eval_validity_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

os.makedirs(expr_dir_results_name, exist_ok=False)

def swe_bench_verified_annotation_evaluate(
    example: dspy.Example, pred: dspy.Prediction, target: str = None
):
    score = 0
    if pred.evaluation_validity_score in example.false_negative:
        score += 1
    
    with open(os.path.join(expr_dir_results_name, f"{example.instance_id}.json"), "w") as f:
        json.dump({"example": {**example}, "pred": {**pred}}, f)

    return score

bench = SWEBenchVerifiedAnnotationTaskBench()
evaluate_naive_program = EvaluateBench(
    bench, EvaluationValidityModule(), swe_bench_verified_annotation_evaluate, optimizer=dspy.teleprompt.BootstrapFewShotWithRandomSearch(metric=swe_bench_verified_annotation_evaluate)
)

with dspy.context(lm=dspy.OpenAI(model="gpt-4o-mini", max_tokens=16000)):
    evaluate_naive_program.evaluate()
