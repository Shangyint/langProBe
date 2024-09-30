import dspy.teleprompt

from ..benchmark import EvaluateBench
from .swe_bench_verified_annotation_task_data import SWEBenchVerifiedAnnotationTaskBench
from .swe_bench_verified_underspecified_annotation_program import UnderspecifiedAnnotationGenerator

import dspy
import os
import time
import json

def swe_bench_verified_annotation_evaluate(
    example: dspy.Example, pred: dspy.Prediction, target: str = None
):
    score = 0
    if pred.underspecification_score in example.underspecified:
        score += 1
    
    with open(os.path.join(expr_dir_results_name, f"{example.instance_id}.json"), "w") as f:
        json.dump({"example": {**example}, "pred": {**pred}}, f)

    return score

expr_dir_results_name = f"langProBe/SweBenchVerifiedAnnotationTask/saved_outputs/underspecified_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
os.makedirs(expr_dir_results_name, exist_ok=False)

bench = SWEBenchVerifiedAnnotationTaskBench()
evaluate_naive_program = EvaluateBench(
    bench, UnderspecifiedAnnotationGenerator(), swe_bench_verified_annotation_evaluate, optimizer=dspy.teleprompt.BootstrapFewShotWithRandomSearch(metric=swe_bench_verified_annotation_evaluate)
)

with dspy.context(lm=dspy.OpenAI(model="gpt-4o-mini", max_tokens=16000)):
    evaluate_naive_program.evaluate()
