import dspy.teleprompt

from ..benchmark import EvaluateBench
from .swe_bench_verified_annotation_task_data import SWEBenchVerifiedAnnotationTaskBench
from .swe_bench_verifier_eval_validity_annotation_program import EvaluationValidityModule

import dspy
import json
import time
import os

def evaluation_validity_evaluate(
    example: dspy.Example, pred: dspy.Prediction, target: str = None
):
    score = 0
    if pred.evaluation_validity_score in example.false_negative:
        score += 1

    return score
