import dspy.teleprompt

from ..benchmark import EvaluateBench
from .swe_bench_verified_annotation_task_data import SWEBenchVerifiedAnnotationTaskBench
from .swe_bench_verified_underspecified_annotation_program import UnderspecifiedAnnotationGenerator

import dspy
import os
import time
import json

def underspecified_annotation_evaluate(
    example: dspy.Example, pred: dspy.Prediction, target: str = None
):
    score = 0
    if pred.underspecification_score in example.underspecified:
        score += 1

    return score
