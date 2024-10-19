from typing import Any, Callable
from .MMLU_data import MMLUBench
from .MMLU_program import CoT, RAG, SimplifiedBaleen
from langProBe.benchmark import Benchmark
import dspy

benchmark: Callable[[], Benchmark] = MMLUBench
programs = [CoT, RAG, SimplifiedBaleen]
# programs = [ SimplifiedBaleen]

def MMLU_metric(gt, pred, trace=None):
    pred_processed = pred.answer.split(".")[0]
    return gt.answer == pred_processed

metric = MMLU_metric