from typing import Any, Callable
from .MMLU_data import MMLUBench
from .MMLU_program import (
    MMLUPredict,
    MMLUCoT,
    MMLURAG,
    MMLUSimplifiedBaleen,
    MMLUGeneratorCriticRanker,
    MMLUGeneratorCriticFuser,
)
from langProBe.benchmark import Benchmark, BenchmarkMeta

benchmark: Callable[[], Benchmark] = MMLUBench
# programs = [
#     MMLUPredict,
#     MMLUCoT,
#     MMLURAG,
#     MMLUSimplifiedBaleen,
#     MMLUGeneratorCriticRanker,
#     MMLUGeneratorCriticFuser,
# ]
programs = [
    MMLUPredict
]

def MMLU_metric(gt, pred, trace=None):
    pred_processed = pred.answer.split(".")[0]
    return gt.answer == pred_processed


benchmark = [BenchmarkMeta(MMLUBench, programs, MMLU_metric)]
