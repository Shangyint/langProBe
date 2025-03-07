from typing import Any, Callable

import dspy.datasets
import dspy.evaluate
from .medmcqa_data import MedMCQABench
from .medmcqa_program import *
from langProBe.benchmark import BenchmarkMeta
import dspy

benchmark = [
    BenchmarkMeta(
        MedMCQABench,
        [
            MedMCQAPredict,
            MedMCQACoT
        ],
        dspy.evaluate.answer_exact_match,
    )
]