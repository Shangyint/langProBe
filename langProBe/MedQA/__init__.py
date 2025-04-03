from typing import Any, Callable

import dspy.datasets
import dspy.evaluate
from .medqa_data import MedQABench
from .medqa_program import *
from langProBe.benchmark import BenchmarkMeta
import dspy

benchmark = [
    BenchmarkMeta(
        MedQABench,
        [
            MedQAPredict,
            MedQACoT
        ],
        dspy.evaluate.answer_exact_match,
    )
]