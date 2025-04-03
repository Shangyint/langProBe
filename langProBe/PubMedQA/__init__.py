from typing import Any, Callable

import dspy.datasets
import dspy.evaluate
from .pubmedqa_data import PubMedQABench
from .pubmedqa_program import *
from langProBe.benchmark import BenchmarkMeta
import dspy

benchmark = [
    BenchmarkMeta(
        PubMedQABench,
        [
            PubMedQAPredict,
            PubMedQACoT
        ],
        dspy.evaluate.answer_exact_match,
    )
]