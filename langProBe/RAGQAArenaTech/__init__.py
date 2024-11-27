from typing import Any, Callable

import dspy.evaluate
from .RAGQAArenaTech_data import RAGQAArenaBench
from .RAGQAArenaTech_program import RAGQACoT, RAGQAPredict, RAGQARAG
from langProBe.benchmark import BenchmarkMeta
import dspy

benchmark = [
    BenchmarkMeta(
        RAGQAArenaBench, [RAGQACoT, RAGQAPredict, RAGQARAG], dspy.evaluate.SemanticF1()
    )
]
