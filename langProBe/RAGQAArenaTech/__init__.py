from typing import Any, Callable

import dspy.evaluate
from .RAGQAArenaTech_data import RAGQAArenaBench
from .RAGQAArenaTech_program import CoT
from langProBe.benchmark import BenchmarkMeta
import dspy

benchmark = [BenchmarkMeta(RAGQAArenaBench, [CoT], dspy.evaluate.SemanticF1())]
