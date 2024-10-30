from typing import Any, Callable

import dspy.evaluate
from .RAGQAArenaTech_data import RAGQAArenaBench
from .RAGQAArenaTech_program import CoT, RAG
from langProBe.benchmark import BenchmarkMeta
import dspy

benchmark = [BenchmarkMeta(RAGQAArenaBench, [CoT(), RAG()], dspy.evaluate.SemanticF1())]
